"""Run the LiveKit worker that hears with Deepgram and speaks with ElevenLabs."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable, Coroutine, Sequence
from typing import cast

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobExecutorType,
    cli,
)
from livekit.agents.llm import ChatContext, ChatMessage, LLMError
from livekit.agents.stt import STTError
from livekit.agents.tts import TTSError
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    ErrorEvent,
    UserInputTranscribedEvent,
)

from sarjy_agent.backend import (
    AuthorizedSession,
    BackendClient,
    BackendError,
    TranscriptStore,
    create_backend,
)
from sarjy_agent.config import Settings, load_settings
from sarjy_agent.context import bound_chat, drop_repeated_user_line
from sarjy_agent.dispatch import (
    AGENT_NAME,
    AssignedRoom,
    bind_session,
    conversation_id_from_job,
    handle_job_request,
)
from sarjy_agent.instructions import SYSTEM_PROMPT, append_user_facts
from sarjy_agent.lifecycle import SessionLifecycle
from sarjy_agent.llm import create_llm
from sarjy_agent.log import (
    configure_logging,
    log_context,
    log_provider_failure,
    log_user_turn,
    public_id,
    register_secrets,
)
from sarjy_agent.segments import StreamTimingLog
from sarjy_agent.stt import create_code_switch_stt
from sarjy_agent.tools import FALLBACK_LINE, build_voice_tools, memory_fact_lines
from sarjy_agent.tts import create_tts
from sarjy_agent.turn_detection import create_vad, turn_handling
from sarjy_agent.turns import UserTurns

_CLASS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class AuthorizationRejected(Exception):
    """The backend did not authorize this room. The voice session does not start."""


class Assistant(Agent):
    """The Sarjy voice agent.

    It keeps a short history for this call.
    Tools are attached only for an authorized session.
    When a transcript store is attached, each finished line is saved once.
    """

    def __init__(
        self,
        *,
        user_facts: Sequence[str] = (),
        transcripts: TranscriptStore | None = None,
        tools: list[object] | None = None,
    ) -> None:
        """Set the Saudi Arabic instructions and an empty turn list."""
        facts = ChatContext.empty()
        append_user_facts(facts, user_facts)
        super().__init__(
            instructions=SYSTEM_PROMPT,
            chat_ctx=facts,
            tools=tools,  # type: ignore[arg-type]
        )
        self.turns = UserTurns()
        self._transcripts = transcripts

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Save the finished user turn, then keep only the latest messages."""
        text = new_message.text_content or ""
        turn = self.turns.commit(text)
        if turn is not None:
            log_user_turn(turn.turn_id)
            if self._transcripts is not None:
                await self._transcripts.save_user(turn)
        drop_repeated_user_line(turn_ctx, text)
        bound_chat(turn_ctx)
        stored = self.chat_ctx.copy()
        drop_repeated_user_line(stored, text)
        bound_chat(stored)
        await self.update_chat_ctx(stored, exclude_invalid_function_calls=False)


def create_server(settings: Settings) -> AgentServer:
    """Build the worker that joins rooms as the sarjy agent."""
    server = AgentServer(
        ws_url=_livekit_url(settings),
        api_key=settings.livekit_api_key.get_secret_value(),
        api_secret=settings.livekit_api_secret.get_secret_value(),
        job_executor_type=_job_executor_type(),
    )
    server.rtc_session(
        session_entrypoint,
        agent_name=AGENT_NAME,
        on_request=handle_job_request,
    )
    return server


async def session_entrypoint(ctx: JobContext) -> None:
    """Join the job's room, run one voice session, and leave on shutdown."""
    configure_logging()
    assigned = bind_session(ctx.job, ctx.room.name)
    if assigned is None:
        ctx.shutdown("dispatch_rejected")
        return

    async def connect() -> None:
        """Start the voice session, or stop when the backend rejects the room."""
        try:
            await _start_voice_session(ctx, assigned)
        except BackendError:
            ctx.shutdown("authorization_rejected")
            raise AuthorizationRejected from None

    def register_shutdown(
        callback: Callable[[str], Coroutine[None, None, None]],
    ) -> None:
        """Ask LiveKit to call this function when the job stops."""
        ctx.add_shutdown_callback(callback)

    lifecycle = SessionLifecycle(
        room=ctx.room,
        connect=connect,
        add_shutdown_callback=register_shutdown,
        room_id=public_id(assigned.room_sid) or public_id(assigned.room_name),
        session_id=public_id(assigned.job_id),
        stop_errors=(AuthorizationRejected,),
    )
    await lifecycle.run()


def create_voice_session(settings: Settings) -> AgentSession[None]:
    """Make one room session. Hearing, speech, and the end of a turn are set here."""
    return AgentSession(
        stt=create_code_switch_stt(settings),
        llm=create_llm(settings),
        tts=create_tts(settings),
        vad=create_vad(),
        turn_handling=turn_handling(),
    )


async def _start_voice_session(ctx: JobContext, assigned: AssignedRoom) -> None:
    """Authorize the room, then watch transcripts and start the agent."""
    settings = load_settings()
    register_secrets(settings.secret_values())
    client = create_backend(settings)
    conversation_id = conversation_id_from_job(getattr(ctx, "job", None))
    try:
        voice = await _authorize(assigned, client, conversation_id)
    except BackendError:
        await client.aclose()
        raise
    facts: tuple[str, ...] = ()
    tool_list: list[object] = []
    if isinstance(client, BackendClient):
        facts = await _load_memory(client, voice)
        tool_list = build_voice_tools(client, voice)
    store = TranscriptStore(client, voice)
    agent = Assistant(user_facts=facts, transcripts=store, tools=tool_list)

    async def close_backend(_reason: str) -> None:
        """Finish transcript writes, then close the Next.js client."""
        await store.flush()
        await client.aclose()

    ctx.add_shutdown_callback(close_backend)
    session = create_voice_session(settings)
    _watch_transcripts(session, agent.turns, store)
    await session.start(agent=agent, room=ctx.room)


async def _authorize(
    assigned: AssignedRoom,
    client: object,
    conversation_id: str | None,
) -> AuthorizedSession:
    """Ask Next.js which user owns this conversation and room."""
    room_id = public_id(assigned.room_sid) or public_id(assigned.room_name)
    with log_context(room_id=room_id, session_id=public_id(assigned.job_id)):
        if isinstance(client, BackendClient):
            if conversation_id is None:
                log_provider_failure(
                    provider="nextjs",
                    operation="authorize",
                    error_type="MissingConversation",
                )
                raise BackendError("MissingConversation")
            return await client.authorize(assigned, conversation_id)
        result = await client.authorize(assigned, conversation_id)  # type: ignore[attr-defined]
        if not isinstance(result, AuthorizedSession):
            raise BackendError("InvalidResponse")
        return result


async def _load_memory(
    client: BackendClient, voice: AuthorizedSession
) -> tuple[str, ...]:
    """Load saved preferences. A failure leaves the call with no facts."""
    try:
        facts = await client.list_memory(voice.user_id)
    except BackendError as exc:
        log_provider_failure(
            provider="nextjs",
            operation="list_memory",
            error_type=exc.error_type,
        )
        return ()
    return memory_fact_lines(facts)


def _watch_transcripts(
    session: AgentSession[None],
    turns: UserTurns,
    transcripts: TranscriptStore | None = None,
) -> None:
    """Listen for transcripts, finished replies, errors, and reply latency."""
    timing = StreamTimingLog()
    fallback = _Fallback()
    session.on("user_input_transcribed", lambda event: _on_transcript(turns, event))
    session.on("error", lambda event: _on_error(session, turns, fallback, event))
    if transcripts is not None:
        session.on(
            "conversation_item_added",
            lambda event: _on_assistant_item(transcripts, turns, event),
        )
    # AgentSession.on warns that this event is deprecated. Registering on the
    # parent still receives the latency numbers, without that warning.
    rtc.EventEmitter.on(
        cast(rtc.EventEmitter[str], session),
        "metrics_collected",
        timing.observe_event,
    )


def _on_transcript(turns: UserTurns, event: UserInputTranscribedEvent) -> None:
    """Pass one Deepgram transcript into the open user turn."""
    turns.observe(
        text=event.transcript,
        is_final=event.is_final,
        speaker_id=event.speaker_id,
    )


def _on_assistant_item(
    transcripts: TranscriptStore,
    turns: UserTurns,
    event: ConversationItemAddedEvent,
) -> None:
    """Save one finished assistant line. A partial line is not saved here."""
    item = event.item
    if not isinstance(item, ChatMessage) or item.role != "assistant":
        return
    turn_id = turns.finalized[-1].turn_id if turns.finalized else "t0"
    transcripts.schedule_assistant(
        message_id=item.id,
        turn_id=turn_id,
        text=item.text_content or "",
        interrupted=item.interrupted,
    )


def _on_error(
    session: AgentSession[None],
    turns: UserTurns,
    fallback: _Fallback,
    event: ErrorEvent,
) -> None:
    """Log a provider failure. Speak one short line when the reply cannot continue."""
    error = event.error
    if isinstance(error, STTError):
        turns.fail()
        log_provider_failure(
            provider="deepgram",
            operation="transcribe",
            error_type=_error_type(error.error),
        )
        return
    if isinstance(error, LLMError):
        _recover(
            session, fallback, "openai", "generate", error.error, error.recoverable
        )
        return
    if isinstance(error, TTSError):
        _recover(
            session, fallback, "elevenlabs", "speak", error.error, error.recoverable
        )


def _recover(
    session: AgentSession[None],
    fallback: _Fallback,
    provider: str,
    operation: str,
    error: BaseException,
    recoverable: bool,
) -> None:
    """Log the failure. A cancelled call does not speak. One fallback is enough."""
    log_provider_failure(
        provider=provider,
        operation=operation,
        error_type=_error_type(error),
    )
    if recoverable or isinstance(error, asyncio.CancelledError):
        return
    fallback.speak(session)


class _Fallback:
    """Speak the fallback line at most once in a call."""

    def __init__(self) -> None:
        """Start with the fallback unsaid."""
        self._said = False

    def speak(self, session: AgentSession[None]) -> None:
        """Say one short line. A second call does nothing."""
        if self._said:
            return
        self._said = True
        try:
            session.say(FALLBACK_LINE, allow_interruptions=True)
        except Exception as exc:
            log_provider_failure(
                provider="livekit",
                operation="fallback",
                error_type=_error_type(exc),
            )


def _error_type(exc: BaseException) -> str:
    """Return a short error class name that is safe to log."""
    name = type(exc).__name__
    if _CLASS_NAME.fullmatch(name):
        return name
    return "Error"


def run_worker(settings: Settings) -> None:
    """Start the worker. Use the start command when none was given."""
    if len(sys.argv) == 1:
        sys.argv.append("start")
    cli.run_app(create_server(settings))


def _job_executor_type() -> JobExecutorType:
    """Run jobs in threads during dev or debugging, otherwise in processes."""
    if "dev" in sys.argv[1:]:
        return JobExecutorType.THREAD
    if sys.gettrace() is not None or "debugpy" in sys.modules:
        return JobExecutorType.THREAD
    return JobExecutorType.PROCESS


def _livekit_url(settings: Settings) -> str:
    """Return the LiveKit websocket URL without a trailing slash."""
    rendered = str(settings.livekit_url)
    if settings.livekit_url.path in ("", "/") and rendered.endswith("/"):
        return rendered[:-1]
    return rendered

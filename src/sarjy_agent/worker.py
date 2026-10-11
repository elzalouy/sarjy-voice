"""Run the LiveKit worker that hears with Deepgram and speaks with ElevenLabs."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Coroutine

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobExecutorType,
    cli,
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.stt import STTError
from livekit.agents.voice.events import ErrorEvent, UserInputTranscribedEvent

from sarjy_agent.config import Settings, load_settings
from sarjy_agent.dispatch import AGENT_NAME, bind_session, handle_job_request
from sarjy_agent.lifecycle import SessionLifecycle
from sarjy_agent.log import (
    configure_logging,
    log_provider_failure,
    log_user_turn,
    public_id,
    register_secrets,
)
from sarjy_agent.stt import create_stt
from sarjy_agent.tts import create_tts
from sarjy_agent.turns import UserTurns

_INSTRUCTIONS = "You are the Sarjy voice assistant. Keep spoken replies short."
_CLASS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class Assistant(Agent):
    """The Sarjy voice agent. It has no tools and no memory."""

    def __init__(self) -> None:
        """Set the short spoken instructions and an empty turn list."""
        super().__init__(instructions=_INSTRUCTIONS)
        self.turns = UserTurns()

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Save the finished user turn once, then log its id."""
        del turn_ctx
        text = new_message.text_content or ""
        turn = self.turns.commit(text)
        if turn is not None:
            log_user_turn(turn.turn_id)


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

    def register_shutdown(
        callback: Callable[[str], Coroutine[None, None, None]],
    ) -> None:
        """Ask LiveKit to call this function when the job stops."""
        ctx.add_shutdown_callback(callback)

    lifecycle = SessionLifecycle(
        room=ctx.room,
        connect=lambda: _start_voice_session(ctx),
        add_shutdown_callback=register_shutdown,
        room_id=public_id(assigned.room_sid) or public_id(assigned.room_name),
        session_id=public_id(assigned.job_id),
    )
    await lifecycle.run()


def create_voice_session(settings: Settings) -> AgentSession[None]:
    """Make one room session. Hearing and speech come from the current settings."""
    return AgentSession(stt=create_stt(settings), tts=create_tts(settings))


async def _start_voice_session(ctx: JobContext) -> None:
    """Load settings, watch transcripts, and start the agent in this room."""
    settings = load_settings()
    register_secrets(settings.secret_values())
    agent = Assistant()
    session = create_voice_session(settings)
    _watch_transcripts(session, agent.turns)
    await session.start(agent=agent, room=ctx.room)


def _watch_transcripts(session: AgentSession[None], turns: UserTurns) -> None:
    """Listen for transcripts and recognition errors before audio starts."""
    session.on("user_input_transcribed", lambda event: _on_transcript(turns, event))
    session.on("error", lambda event: _on_error(turns, event))


def _on_transcript(turns: UserTurns, event: UserInputTranscribedEvent) -> None:
    """Pass one Deepgram transcript into the open user turn."""
    turns.observe(
        text=event.transcript,
        is_final=event.is_final,
        speaker_id=event.speaker_id,
    )


def _on_error(turns: UserTurns, event: ErrorEvent) -> None:
    """Drop a partial line and log a Deepgram failure. Other errors are ignored."""
    error = event.error
    if not isinstance(error, STTError):
        return
    turns.fail()
    log_provider_failure(
        provider="deepgram",
        operation="transcribe",
        error_type=_error_type(error.error),
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

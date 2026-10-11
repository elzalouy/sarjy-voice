"""Next.js authorization and transcript writes. No live speech calls."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO

import httpx
import pytest
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
)
from livekit.protocol.agent import Job
from livekit.protocol.models import ParticipantInfo, Room

from conftest import LEAK_MARKER, VALID_ENV
from sarjy_agent.backend import (
    AUTHORIZE_PATH,
    AuthorizedSession,
    BackendClient,
    BackendError,
    TranscriptStore,
    authorized_user_id,
    transcript_path,
)
from sarjy_agent.config import Settings, load_settings
from sarjy_agent.dispatch import AGENT_NAME, AssignedRoom
from sarjy_agent.log import configure_logging, register_secrets
from sarjy_agent.turns import FinalUserTurn
from sarjy_agent.worker import _on_transcript, _start_voice_session, session_entrypoint

_SPEECH = "السلام عليكم"
_TOOL_USER = "user-from-tool"
_LIVEKIT_USER = "user-from-livekit"
_USER = "11111111-1111-4111-8111-111111111111"
_CONVERSATION = "22222222-2222-4222-8222-222222222222"
_MESSAGE = "33333333-3333-4333-8333-333333333333"


def test_authorize_uses_the_room_and_ignores_a_tool_user(
    valid_env: None,
) -> None:
    calls = _Calls()

    async def scenario() -> AuthorizedSession:
        client = _client(calls, lambda: _active())
        try:
            return await client.authorize(_assigned(), _CONVERSATION)
        finally:
            await client.aclose()

    session = asyncio.run(scenario())
    body = calls.bodies[0]

    assert calls.paths == [AUTHORIZE_PATH]
    assert body == {
        "conversationId": _CONVERSATION,
        "roomName": "test-room",
    }
    assert calls.authorizations == [f"Bearer {VALID_ENV['AGENT_SERVICE_TOKEN']}"]
    assert calls.request_ids[0]
    assert session.user_id == _USER
    assert session.conversation_id == _CONVERSATION
    assert authorized_user_id(session, _TOOL_USER) == _USER
    assert _TOOL_USER not in json.dumps(body)


def test_a_closed_conversation_does_not_bind_a_session(valid_env: None) -> None:
    calls = _Calls()
    stream = StringIO()
    configure_logging(stream)

    async def scenario() -> str:
        client = _client(calls, lambda: {"error": "closed"}, status=409)
        try:
            await client.authorize(_assigned(), _CONVERSATION)
        except BackendError as exc:
            return exc.error_type
        finally:
            await client.aclose()
        return ""

    error_type = asyncio.run(scenario())
    logs = _logs(stream)

    assert error_type == "Rejected"
    assert logs[-1]["provider"] == "nextjs"
    assert logs[-1]["operation"] == "authorize"
    assert logs[-1]["error_type"] == "Rejected"
    assert logs[-1]["request_id"] == calls.request_ids[0]
    assert _SPEECH not in stream.getvalue()
    assert LEAK_MARKER not in stream.getvalue()


def test_an_invalid_authorize_body_is_rejected(valid_env: None) -> None:
    calls = _Calls()

    async def scenario() -> str:
        client = _client(calls, lambda: {"user_id": "user real", "status": "active"})
        try:
            await client.authorize(_assigned(), _CONVERSATION)
        except BackendError as exc:
            return exc.error_type
        finally:
            await client.aclose()
        return ""

    assert asyncio.run(scenario()) == "InvalidResponse"
    assert len(calls.paths) == 1


def test_unauthorized_and_redirects_are_not_retried(valid_env: None) -> None:
    denied = _Calls()
    redirected = _Calls()

    async def scenario() -> tuple[str, str]:
        first = _client(denied, lambda: {}, status=401)
        second = _client(
            redirected,
            lambda: {},
            status=302,
            headers={"Location": "http://evil.example/collect"},
        )
        try:
            with pytest.raises(BackendError) as denied_error:
                await first.authorize(_assigned(), _CONVERSATION)
            with pytest.raises(BackendError) as redirect_error:
                await second.authorize(_assigned(), _CONVERSATION)
        finally:
            await first.aclose()
            await second.aclose()
        return denied_error.value.error_type, redirect_error.value.error_type

    denied_type, redirect_type = asyncio.run(scenario())

    assert denied_type == "Unauthorized"
    assert redirect_type == "Rejected"
    assert len(denied.paths) == 1
    assert len(redirected.paths) == 1
    assert "evil.example" not in json.dumps(redirected.paths)


def test_a_server_error_is_retried_with_the_same_event_id(valid_env: None) -> None:
    calls = _Calls()
    stream = StringIO()
    configure_logging(stream)
    replies = {"left": 1}

    def body() -> tuple[int, dict[str, object]]:
        if replies["left"]:
            replies["left"] -= 1
            return 503, {"error": VALID_ENV["AGENT_SERVICE_TOKEN"], "text": _SPEECH}
        return 201, _message(calls.bodies[-1])

    async def scenario() -> None:
        client = _client(calls, body)
        store = _store(client)
        try:
            saved = await store.save_user(_turn("t1", _SPEECH))
            again = await store.save_user(_turn("t1", _SPEECH))
            assert saved is True
            assert again is False
        finally:
            await client.aclose()

    asyncio.run(scenario())
    transcripts = [body for body in calls.bodies if body.get("role") == "user"]

    assert calls.paths == [
        transcript_path(_CONVERSATION),
        transcript_path(_CONVERSATION),
    ]
    assert len(transcripts) == 2
    assert transcripts[0]["eventId"] == transcripts[1]["eventId"]
    assert transcripts[0]["eventId"] == f"{_CONVERSATION}:user:t1"
    assert calls.request_ids[-2] != calls.request_ids[-1]
    assert transcripts[0]["userId"] == _USER
    assert _TOOL_USER not in json.dumps(transcripts)
    assert LEAK_MARKER not in stream.getvalue()
    assert _SPEECH not in stream.getvalue()


def test_a_timeout_is_bounded_and_hides_the_response(valid_env: None) -> None:
    stream = StringIO()
    configure_logging(stream)
    attempts = 0

    def explode(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout(
            f"{VALID_ENV['AGENT_SERVICE_TOKEN']} {_SPEECH}",
            request=request,
        )

    async def scenario() -> str:
        settings = load_settings()
        client = BackendClient(
            settings,
            transport=httpx.MockTransport(explode),
            retry_pause_seconds=0,
            timeout_seconds=0.2,
        )
        try:
            await client.authorize(_assigned(), _CONVERSATION)
        except BackendError as exc:
            return str(exc)
        finally:
            await client.aclose()
        return ""

    message = asyncio.run(scenario())

    assert message == "Timeout"
    assert attempts == 2
    assert LEAK_MARKER not in message
    assert _SPEECH not in message
    assert LEAK_MARKER not in stream.getvalue()
    assert _SPEECH not in stream.getvalue()
    assert [item["error_type"] for item in _logs(stream)] == ["Timeout"]


def test_finished_lines_keep_order_and_status(valid_env: None) -> None:
    calls = _Calls()

    async def scenario() -> None:
        client = _client(calls, lambda: _stored(calls))
        store = _store(client)
        try:
            await store.save_user(_turn("t1", _SPEECH))
            await store.save_assistant(
                message_id="item_done1",
                turn_id="t1",
                text="وعليكم السلام",
                interrupted=False,
            )
            await store.save_assistant(
                message_id="item_cut01",
                turn_id="t2",
                text="كنت بقول",
                interrupted=True,
            )
            await store.save_assistant(
                message_id="item_empty",
                turn_id="t2",
                text="   ",
                interrupted=False,
            )
        finally:
            await client.aclose()

    asyncio.run(scenario())
    lines = calls.bodies

    assert [item["role"] for item in lines] == ["user", "assistant", "assistant"]
    assert [item["content"] for item in lines] == [_SPEECH, "وعليكم السلام", "كنت بقول"]
    assert "status" not in lines[2]
    assert lines[0]["eventId"] == f"{_CONVERSATION}:user:t1"
    assert lines[1]["eventId"] == f"{_CONVERSATION}:assistant:item_done1"
    assert lines[2]["eventId"] == f"{_CONVERSATION}:assistant:item_cut01"


def test_a_conflict_counts_as_one_saved_line(valid_env: None) -> None:
    calls = _Calls()

    async def scenario() -> None:
        client = _client(calls, lambda: _stored(calls))
        store = _store(client)
        try:
            assert await store.save_user(_turn("t1", _SPEECH)) is True
            assert await store.save_user(_turn("t1", _SPEECH)) is False
        finally:
            await client.aclose()

    asyncio.run(scenario())

    assert calls.paths == [transcript_path(_CONVERSATION)]
    assert calls.bodies[0]["eventId"] == f"{_CONVERSATION}:user:t1"


def test_a_local_endpoint_checks_the_service_token(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = VALID_ENV["AGENT_SERVICE_TOKEN"]
    seen: list[dict[str, object]] = []
    server = _local_server(token, seen)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("NEXTJS_API_BASE_URL", f"http://127.0.0.1:{port}")

    async def scenario() -> AuthorizedSession:
        client = BackendClient(load_settings(), retry_pause_seconds=0, max_attempts=1)
        try:
            return await client.authorize(_assigned(), _CONVERSATION)
        finally:
            await client.aclose()

    try:
        session = asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert session.user_id == _USER
    assert session.conversation_id == _CONVERSATION
    assert seen[0]["path"] == AUTHORIZE_PATH
    assert seen[0]["authorization"] == f"Bearer {token}"
    assert isinstance(seen[0]["request_id"], str) and seen[0]["request_id"]
    assert seen[0]["body"] == {
        "conversationId": _CONVERSATION,
        "roomName": "test-room",
    }


def test_a_rejected_room_does_not_start_the_voice_session(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = StringIO()
    configure_logging(stream)
    calls = _Calls()
    started: list[object] = []

    def factory(settings: Settings) -> BackendClient:
        return BackendClient(
            settings,
            transport=httpx.MockTransport(
                calls.respond(lambda: {"error": "closed"}, status=409)
            ),
            retry_pause_seconds=0,
        )

    def voice_session(_settings: object) -> object:
        started.append("session")
        raise AssertionError("voice session started")

    monkeypatch.setattr("sarjy_agent.worker.create_backend", factory)
    monkeypatch.setattr("sarjy_agent.worker.create_voice_session", voice_session)
    context = _JobContext()

    async def scenario() -> None:
        await session_entrypoint(context)  # type: ignore[arg-type]

    asyncio.run(scenario())
    logs = _logs(stream)

    assert started == []
    assert context.reason == "authorization_rejected"
    assert "session_started" not in [item["event"] for item in logs]
    failure = logs[-1]
    assert failure["event"] == "provider_failure"
    assert failure["provider"] == "nextjs"
    assert failure["operation"] == "authorize"
    assert failure["error_type"] == "Rejected"
    assert failure["room_id"] == "RM_testroom"
    assert failure["session_id"] == "AJ_job1"
    assert _LIVEKIT_USER not in stream.getvalue()
    assert LEAK_MARKER not in stream.getvalue()


def test_a_call_saves_the_finished_lines_once(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = StringIO()
    configure_logging(stream)
    register_secrets([VALID_ENV["AGENT_SERVICE_TOKEN"]])
    calls = _Calls()
    sessions: list[_FakeSession] = []

    def reply() -> dict[str, object]:
        path = calls.paths[-1]
        if path == AUTHORIZE_PATH:
            return _active()
        if path.endswith("/memory"):
            return {
                "memories": [
                    {
                        "key": "preferred_name",
                        "value": "نورة",
                        "updatedAt": "2026-10-11T09:00:00.000Z",
                    }
                ]
            }
        return _message(calls.bodies[-1])

    def factory(settings: Settings) -> BackendClient:
        return BackendClient(
            settings,
            transport=httpx.MockTransport(calls.respond(reply)),
            retry_pause_seconds=0,
        )

    monkeypatch.setattr("sarjy_agent.worker.create_backend", factory)
    monkeypatch.setattr("sarjy_agent.worker.AgentSession", _capturing_session(sessions))
    context = _JobContext()

    async def scenario() -> None:
        await _start_voice_session(context, _assigned())  # type: ignore[arg-type]
        agent = sessions[0].agent
        assert agent is not None
        _on_transcript(
            agent.turns,
            UserInputTranscribedEvent(
                transcript="السلام",
                is_final=False,
                speaker_id="S0",
            ),
        )
        assert [item.get("role") for item in calls.bodies] == [None, None]
        await agent.on_user_turn_completed(
            ChatContext.empty(),
            ChatMessage(role="user", content=[_SPEECH]),
        )
        await agent.on_user_turn_completed(
            ChatContext.empty(),
            ChatMessage(role="user", content=[_SPEECH]),
        )
        sessions[0].emit(
            "conversation_item_added",
            ConversationItemAddedEvent(
                item=ChatMessage(role="user", content=[_SPEECH], id="item_user01")
            ),
        )
        sessions[0].emit(
            "conversation_item_added",
            ConversationItemAddedEvent(
                item=ChatMessage(
                    role="assistant",
                    content=["وعليكم السلام"],
                    id="item_reply1",
                )
            ),
        )
        sessions[0].emit(
            "conversation_item_added",
            ConversationItemAddedEvent(
                item=ChatMessage(
                    role="assistant",
                    content=["وعليكم السلام"],
                    id="item_reply1",
                )
            ),
        )
        await agent._transcripts.flush()  # type: ignore[union-attr]
        for callback in context.callbacks:
            await callback("shutdown")

    asyncio.run(scenario())
    lines = [body for body in calls.bodies if "role" in body]

    assert calls.paths[0] == AUTHORIZE_PATH
    assert calls.bodies[0]["roomName"] == "test-room"
    assert calls.bodies[0]["conversationId"] == _CONVERSATION
    assert "userId" not in calls.bodies[0]
    assert _LIVEKIT_USER not in json.dumps(calls.bodies[0])
    assert [item["role"] for item in lines] == ["user", "assistant"]
    assert [item["content"] for item in lines] == [_SPEECH, "وعليكم السلام"]
    assert lines[0]["userId"] == _USER
    assert lines[1]["userId"] == _USER
    assert lines[0]["eventId"] == f"{_CONVERSATION}:user:t1"
    assert lines[1]["eventId"] == f"{_CONVERSATION}:assistant:item_reply1"
    saved_agent = sessions[0].agent
    assert saved_agent is not None
    assert any(
        "نورة" in (item.text_content or "")
        for item in saved_agent.chat_ctx.messages()  # type: ignore[attr-defined]
    )
    assert _SPEECH not in stream.getvalue()
    assert LEAK_MARKER not in stream.getvalue()
    assert "user_turn_final" in [item["event"] for item in _logs(stream)]


class _Calls:
    """Record each mocked Next.js request."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.bodies: list[dict[str, object]] = []
        self.authorizations: list[str | None] = []
        self.request_ids: list[str | None] = []
        self.idempotency_keys: list[str | None] = []

    def respond(
        self,
        reply: Callable[[], tuple[int, dict[str, object]] | dict[str, object]],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> Callable[[httpx.Request], httpx.Response]:
        """Build a transport handler that stores the request, then answers."""

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content) if request.content else {}
            assert isinstance(payload, dict)
            self.paths.append(request.url.path)
            self.bodies.append(payload)
            self.authorizations.append(request.headers.get("authorization"))
            self.request_ids.append(request.headers.get("x-request-id"))
            self.idempotency_keys.append(request.headers.get("idempotency-key"))
            result = reply()
            if isinstance(result, tuple):
                code, document = result
            else:
                code, document = status, result
            return httpx.Response(code, json=document, headers=headers)

        return handler


class _FakeSession:
    """Stand-in for a room session. It records callbacks and the agent."""

    def __init__(self, **_kwargs: object) -> None:
        self.callbacks: dict[str, list[Callable[..., object]]] = {}
        self.agent: object | None = None
        self._events: dict[str, dict[object, None]] = {}

    def on(self, event: str, callback: Callable[..., object]) -> None:
        self.callbacks.setdefault(event, []).append(callback)

    async def start(self, *, agent: object, room: object) -> None:
        self.agent = agent
        del room

    def emit(self, event: str, payload: object) -> None:
        for callback in self.callbacks.get(event, []):
            callback(payload)


def _capturing_session(
    found: list[_FakeSession],
) -> type[_FakeSession]:
    class _Session(_FakeSession):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            found.append(self)

    return _Session


class _JobContext:
    """The job fields the entrypoint reads."""

    def __init__(self) -> None:
        self.job = Job(
            id="AJ_job1",
            dispatch_id="AD_dispatch1",
            agent_name=AGENT_NAME,
            room=Room(name="test-room", sid="RM_testroom"),
            participant=ParticipantInfo(identity=_LIVEKIT_USER, name="Ada"),
            metadata='{"conversationId":"22222222-2222-4222-8222-222222222222","userId":"11111111-1111-4111-8111-111111111111"}',
        )
        self.room = _ConnectedRoom()
        self.reason: str | None = None
        self.callbacks: list[Callable[[str], object]] = []

    def add_shutdown_callback(self, callback: Callable[[str], object]) -> None:
        self.callbacks.append(callback)

    def shutdown(self, reason: str = "") -> None:
        self.reason = reason


class _ConnectedRoom:
    """Room events used while a visit starts and stops."""

    def __init__(self) -> None:
        self.name = "test-room"
        self.handlers: dict[str, list[Callable[..., object]]] = {}
        self.remote_participants: dict[str, object] = {}

    def on(self, event: str, callback: Callable[..., object]) -> object:
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def off(self, event: str, callback: Callable[..., object]) -> None:
        callbacks = self.handlers.get(event, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def isconnected(self) -> bool:
        return False

    async def disconnect(self) -> None:
        return None


def _client(
    calls: _Calls,
    reply: Callable[[], tuple[int, dict[str, object]] | dict[str, object]],
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> BackendClient:
    settings = load_settings()
    return BackendClient(
        settings,
        transport=httpx.MockTransport(
            calls.respond(reply, status=status, headers=headers)
        ),
        retry_pause_seconds=0,
    )


def _store(client: BackendClient) -> TranscriptStore:
    return TranscriptStore(
        client,
        AuthorizedSession(
            user_id=_USER,
            conversation_id=_CONVERSATION,
            room_name="test-room",
            room_sid="RM_testroom",
            job_id="AJ_job1",
            dispatch_id="AD_dispatch1",
        ),
    )


def _assigned() -> AssignedRoom:
    return AssignedRoom(
        job_id="AJ_job1",
        dispatch_id="AD_dispatch1",
        room_name="test-room",
        room_sid="RM_testroom",
    )


def _turn(turn_id: str, text: str) -> FinalUserTurn:
    return FinalUserTurn(turn_id=turn_id, speaker_id="S0", text=text)


def _active() -> dict[str, object]:
    return {
        "conversationId": _CONVERSATION,
        "userId": _USER,
        "roomName": "test-room",
    }


def _stored(calls: _Calls) -> dict[str, object]:
    return _message(calls.bodies[-1])


def _message(body: dict[str, object]) -> dict[str, object]:
    return {
        "message": {
            "id": _MESSAGE,
            "role": body["role"],
            "content": body["content"],
            "createdAt": "2026-10-11T09:00:00.000Z",
        }
    }


def _logs(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def _local_server(token: str, seen: list[dict[str, object]]) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            seen.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "request_id": self.headers.get("X-Request-Id"),
                    "body": body,
                }
            )
            allowed = self.headers.get("Authorization") == f"Bearer {token}"
            if allowed:
                payload = json.dumps(_active()).encode()
                code = 200
            else:
                payload = b'{"error":"unauthorized"}'
                code = 401
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return None

    return ThreadingHTTPServer(("127.0.0.1", 0), _Handler)

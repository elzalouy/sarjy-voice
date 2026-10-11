import asyncio
import json
from collections.abc import Callable
from io import StringIO

import pytest
from livekit.agents import JobRequest
from livekit.agents.job import JobAcceptArguments
from livekit.api import AccessToken, RoomConfiguration, TokenVerifier, VideoGrants
from livekit.protocol.agent import Job
from livekit.protocol.models import ParticipantInfo, Room

from conftest import LEAK_MARKER
from sarjy_agent.config import load_settings
from sarjy_agent.dispatch import (
    AGENT_NAME,
    AssignedRoom,
    bind_session,
    browser_dispatch,
    conversation_id_from_job,
    dispatch_request,
    handle_job_request,
)
from sarjy_agent.log import configure_logging
from sarjy_agent.worker import session_entrypoint

_USER_ID = "user-123"
_OTHER_ROOM = "other-room"
_METADATA = f'{{"user_id":"{_USER_ID}","room":"{_OTHER_ROOM}"}}'


class _Room:
    def __init__(self, name: str, sid: str = "") -> None:
        self.name = name
        self.sid = sid
        self.metadata = LEAK_MARKER
        self.handlers: dict[str, list[Callable[..., object]]] = {}
        self.connected = False
        self.remote_participants: dict[str, object] = {}

    def on(self, event: str, callback: Callable[..., object]) -> object:
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def off(self, event: str, callback: Callable[..., object]) -> None:
        callbacks = self.handlers.get(event, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def isconnected(self) -> bool:
        return self.connected

    async def disconnect(self) -> None:
        self.connected = False


class _Context:
    def __init__(self, job: Job, room_name: str) -> None:
        self.job = job
        self.room = _Room(room_name)
        self.reason: str | None = None

    def add_shutdown_callback(self, _callback: object) -> None:
        return None

    def shutdown(self, reason: str = "") -> None:
        self.reason = reason


def _job(
    *,
    agent_name: str = AGENT_NAME,
    job_id: str = "AJ_job1",
    dispatch_id: str = "AD_dispatch1",
    room_name: str = "test-room",
    room_sid: str = "RM_testroom",
    metadata: str = _METADATA,
) -> Job:
    return Job(
        id=job_id,
        dispatch_id=dispatch_id,
        agent_name=agent_name,
        room=Room(name=room_name, sid=room_sid, metadata=LEAK_MARKER),
        metadata=metadata,
        participant=ParticipantInfo(
            identity=_USER_ID,
            name="Ada",
            metadata=LEAK_MARKER,
        ),
    )


def _logs(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


async def _answer(job: Job) -> dict[str, object]:
    result: dict[str, object] = {}

    async def on_reject(terminate: bool) -> None:
        result["rejected"] = terminate

    async def on_accept(args: JobAcceptArguments) -> None:
        result["accepted"] = args

    await handle_job_request(
        JobRequest(job=job, on_reject=on_reject, on_accept=on_accept)
    )
    return result


def test_browser_token_dispatches_this_worker_into_one_room(
    valid_env: None,
) -> None:
    settings = load_settings()
    api_key = settings.livekit_api_key.get_secret_value()
    api_secret = settings.livekit_api_secret.get_secret_value()
    token = (
        AccessToken(api_key, api_secret)
        .with_identity("participant-1")
        .with_grants(VideoGrants(room_join=True, room="test-room"))
        .with_room_config(RoomConfiguration(agents=[browser_dispatch()]))
        .to_jwt()
    )

    claims = TokenVerifier(api_key, api_secret).verify(token)
    dispatch = browser_dispatch()
    request = dispatch_request("test-room")

    assert claims.video is not None
    assert claims.video.room == "test-room"
    assert claims.video.room_join is True
    assert claims.metadata == ""
    assert claims.room_config is not None
    assert len(claims.room_config.agents) == 1
    assert claims.room_config.agents[0].agent_name == AGENT_NAME
    assert claims.room_config.agents[0].metadata == ""
    assert dispatch.agent_name == AGENT_NAME
    assert dispatch.metadata == ""
    assert request.agent_name == AGENT_NAME
    assert request.room == "test-room"
    assert request.metadata == ""
    assert api_secret not in token
    assert _USER_ID not in token


def test_api_dispatch_rejects_a_room_name_that_is_not_an_id() -> None:
    try:
        dispatch_request(f"room for {_USER_ID}")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("free-text room name was accepted")

    assert message == "room_name must be a short identifier"
    assert _USER_ID not in message


def test_test_room_activates_this_worker_without_taking_a_user_id() -> None:
    stream = StringIO()
    configure_logging(stream)
    job = _job()

    result = asyncio.run(_answer(job))
    assigned = bind_session(job, "test-room")
    accepted = result["accepted"]

    assert isinstance(accepted, JobAcceptArguments)
    assert isinstance(assigned, AssignedRoom)
    assert accepted.name == "Sarjy"
    assert accepted.identity == "agent-AJ_job1"
    assert accepted.metadata == ""
    assert accepted.attributes is None
    assert accepted.identity != _USER_ID
    assert assigned.room_name == "test-room"
    assert assigned.room_sid == "RM_testroom"
    assert assigned.job_id == "AJ_job1"
    assert assigned.dispatch_id == "AD_dispatch1"
    assert set(AssignedRoom.__dataclass_fields__) == {
        "job_id",
        "dispatch_id",
        "room_name",
        "room_sid",
    }
    assert "rejected" not in result
    assert stream.getvalue() == ""


def test_conversation_id_comes_from_job_metadata_and_a_user_id_does_not() -> None:
    conversation_id = "22222222-2222-4222-8222-222222222222"
    user_id = "11111111-1111-4111-8111-111111111111"
    job = _job(
        metadata=(
            '{"conversationId":"' + conversation_id + '","userId":"' + user_id + '"}'
        )
    )
    dispatch = browser_dispatch(conversation_id)

    assert conversation_id_from_job(job) == conversation_id
    assert conversation_id_from_job(_job()) is None
    assert user_id not in dispatch.metadata
    assert dispatch.metadata == '{"conversationId":"' + conversation_id + '"}'


def test_metadata_room_does_not_replace_the_job_room() -> None:
    job = _job()

    assigned = bind_session(job, "test-room")

    assert assigned is not None
    assert assigned.room_name == "test-room"
    assert assigned.room_name != _OTHER_ROOM


def test_other_agent_does_not_activate_this_worker() -> None:
    stream = StringIO()
    configure_logging(stream)
    job = _job(agent_name="other-agent")

    result = asyncio.run(_answer(job))
    logs = _logs(stream)

    assert result == {"rejected": True}
    assert bind_session(job, "test-room") is None
    assert logs[-1]["event"] == "provider_failure"
    assert logs[-1]["operation"] == "dispatch"
    assert logs[-1]["error_type"] == "AgentMismatch"
    assert "other-agent" not in stream.getvalue()
    assert _METADATA not in stream.getvalue()
    assert LEAK_MARKER not in stream.getvalue()


def test_missing_dispatch_and_unsafe_ids_are_rejected() -> None:
    stream = StringIO()
    configure_logging(stream)
    cases = (
        (_job(dispatch_id=""), "MissingDispatch"),
        (_job(job_id="job for user-123"), "UnsafeJob"),
        (_job(room_name="room for user-123"), "UnsafeRoom"),
        (_job(room_sid="sid for user-123"), "UnsafeRoom"),
    )

    for job, error_type in cases:
        result = asyncio.run(_answer(job))
        assert result == {"rejected": True}
        assert bind_session(job, job.room.name) is None
        assert _logs(stream)[-1]["error_type"] == error_type

    text = stream.getvalue()
    assert _USER_ID not in text
    assert LEAK_MARKER not in text


def test_a_different_connected_room_is_not_the_job_room() -> None:
    stream = StringIO()
    configure_logging(stream)
    job = _job()

    assert bind_session(job, _OTHER_ROOM) is None
    logs = _logs(stream)

    assert logs[-1]["error_type"] == "RoomMismatch"
    assert logs[-1]["room_id"] == "RM_testroom"
    assert _OTHER_ROOM not in stream.getvalue()
    assert _USER_ID not in stream.getvalue()


def test_session_joins_the_job_room_only(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = StringIO()
    started: list[str] = []

    async def start(ctx: _Context, _assigned: object) -> None:
        started.append(ctx.room.name)
        ctx.room.connected = True

    monkeypatch.setattr("sarjy_agent.worker._start_voice_session", start)

    async def scenario() -> tuple[_Context, _Context]:
        configure_logging(stream)
        accepted = _Context(_job(), "test-room")
        rejected = _Context(_job(), _OTHER_ROOM)
        await session_entrypoint(accepted)  # type: ignore[arg-type]
        await session_entrypoint(rejected)  # type: ignore[arg-type]
        return accepted, rejected

    accepted, rejected = asyncio.run(scenario())
    logs = _logs(stream)
    started_logs = [item for item in logs if item["event"] == "session_started"]

    assert started == ["test-room"]
    assert accepted.reason is None
    assert rejected.reason == "dispatch_rejected"
    failures = [item for item in logs if item["event"] == "provider_failure"]
    assert started_logs == [
        {
            "timestamp": started_logs[0]["timestamp"],
            "level": "INFO",
            "event": "session_started",
            "room_id": "RM_testroom",
            "session_id": "AJ_job1",
            "turn_id": None,
        }
    ]
    assert [item["error_type"] for item in failures] == ["RoomMismatch"]
    assert _OTHER_ROOM not in stream.getvalue()
    assert _USER_ID not in stream.getvalue()
    assert LEAK_MARKER not in stream.getvalue()

"""Explicit LiveKit dispatch for the sarjy worker.

A browser session activates this worker by dispatching the agent name
``sarjy`` into one room. The room on the job is the room the worker joins.
Job metadata and participant identity are not a user identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from livekit.agents import JobRequest
from livekit.api import CreateAgentDispatchRequest, RoomAgentDispatch

from sarjy_agent.log import log_context, log_provider_failure, public_id

AGENT_NAME = "sarjy"
AGENT_PARTICIPANT_NAME = "Sarjy"

_AGENT_MISMATCH = "AgentMismatch"
_MISSING_DISPATCH = "MissingDispatch"
_UNSAFE_JOB = "UnsafeJob"
_UNSAFE_ROOM = "UnsafeRoom"
_ROOM_MISMATCH = "RoomMismatch"


@dataclass(frozen=True)
class AssignedRoom:
    """The room LiveKit assigned to one job.

    These fields identify the job and the room. They are not a user id.
    """

    job_id: str
    dispatch_id: str
    room_name: str
    room_sid: str


@dataclass(frozen=True)
class _Decision:
    room: AssignedRoom | None
    error_type: str | None


def browser_dispatch() -> RoomAgentDispatch:
    """Agent entry for a participant token.

    The room comes from the token's room grant, not from metadata.
    """
    return RoomAgentDispatch(agent_name=AGENT_NAME)


def dispatch_request(room_name: str) -> CreateAgentDispatchRequest:
    """Build an API dispatch for this worker and one room.

    The request does not carry metadata. A room name that is not a short id
    is rejected here, before any call to LiveKit.
    """
    if public_id(room_name) is None:
        raise ValueError("room_name must be a short identifier")
    return CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=room_name)


async def handle_job_request(request: JobRequest) -> None:
    """Accept a job for this worker and the room on that job.

    The agent participant identity is the job id. Metadata and the
    publisher identity are not copied onto the agent.
    """
    decision = _decide_job(request.job)
    if decision.room is None:
        _log_rejected(decision, request.job)
        await request.reject()
        return
    await request.accept(
        name=AGENT_PARTICIPANT_NAME,
        identity=f"agent-{decision.room.job_id}",
        metadata="",
    )


def bind_session(job: object, connected_room_name: str) -> AssignedRoom | None:
    """Return the job room when this process may join it.

    ``connected_room_name`` is the room object the worker would join. It must
    be the same name as the room on the job.
    """
    decision = _decide_job(job)
    if decision.room is None:
        _log_rejected(decision, job)
        return None
    if connected_room_name != decision.room.room_name:
        _log_rejected(_Decision(None, _ROOM_MISMATCH), job)
        return None
    return decision.room


def _decide_job(job: object) -> _Decision:
    agent_name, job_id, dispatch_id, room_name, room_sid = _job_fields(job)
    if agent_name != AGENT_NAME:
        return _Decision(None, _AGENT_MISMATCH)
    if public_id(dispatch_id) is None:
        return _Decision(None, _MISSING_DISPATCH)
    if public_id(job_id) is None:
        return _Decision(None, _UNSAFE_JOB)
    if public_id(room_name) is None:
        return _Decision(None, _UNSAFE_ROOM)
    if room_sid != "" and public_id(room_sid) is None:
        return _Decision(None, _UNSAFE_ROOM)
    return _Decision(
        AssignedRoom(
            job_id=job_id,
            dispatch_id=dispatch_id,
            room_name=room_name,
            room_sid=room_sid,
        ),
        None,
    )


def _job_fields(job: object) -> tuple[str, str, str, str, str]:
    room = getattr(job, "room", None)
    return (
        _text(getattr(job, "agent_name", "")),
        _text(getattr(job, "id", "")),
        _text(getattr(job, "dispatch_id", "")),
        _text(getattr(room, "name", "")),
        _text(getattr(room, "sid", "")),
    )


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _log_rejected(decision: _Decision, job: object) -> None:
    _agent_name, job_id, _dispatch_id, _room_name, room_sid = _job_fields(job)
    error_type = decision.error_type or "DispatchRejected"
    with log_context(room_id=public_id(room_sid), session_id=public_id(job_id)):
        log_provider_failure(
            provider="livekit",
            operation="dispatch",
            error_type=error_type,
        )

"""Decide which LiveKit room this worker is allowed to join."""

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
    """The job id and room LiveKit assigned. This is not a user id."""

    job_id: str
    dispatch_id: str
    room_name: str
    room_sid: str


@dataclass(frozen=True)
class _Decision:
    """The room to join, or the short reason the job was rejected."""

    room: AssignedRoom | None
    error_type: str | None


def browser_dispatch() -> RoomAgentDispatch:
    """Name the sarjy agent on a participant token. The room comes from the token."""
    return RoomAgentDispatch(agent_name=AGENT_NAME)


def dispatch_request(room_name: str) -> CreateAgentDispatchRequest:
    """Build a LiveKit request that starts sarjy in one room."""
    if public_id(room_name) is None:
        raise ValueError("room_name must be a short identifier")
    return CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=room_name)


async def handle_job_request(request: JobRequest) -> None:
    """Accept this job when it is for sarjy and one safe room name."""
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
    """Return the job room only when the connected room has that same name."""
    decision = _decide_job(job)
    if decision.room is None:
        _log_rejected(decision, job)
        return None
    if connected_room_name != decision.room.room_name:
        _log_rejected(_Decision(None, _ROOM_MISMATCH), job)
        return None
    return decision.room


def _decide_job(job: object) -> _Decision:
    """Accept sarjy plus short job, dispatch, and room ids. Reject anything else."""
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
    """Read the agent name, job id, dispatch id, room name, and room sid."""
    room = getattr(job, "room", None)
    return (
        _text(getattr(job, "agent_name", "")),
        _text(getattr(job, "id", "")),
        _text(getattr(job, "dispatch_id", "")),
        _text(getattr(room, "name", "")),
        _text(getattr(room, "sid", "")),
    )


def _text(value: object) -> str:
    """Return the value when it is text, otherwise an empty string."""
    if isinstance(value, str):
        return value
    return ""


def _log_rejected(decision: _Decision, job: object) -> None:
    """Log a rejected dispatch with a short error name and no free text."""
    _agent_name, job_id, _dispatch_id, _room_name, room_sid = _job_fields(job)
    error_type = decision.error_type or "DispatchRejected"
    with log_context(room_id=public_id(room_sid), session_id=public_id(job_id)):
        log_provider_failure(
            provider="livekit",
            operation="dispatch",
            error_type=error_type,
        )

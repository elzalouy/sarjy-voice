"""Call the Next.js internal API for the voice session.

Identity comes from the authorize response. A user id on a tool call is ignored.
Partial speech is not sent. The service token stays in the Authorization header.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from sarjy_agent.config import Settings
from sarjy_agent.dispatch import AssignedRoom
from sarjy_agent.log import log_context, log_provider_failure, public_id
from sarjy_agent.turns import FinalUserTurn

AUTHORIZE_PATH = "/api/internal/voice-sessions/authorize"
WEATHER_PATH = "/api/internal/weather"
REQUEST_ID_HEADER = "X-Request-Id"

_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 2
_CONTENT_LIMIT = 8000

TranscriptRole = Literal["user", "assistant"]
TranscriptStatus = Literal["completed", "interrupted", "incomplete"]
MemoryKey = Literal["preferred_name", "city", "language"]


class BackendError(Exception):
    """The Next.js call failed. The message is a short error name."""

    def __init__(self, error_type: str) -> None:
        """Store a short name such as Timeout or Unauthorized."""
        super().__init__(error_type)
        self.error_type = error_type


class AuthorizeRequest(BaseModel):
    """The conversation and room. This body has no user id."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    conversation_id: str = Field(
        validation_alias="conversationId",
        serialization_alias="conversationId",
    )
    room_name: str = Field(
        validation_alias="roomName",
        serialization_alias="roomName",
    )

    @field_validator("conversation_id")
    @classmethod
    def conversation_must_be_uuid(cls, value: str) -> str:
        """Keep a uuid conversation id."""
        return _require_uuid(value)

    @field_validator("room_name")
    @classmethod
    def room_must_be_short(cls, value: str) -> str:
        """Keep the LiveKit room name short."""
        return _require_public_id(value)


class AuthorizeResponse(BaseModel):
    """The owner and conversation the backend bound to this room."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    conversation_id: str = Field(
        validation_alias="conversationId",
        serialization_alias="conversationId",
    )
    user_id: str = Field(validation_alias="userId", serialization_alias="userId")
    room_name: str = Field(
        validation_alias="roomName",
        serialization_alias="roomName",
    )

    @field_validator("conversation_id", "user_id")
    @classmethod
    def ids_must_be_uuids(cls, value: str) -> str:
        """Keep uuid ids from the backend."""
        return _require_uuid(value)

    @field_validator("room_name")
    @classmethod
    def room_must_be_short(cls, value: str) -> str:
        """Keep the room name the backend confirmed."""
        return _require_public_id(value)


class TranscriptRequest(BaseModel):
    """One finished line. The user id is the authorized owner."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    user_id: str = Field(validation_alias="userId", serialization_alias="userId")
    role: TranscriptRole
    content: str
    event_id: str = Field(validation_alias="eventId", serialization_alias="eventId")
    created_at: str = Field(
        validation_alias="createdAt",
        serialization_alias="createdAt",
    )

    @field_validator("user_id")
    @classmethod
    def user_must_be_uuid(cls, value: str) -> str:
        """Send the authorized user id."""
        return _require_uuid(value)

    @field_validator("event_id")
    @classmethod
    def event_id_must_be_safe(cls, value: str) -> str:
        """Keep the idempotency key within the backend limit."""
        if _EVENT_ID.fullmatch(value) is None:
            raise ValueError("must be a short identifier")
        return value

    @field_validator("content")
    @classmethod
    def content_must_fit(cls, value: str) -> str:
        """Send one non-empty line, at most 8000 characters."""
        cleaned = value.strip()
        if not cleaned or len(cleaned) > _CONTENT_LIMIT:
            raise ValueError("content length is invalid")
        return cleaned


class TranscriptMessage(BaseModel):
    """The line the backend stored."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    role: TranscriptRole
    content: str
    created_at: str = Field(
        validation_alias="createdAt",
        serialization_alias="createdAt",
    )


class TranscriptResponse(BaseModel):
    """A stored transcript line."""

    model_config = ConfigDict(extra="ignore")

    message: TranscriptMessage


class MemoryFact(BaseModel):
    """One saved preference for the authorized user."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    key: str
    value: str
    updated_at: str = Field(
        validation_alias="updatedAt",
        serialization_alias="updatedAt",
    )


class MemoryWrite(BaseModel):
    """A preference write. The model cannot choose the user or the source."""

    model_config = ConfigDict(extra="forbid")

    key: MemoryKey
    value: str
    source: Literal["user"] = "user"
    approved: Literal[True] = True


class WeatherReport(BaseModel):
    """Weather returned by the backend. Extra fields are dropped."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    location: str
    condition: str
    temperature_c: float = Field(
        validation_alias="temperatureC",
        serialization_alias="temperatureC",
    )
    humidity_percent: int = Field(
        validation_alias="humidityPercent",
        serialization_alias="humidityPercent",
    )
    observed_at: str = Field(
        validation_alias="observedAt",
        serialization_alias="observedAt",
    )


@dataclass(frozen=True)
class AuthorizedSession:
    """The user and conversation for one LiveKit room."""

    user_id: str
    conversation_id: str
    room_name: str
    room_sid: str
    job_id: str
    dispatch_id: str


@dataclass(frozen=True)
class _Reply:
    """A parsed JSON object, the status, and the request id."""

    body: dict[str, object]
    status_code: int
    request_id: str


def authorized_user_id(
    session: AuthorizedSession, tool_user_id: object | None = None
) -> str:
    """Return the user the backend authorized for this room.

    A tool argument is not identity. It is ignored.
    """
    del tool_user_id
    return session.user_id


def assistant_status(*, text: str, interrupted: bool) -> TranscriptStatus:
    """Name how an assistant line ended."""
    if interrupted:
        return "interrupted"
    if not text.strip():
        return "incomplete"
    return "completed"


def transcript_path(conversation_id: str) -> str:
    """Return the transcript path for one conversation uuid."""
    return f"/api/internal/conversations/{_require_uuid(conversation_id)}/transcripts"


def memory_path(user_id: str) -> str:
    """Return the memory path for the authorized user uuid."""
    return f"/api/internal/users/{_require_uuid(user_id)}/memory"


def create_backend(settings: Settings) -> BackendClient:
    """Build the client for the configured Next.js origin."""
    return BackendClient(settings)


class BackendClient:
    """HTTP calls to the Next.js internal voice API."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float | None = None,
        retry_pause_seconds: float = 0.1,
        max_attempts: int = _MAX_ATTEMPTS,
    ) -> None:
        """Store the origin, the service token, and the wait limit."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_pause_seconds < 0:
            raise ValueError("retry_pause_seconds must be zero or greater")
        timeout = (
            settings.backend_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self._base = str(settings.nextjs_api_base_url).rstrip("/")
        self._retry_pause = retry_pause_seconds
        self._max_attempts = max_attempts
        self._written: set[str] = set()
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": (
                    f"Bearer {settings.agent_service_token.get_secret_value()}"
                ),
                "Accept": "application/json",
            },
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def __repr__(self) -> str:
        """Show the origin only. The service token stays hidden."""
        return f"BackendClient(base_url={self._base!r})"

    async def aclose(self) -> None:
        """Close the HTTP connection."""
        await self._http.aclose()

    async def authorize(
        self, room: AssignedRoom, conversation_id: str
    ) -> AuthorizedSession:
        """Bind this room to the active conversation. The backend names the user."""
        request = AuthorizeRequest(
            conversation_id=conversation_id,
            room_name=room.room_name,
        )
        reply = await self._request(
            "POST",
            AUTHORIZE_PATH,
            request.model_dump(by_alias=True),
            operation="authorize",
        )
        parsed = _parse(AuthorizeResponse, reply.body)
        if isinstance(parsed, BackendError):
            _log_failure("authorize", parsed.error_type, reply.request_id, None)
            raise parsed
        if (
            parsed.conversation_id != request.conversation_id
            or parsed.room_name != request.room_name
        ):
            _log_failure("authorize", "Rejected", reply.request_id, None)
            raise BackendError("Rejected")
        return AuthorizedSession(
            user_id=parsed.user_id,
            conversation_id=parsed.conversation_id,
            room_name=room.room_name,
            room_sid=room.room_sid,
            job_id=room.job_id,
            dispatch_id=room.dispatch_id,
        )

    async def save_transcript(
        self, conversation_id: str, message: TranscriptRequest
    ) -> None:
        """Store one finished line. The same event id is written once."""
        if message.event_id in self._written:
            return
        reply = await self._request(
            "POST",
            transcript_path(conversation_id),
            message.model_dump(by_alias=True),
            operation="save_transcript",
            ok_statuses=frozenset({200, 201}),
        )
        parsed = _parse(TranscriptResponse, reply.body)
        if isinstance(parsed, BackendError):
            _log_failure("save_transcript", parsed.error_type, reply.request_id, None)
            raise parsed
        if parsed.message.role != message.role or not parsed.message.content.strip():
            _log_failure("save_transcript", "InvalidResponse", reply.request_id, None)
            raise BackendError("InvalidResponse")
        self._written.add(message.event_id)

    async def list_memory(self, user_id: str) -> tuple[MemoryFact, ...]:
        """Load the authorized user's saved preferences."""
        try:
            path = memory_path(user_id)
        except ValueError:
            raise BackendError("Rejected") from None
        reply = await self._request(
            "GET",
            path,
            None,
            operation="list_memory",
        )
        return _memory_facts(reply.body, reply.request_id)

    async def remember(self, user_id: str, write: MemoryWrite) -> MemoryFact:
        """Store or replace one preference. The result is the saved row."""
        return await self._write_memory("PUT", "remember_memory", user_id, write)

    async def correct(self, user_id: str, write: MemoryWrite) -> MemoryFact:
        """Replace a preference that already exists."""
        return await self._write_memory("PATCH", "correct_memory", user_id, write)

    async def forget(self, user_id: str, key: MemoryKey) -> bool:
        """Delete one preference. False means that key was not stored."""
        reply = await self._request(
            "DELETE",
            memory_path(user_id),
            {"key": key, "source": "user", "approved": True},
            operation="forget_memory",
        )
        deleted = reply.body.get("deleted")
        if not isinstance(deleted, bool):
            _log_failure("forget_memory", "InvalidResponse", reply.request_id, None)
            raise BackendError("InvalidResponse")
        return deleted

    async def weather(self, location: str) -> WeatherReport:
        """Ask the backend for weather. The provider key stays on the backend."""
        reply = await self._request(
            "POST",
            WEATHER_PATH,
            {"location": location},
            operation="weather",
        )
        parsed = _parse(WeatherReport, reply.body)
        if isinstance(parsed, BackendError):
            _log_failure("weather", parsed.error_type, reply.request_id, None)
            raise parsed
        return parsed

    async def _write_memory(
        self,
        method: str,
        operation: str,
        user_id: str,
        write: MemoryWrite,
    ) -> MemoryFact:
        """Send one memory write and require the saved row to match."""
        reply = await self._request(
            method,
            memory_path(user_id),
            write.model_dump(),
            operation=operation,
        )
        raw = reply.body.get("memory")
        if not isinstance(raw, dict):
            _log_failure(operation, "InvalidResponse", reply.request_id, None)
            raise BackendError("InvalidResponse")
        parsed = _parse(MemoryFact, raw)
        if isinstance(parsed, BackendError):
            _log_failure(operation, parsed.error_type, reply.request_id, None)
            raise parsed
        if parsed.key != write.key or parsed.value != write.value:
            _log_failure(operation, "InvalidResponse", reply.request_id, None)
            raise BackendError("InvalidResponse")
        return parsed

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        *,
        operation: str,
        ok_statuses: frozenset[int] = frozenset({200}),
    ) -> _Reply:
        """Send one request. Retry a timeout, 408, 429, or 5xx once."""
        started = perf_counter()
        last_error = BackendError("Unavailable")
        last_request_id = _request_id()
        for attempt in range(self._max_attempts):
            request_id = _request_id()
            last_request_id = request_id
            headers = {REQUEST_ID_HEADER: request_id}
            try:
                if payload is None:
                    response = await self._http.request(
                        method,
                        self._base + path,
                        headers=headers,
                    )
                else:
                    response = await self._http.request(
                        method,
                        self._base + path,
                        headers=headers,
                        json=payload,
                    )
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException:
                last_error = BackendError("Timeout")
            except httpx.TransportError:
                last_error = BackendError("Unavailable")
            else:
                if response.status_code in ok_statuses:
                    try:
                        return _Reply(
                            _json_object(response),
                            response.status_code,
                            request_id,
                        )
                    except BackendError as exc:
                        last_error = exc
                        break
                last_error = BackendError(_error_for_status(response.status_code))
                if response.status_code not in _RETRYABLE:
                    break
            if attempt + 1 < self._max_attempts and self._retry_pause:
                await asyncio.sleep(self._retry_pause * (attempt + 1))
        _log_failure(
            operation,
            last_error.error_type,
            last_request_id,
            (perf_counter() - started) * 1000,
        )
        raise last_error


class TranscriptStore:
    """Save finished lines for one authorized conversation, in order."""

    def __init__(self, client: BackendClient, session: AuthorizedSession) -> None:
        """Remember the client and the authorized user."""
        self._client = client
        self._session = session
        self._saved: set[str] = set()
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[bool]] = set()

    async def save_user(self, turn: FinalUserTurn) -> bool:
        """Save one finished user line. A second call does nothing."""
        try:
            return await self._save(
                role="user",
                item_id=turn.turn_id,
                turn_id=turn.turn_id,
                text=turn.text,
                status="completed",
            )
        except BackendError:
            return False

    async def save_assistant(
        self,
        *,
        message_id: str,
        turn_id: str,
        text: str,
        interrupted: bool,
    ) -> bool:
        """Save the final assistant text. An empty line is not stored."""
        cleaned = text.strip()
        try:
            return await self._save(
                role="assistant",
                item_id=message_id,
                turn_id=turn_id,
                text=cleaned,
                status=assistant_status(text=cleaned, interrupted=interrupted),
            )
        except BackendError:
            return False

    def schedule_assistant(
        self,
        *,
        message_id: str,
        turn_id: str,
        text: str,
        interrupted: bool,
    ) -> None:
        """Save an assistant line without blocking the audio callback."""
        task = asyncio.create_task(
            self.save_assistant(
                message_id=message_id,
                turn_id=turn_id,
                text=text,
                interrupted=interrupted,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    async def flush(self) -> None:
        """Wait until scheduled assistant lines have finished saving."""
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks))

    async def _save(
        self,
        *,
        role: TranscriptRole,
        item_id: str,
        turn_id: str,
        text: str,
        status: TranscriptStatus,
    ) -> bool:
        """Send one line when it has text. The same event id is sent once."""
        if not text.strip() or status == "incomplete":
            return False
        if public_id(turn_id) is None or public_id(item_id) is None:
            _log_failure("save_transcript", "InvalidMessage", None, None)
            return False
        event_id = _event_id(self._session.conversation_id, role, item_id)
        async with self._lock:
            if event_id in self._saved:
                return False
            try:
                message = TranscriptRequest(
                    user_id=authorized_user_id(self._session),
                    role=role,
                    content=text,
                    event_id=event_id,
                    created_at=_timestamp(),
                )
            except ValidationError:
                _log_failure("save_transcript", "InvalidMessage", None, None)
                return False
            room_id = public_id(self._session.room_sid) or public_id(
                self._session.room_name
            )
            with log_context(
                room_id=room_id,
                session_id=public_id(self._session.job_id),
                turn_id=turn_id,
            ):
                await self._client.save_transcript(
                    self._session.conversation_id, message
                )
            self._saved.add(event_id)
            return True

    def _task_done(self, task: asyncio.Task[bool]) -> None:
        """Drop a finished task and read its error so it is not reported later."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        task.exception()


def _memory_facts(body: dict[str, object], request_id: str) -> tuple[MemoryFact, ...]:
    """Keep well-formed memory rows. A missing list is an invalid response."""
    raw = body.get("memories")
    if not isinstance(raw, list):
        _log_failure("list_memory", "InvalidResponse", request_id, None)
        raise BackendError("InvalidResponse")
    facts: list[MemoryFact] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        parsed = _parse(MemoryFact, item)
        if isinstance(parsed, BackendError):
            continue
        facts.append(parsed)
    return tuple(facts)


def _require_public_id(value: str) -> str:
    """Return the id when it is safe to send and to log."""
    checked = public_id(value)
    if checked is None:
        raise ValueError("must be a short identifier")
    return checked


def _require_uuid(value: str) -> str:
    """Return a lowercase uuid, or raise when the value is not one."""
    if _UUID.fullmatch(value) is None:
        raise ValueError("must be a uuid")
    return value.lower()


def _event_id(conversation_id: str, role: str, item_id: str) -> str:
    """Build the idempotency key for one line."""
    value = f"{conversation_id}:{role}:{item_id}"
    if _EVENT_ID.fullmatch(value) is None:
        _log_failure("save_transcript", "InvalidMessage", None, None)
        raise BackendError("InvalidMessage")
    return value


def _timestamp() -> str:
    """Return the current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _request_id() -> str:
    """Return a new correlation id for one HTTP attempt."""
    return uuid4().hex


def _parse[T: BaseModel](model: type[T], body: dict[str, object]) -> T | BackendError:
    """Validate a response. A bad shape becomes InvalidResponse."""
    try:
        return model.model_validate(body)
    except ValidationError:
        return BackendError("InvalidResponse")


def _json_object(response: httpx.Response) -> dict[str, object]:
    """Return a JSON object. Anything else is an invalid response."""
    try:
        body = response.json()
    except ValueError:
        raise BackendError("InvalidResponse") from None
    if not isinstance(body, dict):
        raise BackendError("InvalidResponse")
    return body


def _error_for_status(status: int) -> str:
    """Map an HTTP status to a short error name."""
    if status == 401:
        return "Unauthorized"
    if status == 403:
        return "Forbidden"
    if status == 404:
        return "NotFound"
    if status in _RETRYABLE:
        return "Unavailable"
    return "Rejected"


def _log_failure(
    operation: str,
    error_type: str,
    request_id: str | None,
    duration_ms: float | None,
) -> None:
    """Log a Next.js failure. The body and the token are not included."""
    if public_id(request_id) is None:
        request_id = None
    log_provider_failure(
        provider="nextjs",
        operation=operation,
        error_type=error_type,
        duration_ms=duration_ms,
        request_id=request_id,
    )

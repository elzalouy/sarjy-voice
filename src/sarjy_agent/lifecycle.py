"""Join one LiveKit room, watch who enters and leaves, then disconnect once."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping
from contextlib import contextmanager
from typing import Literal, Protocol

from livekit.protocol.models import DisconnectReason

from sarjy_agent.log import (
    SessionEndReason,
    log_context,
    log_lifecycle,
    log_participant,
    log_provider_failure,
    log_session_ended,
    public_id,
)

_CLIENT_DISCONNECT = "CLIENT_INITIATED"
_CLASS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class _Participant(Protocol):
    """The only participant fact this module reads: a short session id."""

    @property
    def sid(self) -> str:
        """The person's short session id."""
        ...


_RoomEvent = Literal[
    "participant_connected",
    "participant_disconnected",
    "disconnected",
]


class _Room(Protocol):
    """The room events this module needs: join, leave, disconnect, and close."""

    def on(self, event: _RoomEvent, callback: Callable[..., object]) -> object:
        """Start listening for one room event."""
        ...

    def off(self, event: _RoomEvent, callback: Callable[..., object]) -> None:
        """Stop listening for one room event."""
        ...

    def isconnected(self) -> bool:
        """Return true when the room is still connected."""
        ...

    async def disconnect(self) -> None:
        """Leave the room."""
        ...

    @property
    def remote_participants(self) -> Mapping[str, _Participant]:
        """The people in the room, by their id."""
        ...


class SessionLifecycle:
    """One room visit. Connect once, then leave once."""

    def __init__(
        self,
        *,
        room: _Room,
        connect: Callable[[], Awaitable[None]],
        add_shutdown_callback: Callable[
            [Callable[[str], Coroutine[None, None, None]]], None
        ],
        room_id: str | None,
        session_id: str | None,
        stop_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        """Remember the room, the connect function, and the log ids.

        An error listed in stop_errors ends the visit before the session starts.
        """
        self._room = room
        self._connect = connect
        self._add_shutdown_callback = add_shutdown_callback
        self._room_id = room_id
        self._session_id = session_id
        self._stop_errors = stop_errors
        self._started = False
        self._closed = False
        self._lost = False
        self._seen: set[tuple[str, str]] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        """Connect to the room and log that the session started."""
        self._watch()
        self._add_shutdown_callback(self._on_shutdown)
        try:
            await self._connect()
        except asyncio.CancelledError:
            await self.leave("cancelled")
            raise
        except Exception as exc:
            if isinstance(exc, self._stop_errors):
                await self.leave("shutdown")
                return
            self._log_failure("connect", _error_type(exc))
            await self.leave("connection_lost")
            raise
        self._started = True
        if self._lost:
            await self.leave("connection_lost")
            return
        with self._context():
            log_lifecycle("session_started")
            for participant in self._room.remote_participants.values():
                self._on_joined(participant)

    async def leave(self, reason: SessionEndReason) -> None:
        """Disconnect if still connected, and log the end reason once."""
        if self._closed:
            return
        self._closed = True
        self._unwatch()
        if self._room.isconnected():
            try:
                await self._room.disconnect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_failure("disconnect", _error_type(exc))
        if self._started:
            with self._context():
                log_session_ended(reason)

    async def _on_shutdown(self, _reason: str) -> None:
        """Leave the room because the worker is stopping."""
        await self.leave("shutdown")

    def _watch(self) -> None:
        """Start listening for people joining, leaving, and the room closing."""
        self._room.on("participant_connected", self._on_joined)
        self._room.on("participant_disconnected", self._on_left)
        self._room.on("disconnected", self._on_disconnected)

    def _unwatch(self) -> None:
        """Stop listening for room events."""
        self._room.off("participant_connected", self._on_joined)
        self._room.off("participant_disconnected", self._on_left)
        self._room.off("disconnected", self._on_disconnected)

    def _on_joined(self, participant: _Participant) -> None:
        """Log that one remote person joined."""
        self._log_presence("participant_joined", participant)

    def _on_left(self, participant: _Participant) -> None:
        """Log that one remote person left."""
        self._log_presence("participant_left", participant)

    def _on_disconnected(self, reason: object) -> None:
        """Leave after an unexpected drop. A normal close is just shutdown."""
        if self._closed:
            return
        name = _disconnect_name(reason)
        if not self._started:
            self._lost = True
            if name != _CLIENT_DISCONNECT:
                self._log_failure("room_connection", name)
            return
        if name != _CLIENT_DISCONNECT:
            self._log_failure("room_connection", name)
            self._spawn(self.leave("connection_lost"))
            return
        self._spawn(self.leave("shutdown"))

    def _log_presence(
        self,
        event: Literal["participant_joined", "participant_left"],
        participant: _Participant,
    ) -> None:
        """Log a join or leave once for this person."""
        sid = public_id(participant.sid)
        key = sid if sid is not None else f"obj:{id(participant)}"
        marker = (event, key)
        if marker in self._seen:
            return
        self._seen.add(marker)
        if event == "participant_joined":
            with self._context():
                log_participant("participant_joined", sid)
            return
        with self._context():
            log_participant("participant_left", sid)

    def _log_failure(self, operation: str, error_type: str) -> None:
        """Log a LiveKit failure with a short name and no error text."""
        with self._context():
            log_provider_failure(
                provider="livekit",
                operation=operation,
                error_type=error_type,
            )

    def _spawn(self, work: Coroutine[None, None, None]) -> None:
        """Run the leave work in the background and keep a reference to it."""
        task: asyncio.Task[None] = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @contextmanager
    def _context(self) -> Iterator[None]:
        """Attach this room id and session id to logs inside the block."""
        with log_context(room_id=self._room_id, session_id=self._session_id):
            yield


def _error_type(exc: BaseException) -> str:
    """Return a short exception class name, or Error when the name is unsafe."""
    name = type(exc).__name__
    if _CLASS_NAME.fullmatch(name):
        return name
    return "Error"


def _disconnect_name(reason: object) -> str:
    """Return a short disconnect reason, or UNKNOWN_REASON."""
    if isinstance(reason, str) and _CLASS_NAME.fullmatch(reason):
        return reason
    if isinstance(reason, int):
        try:
            name = DisconnectReason.Name(reason)
        except ValueError:
            return "UNKNOWN_REASON"
        if isinstance(name, str) and _CLASS_NAME.fullmatch(name):
            return name
    return "UNKNOWN_REASON"

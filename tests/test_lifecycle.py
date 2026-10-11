import asyncio
import json
from collections.abc import Awaitable, Callable
from io import StringIO

from conftest import LEAK_MARKER
from sarjy_agent.lifecycle import SessionLifecycle
from sarjy_agent.log import configure_logging, public_id


class _Person:
    def __init__(self, sid: str, identity: str = "hidden") -> None:
        self.sid = sid
        self.identity = identity


class _Room:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable[..., object]]] = {}
        self.connected = False
        self.remote_participants: dict[str, _Person] = {}
        self.disconnects = 0

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
        self.disconnects += 1
        self.connected = False

    def emit(self, event: str, *args: object) -> None:
        for callback in list(self.handlers.get(event, [])):
            callback(*args)


def _logs(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def _events(stream: StringIO) -> list[str]:
    return [str(item["event"]) for item in _logs(stream)]


async def _settle() -> None:
    await asyncio.sleep(0)
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current]
    if pending:
        await asyncio.gather(*pending)


def test_public_id_keeps_only_short_ids() -> None:
    assert public_id("RM_room1") == "RM_room1"
    assert public_id("user said hello") is None
    assert public_id("") is None


def test_a_stop_error_ends_the_visit_before_the_session_starts() -> None:
    stream = StringIO()

    class _Stop(Exception):
        pass

    async def scenario() -> int:
        configure_logging(stream)
        room = _Room()

        async def connect() -> None:
            room.connected = True
            raise _Stop

        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=lambda _callback: None,
            room_id="RM_room1",
            session_id="AJ_job1",
            stop_errors=(_Stop,),
        )
        await lifecycle.run()
        return room.disconnects

    disconnects = asyncio.run(scenario())

    assert disconnects == 1
    assert _events(stream) == []


def test_join_then_shutdown_leaves_once() -> None:
    stream = StringIO()

    async def scenario() -> None:
        configure_logging(stream)
        room = _Room()
        room.connected = False
        shutdowns: list[Callable[[str], Awaitable[None]]] = []

        async def connect() -> None:
            room.connected = True

        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=shutdowns.append,
            room_id="RM_room1",
            session_id="AJ_job1",
        )
        await lifecycle.run()
        room.emit("participant_connected", _Person("PA_guest", identity=LEAK_MARKER))
        room.emit("participant_connected", _Person("PA_guest"))
        room.emit("participant_disconnected", _Person("PA_guest"))
        await shutdowns[0]("room closed because the caller left")
        await shutdowns[0]("room closed because the caller left")

    asyncio.run(scenario())
    logs = _logs(stream)
    assert _events(stream) == [
        "session_started",
        "participant_joined",
        "participant_left",
        "session_ended",
    ]
    assert logs[0]["room_id"] == "RM_room1"
    assert logs[0]["session_id"] == "AJ_job1"
    assert logs[1]["participant_id"] == "PA_guest"
    assert logs[3]["end_reason"] == "shutdown"
    assert LEAK_MARKER not in stream.getvalue()
    assert "caller left" not in stream.getvalue()


def test_participants_already_in_the_room_are_recorded() -> None:
    stream = StringIO()

    async def scenario() -> None:
        configure_logging(stream)
        room = _Room()
        guest = _Person("PA_guest")
        room.remote_participants["guest"] = guest

        async def connect() -> None:
            room.connected = True

        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=lambda _callback: None,
            room_id="RM_room1",
            session_id="AJ_job1",
        )
        await lifecycle.run()
        room.emit("participant_connected", guest)

    asyncio.run(scenario())
    joined = [item for item in _logs(stream) if item["event"] == "participant_joined"]
    assert joined == [
        {
            "timestamp": joined[0]["timestamp"],
            "level": "INFO",
            "event": "participant_joined",
            "room_id": "RM_room1",
            "session_id": "AJ_job1",
            "turn_id": None,
            "participant_id": "PA_guest",
        }
    ]


def test_unsafe_participant_id_is_omitted() -> None:
    stream = StringIO()

    async def scenario() -> None:
        configure_logging(stream)
        room = _Room()

        async def connect() -> None:
            room.connected = True

        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=lambda _callback: None,
            room_id="RM_room1",
            session_id="AJ_job1",
        )
        await lifecycle.run()
        room.emit("participant_connected", _Person("not a safe id"))

    asyncio.run(scenario())
    text = stream.getvalue()
    assert "not a safe id" not in text
    joined = [item for item in _logs(stream) if item["event"] == "participant_joined"]
    assert len(joined) == 1
    assert "participant_id" not in joined[0]


def test_cancellation_during_connect_leaves_without_starting() -> None:
    stream = StringIO()

    async def scenario() -> None:
        configure_logging(stream)
        room = _Room()

        async def connect() -> None:
            room.connected = True
            raise asyncio.CancelledError

        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=lambda _callback: None,
            room_id="RM_room1",
            session_id="AJ_job1",
        )
        try:
            await lifecycle.run()
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation was swallowed")
        assert room.disconnects == 1
        assert room.connected is False

    asyncio.run(scenario())
    assert _events(stream) == []


def test_connect_failure_is_logged_without_the_message() -> None:
    stream = StringIO()

    async def scenario() -> None:
        configure_logging(stream)
        room = _Room()

        async def connect() -> None:
            room.connected = True
            raise RuntimeError(f"websocket {LEAK_MARKER}")

        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=lambda _callback: None,
            room_id="RM_room1",
            session_id="AJ_job1",
        )
        try:
            await lifecycle.run()
        except RuntimeError:
            pass
        else:
            raise AssertionError("connect failure was swallowed")
        assert room.disconnects == 1

    asyncio.run(scenario())
    logs = _logs(stream)
    assert [item["event"] for item in logs] == ["provider_failure"]
    assert logs[0]["provider"] == "livekit"
    assert logs[0]["operation"] == "connect"
    assert logs[0]["error_type"] == "RuntimeError"
    assert logs[0]["room_id"] == "RM_room1"
    assert LEAK_MARKER not in stream.getvalue()


def test_unexpected_disconnect_ends_the_session_once() -> None:
    stream = StringIO()

    async def scenario() -> None:
        configure_logging(stream)
        room = _Room()
        shutdowns: list[Callable[[str], Awaitable[None]]] = []

        async def connect() -> None:
            room.connected = True

        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=shutdowns.append,
            room_id="RM_room1",
            session_id="AJ_job1",
        )
        await lifecycle.run()
        room.connected = False
        room.emit("disconnected", "CONNECTION_TIMEOUT")
        await _settle()
        await shutdowns[0]("server gone")

    asyncio.run(scenario())
    logs = _logs(stream)
    assert [item["event"] for item in logs] == [
        "session_started",
        "provider_failure",
        "session_ended",
    ]
    assert logs[1]["operation"] == "room_connection"
    assert logs[1]["error_type"] == "CONNECTION_TIMEOUT"
    assert logs[2]["end_reason"] == "connection_lost"
    assert "server gone" not in stream.getvalue()


def test_disconnect_failure_is_logged_and_the_session_still_ends() -> None:
    stream = StringIO()

    async def scenario() -> None:
        configure_logging(stream)
        room = _Room()

        async def connect() -> None:
            room.connected = True

        async def disconnect() -> None:
            raise OSError(f"close {LEAK_MARKER}")

        room.disconnect = disconnect  # type: ignore[method-assign]
        lifecycle = SessionLifecycle(
            room=room,
            connect=connect,
            add_shutdown_callback=lambda _callback: None,
            room_id="RM_room1",
            session_id="AJ_job1",
        )
        await lifecycle.run()
        await lifecycle.leave("cancelled")

    asyncio.run(scenario())
    logs = _logs(stream)
    assert [item["event"] for item in logs] == [
        "session_started",
        "provider_failure",
        "session_ended",
    ]
    assert logs[1]["operation"] == "disconnect"
    assert logs[1]["error_type"] == "OSError"
    assert logs[2]["end_reason"] == "cancelled"
    assert LEAK_MARKER not in stream.getvalue()

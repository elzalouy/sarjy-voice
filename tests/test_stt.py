import asyncio

import pytest
from livekit.agents import AgentSession, inference
from livekit.plugins import deepgram, openai

from conftest import VALID_ENV, allowing_backend
from sarjy_agent.config import load_settings
from sarjy_agent.dispatch import AssignedRoom
from sarjy_agent.eval.code_switch import CODE_SWITCH_LANGUAGES, CODE_SWITCH_MODEL
from sarjy_agent.stt import create_stt
from sarjy_agent.worker import Assistant, _start_voice_session, create_voice_session


def test_default_recognizer_is_streaming_nova3_ar_sa(valid_env: None) -> None:
    recognizer = create_stt(load_settings())

    assert isinstance(recognizer, deepgram.STT)
    assert not isinstance(recognizer, inference.STT)
    assert recognizer.provider == "Deepgram"
    assert recognizer.model == "nova-3"
    assert recognizer._opts.language == "ar-SA"
    assert recognizer.capabilities.streaming is True
    assert recognizer.capabilities.interim_results is True
    assert recognizer._api_key == VALID_ENV["DEEPGRAM_API_KEY"]


def test_model_and_locale_come_from_settings(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STT_MODEL", "nova-2")
    monkeypatch.setenv("STT_LOCALE", "ar")

    recognizer = create_stt(load_settings())

    assert recognizer.model == "nova-2"
    assert recognizer._opts.language == "ar"


def test_plugin_uses_the_settings_key_not_a_later_environment_value(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "later-environment-key")

    recognizer = create_stt(settings)

    assert recognizer._api_key == VALID_ENV["DEEPGRAM_API_KEY"]
    assert "later-environment-key" not in repr(recognizer)


def test_voice_session_hears_arabic_and_english_together(valid_env: None) -> None:
    session = asyncio.run(_voice_session())

    assert isinstance(session.stt, openai.STT)
    assert session.stt.model == CODE_SWITCH_MODEL
    assert session.stt._opts.languages == list(CODE_SWITCH_LANGUAGES)
    assert session.stt.capabilities.streaming is True
    assert session.llm is not None
    assert session.tts is not None


async def _voice_session() -> AgentSession[None]:
    return create_voice_session(load_settings())


def test_job_starts_a_session_with_the_configured_recognizer(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STT_LOCALE", "ar-EG")
    started: dict[str, object] = {}

    class _Session:
        def __init__(
            self,
            *,
            stt: object,
            llm: object,
            tts: object,
            vad: object = None,
            turn_handling: object = None,
        ) -> None:
            started["stt"] = stt
            started["llm"] = llm
            started["tts"] = tts
            self._events: dict[str, dict[object, None]] = {}

        def on(self, event: str, _callback: object) -> None:
            events = started.setdefault("events", [])
            assert isinstance(events, list)
            events.append(event)

        async def start(self, *, agent: object, room: object) -> None:
            started["agent"] = agent
            started["room"] = room

    class _Room:
        name = "test-room"

    class _Context:
        room = _Room()

        def add_shutdown_callback(self, _callback: object) -> None:
            return None

    monkeypatch.setattr("sarjy_agent.worker.AgentSession", _Session)
    monkeypatch.setattr("sarjy_agent.worker.create_backend", allowing_backend)

    asyncio.run(
        _start_voice_session(
            _Context(),  # type: ignore[arg-type]
            AssignedRoom(
                job_id="AJ_job1",
                dispatch_id="AD_dispatch1",
                room_name="test-room",
                room_sid="RM_testroom",
            ),
        )
    )

    recognizer = started["stt"]
    assert isinstance(recognizer, openai.STT)
    assert recognizer.model == CODE_SWITCH_MODEL
    assert recognizer._opts.languages == list(CODE_SWITCH_LANGUAGES)
    assert recognizer._client.api_key == VALID_ENV["OPENAI_API_KEY"]
    assert started["events"] == [
        "user_input_transcribed",
        "error",
        "conversation_item_added",
    ]
    assert isinstance(started["agent"], Assistant)
    room = started["room"]
    assert isinstance(room, _Room)
    assert room.name == "test-room"

import asyncio

import pytest
from livekit.agents import AgentSession, inference
from livekit.plugins import deepgram

from conftest import VALID_ENV
from sarjy_agent.config import load_settings
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


def test_voice_session_uses_the_deepgram_recognizer(valid_env: None) -> None:
    session = asyncio.run(_voice_session())

    assert isinstance(session.stt, deepgram.STT)
    assert session.stt.model == "nova-3"
    assert session.stt._opts.language == "ar-SA"
    assert session.llm is None
    assert session.tts is not None


async def _voice_session() -> AgentSession[None]:
    return create_voice_session(load_settings())


def test_job_starts_a_session_with_the_configured_recognizer(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STT_LOCALE", "ar-EG")
    started: dict[str, object] = {}

    class _Session:
        def __init__(self, *, stt: object, tts: object) -> None:
            started["stt"] = stt
            started["tts"] = tts

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

    monkeypatch.setattr("sarjy_agent.worker.AgentSession", _Session)

    asyncio.run(_start_voice_session(_Context()))  # type: ignore[arg-type]

    recognizer = started["stt"]
    assert isinstance(recognizer, deepgram.STT)
    assert recognizer.model == "nova-3"
    assert recognizer._opts.language == "ar-EG"
    assert recognizer._api_key == VALID_ENV["DEEPGRAM_API_KEY"]
    assert started["events"] == ["user_input_transcribed", "error"]
    assert isinstance(started["agent"], Assistant)
    room = started["room"]
    assert isinstance(room, _Room)
    assert room.name == "test-room"

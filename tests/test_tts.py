import asyncio

import pytest
from livekit.agents import AgentSession, inference
from livekit.plugins import elevenlabs

from conftest import VALID_ENV
from sarjy_agent.config import load_settings
from sarjy_agent.tts import create_tts
from sarjy_agent.worker import Assistant, _start_voice_session, create_voice_session


def test_default_speaker_is_streaming_flash_v2_5(valid_env: None) -> None:
    speaker = create_tts(load_settings())

    assert isinstance(speaker, elevenlabs.TTS)
    assert not isinstance(speaker, inference.TTS)
    assert speaker.provider == "ElevenLabs"
    assert speaker.model == "eleven_flash_v2_5"
    assert speaker._opts.voice_id == "voice-test"
    assert speaker._opts.language == "ar-SA"
    assert speaker._opts.language.language == "ar"
    assert speaker.capabilities.streaming is True
    assert speaker._opts.api_key == VALID_ENV["ELEVEN_API_KEY"]


def test_model_voice_and_locale_come_from_settings(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TTS_MODEL", "eleven_multilingual_v2")
    monkeypatch.setenv("TTS_VOICE_ID", "voice-custom")
    monkeypatch.setenv("STT_LOCALE", "ar")

    speaker = create_tts(load_settings())

    assert speaker.model == "eleven_multilingual_v2"
    assert speaker._opts.voice_id == "voice-custom"
    assert speaker._opts.language == "ar"
    assert speaker._opts.language.language == "ar"


def test_plugin_uses_the_settings_key_not_a_later_environment_value(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings()
    monkeypatch.setenv("ELEVEN_API_KEY", "later-environment-key")

    speaker = create_tts(settings)

    assert speaker._opts.api_key == VALID_ENV["ELEVEN_API_KEY"]
    assert "later-environment-key" not in repr(speaker)


def test_voice_session_speaks_with_elevenlabs(valid_env: None) -> None:
    session = asyncio.run(_voice_session())

    assert isinstance(session, AgentSession)
    assert isinstance(session.tts, elevenlabs.TTS)
    assert session.tts.model == "eleven_flash_v2_5"
    assert session.tts._opts.voice_id == "voice-test"
    assert session.tts._opts.language.language == "ar"
    assert session.llm is None


async def _voice_session() -> AgentSession[None]:
    return create_voice_session(load_settings())


def test_job_starts_a_session_with_the_configured_speaker(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TTS_MODEL", "eleven_flash_v2_5")
    monkeypatch.setenv("TTS_VOICE_ID", "voice-job")
    started: dict[str, object] = {}

    class _Session:
        def __init__(self, *, stt: object, tts: object) -> None:
            started["stt"] = stt
            started["tts"] = tts

        def on(self, _event: str, _callback: object) -> None:
            return None

        async def start(self, *, agent: object, room: object) -> None:
            started["agent"] = agent
            started["room"] = room

    class _Room:
        name = "test-room"

    class _Context:
        room = _Room()

    monkeypatch.setattr("sarjy_agent.worker.AgentSession", _Session)

    asyncio.run(_start_voice_session(_Context()))  # type: ignore[arg-type]

    speaker = started["tts"]
    assert isinstance(speaker, elevenlabs.TTS)
    assert speaker.model == "eleven_flash_v2_5"
    assert speaker._opts.voice_id == "voice-job"
    assert speaker._opts.api_key == VALID_ENV["ELEVEN_API_KEY"]
    assert isinstance(started["agent"], Assistant)
    room = started["room"]
    assert isinstance(room, _Room)
    assert room.name == "test-room"

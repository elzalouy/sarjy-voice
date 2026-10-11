import asyncio

import pytest
from livekit.agents import AgentSession, inference
from livekit.plugins import elevenlabs

from conftest import VALID_ENV, allowing_backend
from sarjy_agent.config import load_settings
from sarjy_agent.dispatch import AssignedRoom
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
    monkeypatch.setenv("TTS_LOCALE", "ar")

    speaker = create_tts(load_settings())

    assert speaker.model == "eleven_multilingual_v2"
    assert speaker._opts.voice_id == "voice-custom"
    assert speaker._opts.language == "ar"
    assert speaker._opts.language.language == "ar"


def test_a_new_model_reaches_the_session_without_a_new_startup_path(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TTS_MODEL", "eleven_multilingual_v2")

    async def open_session() -> AgentSession[None]:
        return create_voice_session(load_settings())

    session = asyncio.run(open_session())

    assert isinstance(session.tts, elevenlabs.TTS)
    assert session.tts.model == "eleven_multilingual_v2"


def test_an_unknown_provider_is_rejected_before_a_session_starts(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sarjy_agent.tts_config import TtsConfigError

    monkeypatch.setenv("TTS_PROVIDER", "other-vendor")

    with pytest.raises(TtsConfigError, match="other-vendor"):
        create_tts(load_settings())


def test_plugin_uses_the_settings_key_not_a_later_environment_value(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings()
    monkeypatch.setenv("ELEVEN_API_KEY", "later-environment-key")

    speaker = create_tts(settings)

    assert speaker._opts.api_key == VALID_ENV["ELEVEN_API_KEY"]
    assert "later-environment-key" not in repr(speaker)


def test_the_first_sentence_is_ready_before_the_reply_ends(valid_env: None) -> None:
    speaker = create_tts(load_settings())

    assert speaker._opts.auto_mode is True
    assert speaker.capabilities.streaming is True

    first = asyncio.run(_first_sentence(speaker))

    assert "السلام" in first
    assert "أساعدك" not in first


async def _first_sentence(speaker: elevenlabs.TTS) -> str:
    stream = speaker._opts.word_tokenizer.stream()
    stream.push_text("وعليكم السلام. كيف أقدر أساعدك")
    try:
        item = await asyncio.wait_for(anext(stream), timeout=1)
    finally:
        await stream.aclose()
    return item.token


def test_voice_session_speaks_with_elevenlabs(valid_env: None) -> None:
    session = asyncio.run(_voice_session())

    assert isinstance(session, AgentSession)
    assert isinstance(session.tts, elevenlabs.TTS)
    assert session.tts.model == "eleven_flash_v2_5"
    assert session.tts._opts.voice_id == "voice-test"
    assert session.tts._opts.language.language == "ar"
    assert session.llm is not None


async def _voice_session() -> AgentSession[None]:
    return create_voice_session(load_settings())


def test_job_starts_a_session_with_the_configured_speaker(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TTS_MODEL", "eleven_flash_v2_5")
    monkeypatch.setenv("TTS_VOICE_ID", "voice-job")
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

        def on(self, _event: str, _callback: object) -> None:
            return None

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

    speaker = started["tts"]
    assert isinstance(speaker, elevenlabs.TTS)
    assert speaker.model == "eleven_flash_v2_5"
    assert speaker._opts.voice_id == "voice-job"
    assert speaker._opts.api_key == VALID_ENV["ELEVEN_API_KEY"]
    assert isinstance(started["agent"], Assistant)
    room = started["room"]
    assert isinstance(room, _Room)
    assert room.name == "test-room"

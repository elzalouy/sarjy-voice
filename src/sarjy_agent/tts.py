"""Build the ElevenLabs speaker used by the voice agent."""

from __future__ import annotations

from livekit.agents.tts.tts import ChunkedStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins import elevenlabs

from sarjy_agent.config import Settings
from sarjy_agent.segments import SpeechSegments
from sarjy_agent.speech_text import normalize_for_speech
from sarjy_agent.tts_config import (
    BUILT_PROVIDER,
    TtsConfig,
    TtsConfigError,
    tts_config_from_settings,
)


def create_tts(
    settings: Settings,
    *,
    model: str | None = None,
    voice_id: str | None = None,
) -> elevenlabs.TTS:
    """Build the configured speaker. The room session does not name the provider."""
    config = tts_config_from_settings(settings, model=model, voice_id=voice_id)
    if config.provider != BUILT_PROVIDER:
        raise TtsConfigError(
            f"TTS provider {config.provider} is not built. "
            f"The built provider is {BUILT_PROVIDER}."
        )
    return _elevenlabs_speaker(settings, config)


def _elevenlabs_speaker(settings: Settings, config: TtsConfig) -> elevenlabs.TTS:
    """Turn one speaker configuration into an ElevenLabs voice.

    A ready phrase is spoken as soon as it is ready. A single word is held.
    The rest of the reply does not have to finish first.
    """
    speaker = elevenlabs.TTS(
        api_key=settings.eleven_api_key.get_secret_value(),
        model=config.model,
        voice_id=config.voice_id,
        language=config.locale,
        apply_text_normalization=config.text_normalization,
        auto_mode=True,
        word_tokenizer=SpeechSegments(),
    )
    _normalize_one_shot(speaker)
    return speaker


def _normalize_one_shot(speaker: elevenlabs.TTS) -> None:
    """Spell numbers on the one-shot speak path, not only the live stream."""
    original = speaker.synthesize

    def synthesize(
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Spell numbers, then send that text to ElevenLabs once."""
        return original(normalize_for_speech(text), conn_options=conn_options)

    speaker.synthesize = synthesize  # type: ignore[assignment]

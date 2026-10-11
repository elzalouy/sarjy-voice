"""The speaker settings the voice session asks for.

The room session does not choose a provider. It asks for this configuration,
and one builder turns it into a speaker. To try another provider later, add
a builder for that name and set ``TTS_PROVIDER``. Leave session startup as it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sarjy_agent.config import Settings

TextNormalization = Literal["off", "auto", "on"]
BUILT_PROVIDER = "elevenlabs"


class TtsConfigError(Exception):
    """The speaker settings name a provider this process cannot build."""


@dataclass(frozen=True)
class TtsConfig:
    """Model, voice, locale, and generation options for one speaker."""

    provider: str
    model: str
    voice_id: str
    locale: str
    text_normalization: TextNormalization


def tts_config_from_settings(
    settings: Settings,
    *,
    model: str | None = None,
    voice_id: str | None = None,
) -> TtsConfig:
    """Copy the speaker settings. A comparison run may override model or voice."""
    normalization: TextNormalization
    if settings.tts_text_normalization == "auto":
        normalization = "auto"
    elif settings.tts_text_normalization == "on":
        normalization = "on"
    elif settings.tts_text_normalization == "off":
        normalization = "off"
    else:
        raise TtsConfigError("TTS text normalization must be off, auto, or on.")
    return TtsConfig(
        provider=settings.tts_provider,
        model=settings.tts_model if model is None else model,
        voice_id=settings.tts_voice_id if voice_id is None else voice_id,
        locale=settings.tts_locale,
        text_normalization=normalization,
    )

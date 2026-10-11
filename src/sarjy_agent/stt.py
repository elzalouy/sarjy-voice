"""Build the Deepgram recognizer used by the voice agent."""

from __future__ import annotations

from livekit.plugins import deepgram

from sarjy_agent.config import Settings


def create_stt(settings: Settings) -> deepgram.STT:
    """Make a streaming Deepgram recognizer from the model, locale, and key."""
    return deepgram.STT(
        model=settings.stt_model,
        language=settings.stt_locale,
        api_key=settings.deepgram_api_key.get_secret_value(),
    )

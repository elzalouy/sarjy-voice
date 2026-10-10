"""Streaming speech recognition through the LiveKit Deepgram plugin.

The recognizer uses the Deepgram key from settings. It does not send audio
through LiveKit Inference.
"""

from __future__ import annotations

from livekit.plugins import deepgram

from sarjy_agent.config import Settings


def create_stt(settings: Settings) -> deepgram.STT:
    """Build a streaming Deepgram recognizer.

    ``STT_MODEL`` selects the model. ``STT_LOCALE`` selects the language.
    The defaults are Nova-3 and Saudi Arabic ``ar-SA``.
    """
    return deepgram.STT(
        model=settings.stt_model,
        language=settings.stt_locale,
        api_key=settings.deepgram_api_key.get_secret_value(),
    )

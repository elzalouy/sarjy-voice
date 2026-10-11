"""Build the recognizer used by the voice agent.

Arabic-only checks stay on Deepgram Nova-3 ``ar-SA``. A live call uses
OpenAI ``gpt-live-transcribe`` with Arabic and English together, because
Nova-3 ``multi`` does not include Arabic.
"""

from __future__ import annotations

from livekit.plugins import deepgram, openai

from sarjy_agent.config import Settings
from sarjy_agent.eval.code_switch import (
    CODE_SWITCH_LANGUAGES,
    CODE_SWITCH_MODEL,
    english_keywords,
)


def create_stt(settings: Settings, *, locale: str | None = None) -> deepgram.STT:
    """Make a streaming Deepgram recognizer for one language."""
    return deepgram.STT(
        model=settings.stt_model,
        language=settings.stt_locale if locale is None else locale,
        api_key=settings.deepgram_api_key.get_secret_value(),
    )


def create_code_switch_stt(settings: Settings) -> openai.STT:
    """Hear Arabic and English in the same utterance.

    The language list is ``ar`` and ``en``. Nova-3 ``language=multi`` is not
    used, because Arabic is outside that model's documented language list.
    """
    return openai.STT(
        model=CODE_SWITCH_MODEL,
        language=list(CODE_SWITCH_LANGUAGES),
        keywords=list(english_keywords()),
        api_key=settings.openai_api_key.get_secret_value(),
    )

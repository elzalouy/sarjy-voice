"""Streaming speech synthesis through the LiveKit ElevenLabs plugin.

The speaker uses the ElevenLabs key from settings. It does not send text
through LiveKit Inference, and it does not send the key to the browser.

Numbers are spelled in this process before they are sent. Flash v2.5 does
not normalize numbers unless the account is Enterprise. This account was
not verified as Enterprise, so provider normalization stays off.
Language-specific provider normalization is for Japanese only, so it is
not sent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from livekit.agents.tokenize import blingfire
from livekit.agents.tokenize.tokenizer import (
    SentenceStream,
    SentenceTokenizer,
    TokenData,
)
from livekit.agents.tts.tts import ChunkedStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins import elevenlabs

from sarjy_agent.config import Settings
from sarjy_agent.speech_text import normalize_for_speech


def create_tts(
    settings: Settings,
    *,
    model: str | None = None,
    voice_id: str | None = None,
) -> elevenlabs.TTS:
    """Build a streaming ElevenLabs speaker.

    ``TTS_MODEL`` selects the model. The default is Flash v2.5.
    ``TTS_VOICE_ID`` selects the voice.
    ``STT_LOCALE`` selects the language. ``ar-SA`` is sent to ElevenLabs as ``ar``.
    Phones, prices, dates, and times are spelled before synthesis.
    ``model`` and ``voice_id`` override the settings for one comparison run.
    """
    speaker = elevenlabs.TTS(
        api_key=settings.eleven_api_key.get_secret_value(),
        model=settings.tts_model if model is None else model,
        voice_id=settings.tts_voice_id if voice_id is None else voice_id,
        language=settings.stt_locale,
        apply_text_normalization="off",
        word_tokenizer=_SpeechSentences(),
    )
    _normalize_one_shot(speaker)
    return speaker


def _normalize_one_shot(speaker: elevenlabs.TTS) -> None:
    """Spell numbers on the HTTP synthesis path as well as the stream."""
    original = speaker.synthesize

    def synthesize(
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return original(normalize_for_speech(text), conn_options=conn_options)

    speaker.synthesize = synthesize  # type: ignore[assignment]


class _SpeechSentences(SentenceTokenizer):
    """Split into sentences, then spell numbers in each sentence."""

    def __init__(self) -> None:
        self._inner = blingfire.SentenceTokenizer(min_sentence_len=1)

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            normalize_for_speech(part)
            for part in self._inner.tokenize(text, language=language)
        ]

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return _SpeechSentenceStream(self._inner.stream(language=language))


class _SpeechSentenceStream(SentenceStream):
    def __init__(self, inner: SentenceStream) -> None:
        super().__init__()
        self._inner = inner

    def push_text(self, text: str) -> None:
        self._inner.push_text(text)

    def flush(self) -> None:
        self._inner.flush()

    def end_input(self) -> None:
        self._inner.end_input()

    async def aclose(self) -> None:
        await self._inner.aclose()
        self._do_close()

    async def __anext__(self) -> TokenData:
        item = await self._inner.__anext__()
        return TokenData(
            token=normalize_for_speech(item.token),
            segment_id=item.segment_id,
        )

    def __aiter__(self) -> AsyncIterator[TokenData]:
        return self

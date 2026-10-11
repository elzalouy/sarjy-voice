"""Build the ElevenLabs speaker used by the voice agent."""

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
    """Make an ElevenLabs speaker from settings, with numbers spelled first."""
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


class _SpeechSentences(SentenceTokenizer):
    """Split speech text into sentences, then spell the numbers in each one."""

    def __init__(self) -> None:
        """Use the built-in sentence splitter."""
        self._inner = blingfire.SentenceTokenizer(min_sentence_len=1)

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Return the sentences with phones, prices, dates, and times spelled."""
        return [
            normalize_for_speech(part)
            for part in self._inner.tokenize(text, language=language)
        ]

    def stream(self, *, language: str | None = None) -> SentenceStream:
        """Start a live sentence stream that spells numbers as text arrives."""
        return _SpeechSentenceStream(self._inner.stream(language=language))


class _SpeechSentenceStream(SentenceStream):
    """A live sentence stream that spells numbers before each sentence is spoken."""

    def __init__(self, inner: SentenceStream) -> None:
        """Wrap the built-in sentence stream."""
        super().__init__()
        self._inner = inner

    def push_text(self, text: str) -> None:
        """Add more text to the sentence splitter."""
        self._inner.push_text(text)

    def flush(self) -> None:
        """Finish the current sentence even if it has no ending mark."""
        self._inner.flush()

    def end_input(self) -> None:
        """Tell the splitter that no more text is coming."""
        self._inner.end_input()

    async def aclose(self) -> None:
        """Close the inner stream and this wrapper."""
        await self._inner.aclose()
        self._do_close()

    async def __anext__(self) -> TokenData:
        """Return the next sentence with its numbers spelled."""
        item = await self._inner.__anext__()
        return TokenData(
            token=normalize_for_speech(item.token),
            segment_id=item.segment_id,
        )

    def __aiter__(self) -> AsyncIterator[TokenData]:
        """Let callers loop over sentences with ``async for``."""
        return self

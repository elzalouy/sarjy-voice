"""Choose when streamed reply text is ready to speak.

A phrase goes to speech as soon as it is long enough to sound natural.
A single word stays buffered. A long reply is not held until the last word.
The same text can be cut the old way, one sentence at a time, so the two
waits can be compared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from livekit.agents.metrics import LLMMetrics, TTSMetrics
from livekit.agents.tokenize import blingfire
from livekit.agents.tokenize.tokenizer import (
    SentenceStream,
    SentenceTokenizer,
    TokenData,
)
from livekit.agents.utils import shortuuid
from livekit.agents.utils.aio.channel import ChanEmpty

from sarjy_agent.log import log_timing
from sarjy_agent.speech_text import normalize_for_speech

# A period, question mark, or exclamation with at least this many words.
_MIN_SENTENCE_WORDS = 2
# A comma, semicolon, or colon with at least this many words.
_MIN_CLAUSE_WORDS = 3
# Send this many words once the buffer is longer than one spoken phrase.
_MAX_CHUNK_WORDS = 12
# Keep this many words back so a length cut does not leave one word behind.
_MIN_WORDS_AFTER_CHUNK = 3

_SENTENCE_MARKS = frozenset(".!?؟")
_CLAUSE_MARKS = frozenset("،,؛;:")
_NUMBER_MARKS = frozenset(".:،,")
_CLOSING_QUOTES = frozenset("\"'\u201d\u2019\u00bb")


class _Segmenter(Protocol):
    """Turn arriving text into pieces that can be spoken."""

    def push(self, text: str) -> list[str]:
        """Add text. Return any pieces that are ready now."""

    def flush(self) -> list[str]:
        """The reply is finished. Return whatever is still buffered."""


@dataclass(frozen=True)
class TextChunk:
    """One piece of model text and when it arrived, in milliseconds."""

    text: str
    at_ms: float


@dataclass(frozen=True)
class ReplyPlayback:
    """One long reply after it has been cut into spoken pieces."""

    first_audio_ms: float
    total_ms: float
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class StreamTiming:
    """First token, first speakable piece, and the pieces, for one cut."""

    first_token_ms: float
    first_audio_ms: float
    segments: tuple[str, ...]


@dataclass(frozen=True)
class StreamComparison:
    """The old sentence split and the new phrase split for one reply."""

    utterance: str
    baseline: StreamTiming
    tuned: StreamTiming


class StreamTimingLog:
    """Log the first text token and the first audio frame of each reply."""

    def __init__(self) -> None:
        """Start with no audio heard for the current reply."""
        self._heard_audio = False

    def observe_event(self, event: object) -> None:
        """Read a metrics event. Other events are ignored."""
        self.observe(getattr(event, "metrics", None))

    def observe(self, metrics: object) -> None:
        """Log a first token, or the first audio frame after that token."""
        if isinstance(metrics, LLMMetrics):
            self._heard_audio = False
            if metrics.ttft >= 0:
                log_timing(
                    stage="llm_first_token",
                    duration_ms=metrics.ttft * 1000,
                )
            return
        if not isinstance(metrics, TTSMetrics) or self._heard_audio:
            return
        if metrics.ttfb < 0:
            return
        self._heard_audio = True
        log_timing(stage="tts_first_audio", duration_ms=metrics.ttfb * 1000)


class SpeechSegments(SentenceTokenizer):
    """Split speech into phrases, then spell the numbers in each phrase."""

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Return every phrase, including the tail at the end of the reply."""
        del language
        return list(spoken_segments(text))

    def stream(self, *, language: str | None = None) -> SentenceStream:
        """Start a live phrase stream."""
        del language
        return _SegmentStream(_PhraseSegmenter())


def spoken_segments(text: str) -> tuple[str, ...]:
    """Return every phrase, including the tail at the end of the reply."""
    segmenter = _PhraseSegmenter()
    parts = segmenter.push(text)
    parts.extend(segmenter.flush())
    return tuple(parts)


def opening_segments(text: str) -> tuple[str, ...]:
    """Return phrases that are ready before the reply ends."""
    return tuple(_PhraseSegmenter().push(text))


def measure_reply(text: str, *, ms_per_char: float = 1.0) -> ReplyPlayback:
    """Play one reply as it arrives and record the first piece and the full wait."""
    chunks = char_chunks(text, ms_per_char=ms_per_char)
    timing = _time_segmenter(_PhraseSegmenter(), chunks)
    total_ms = chunks[-1].at_ms if chunks else 0.0
    return ReplyPlayback(
        first_audio_ms=timing.first_audio_ms,
        total_ms=total_ms,
        chunks=timing.segments,
    )


def playback_problems(source: str, playback: ReplyPlayback) -> tuple[str, ...]:
    """Return missing or repeated words, or a reply that waited until the end."""
    problems: list[str] = []
    if playback.chunks and playback.first_audio_ms >= playback.total_ms:
        problems.append("waited_for_the_full_reply")
    heard = _words(" ".join(playback.chunks))
    expected = _words(source)
    if heard != expected:
        if len(heard) < len(expected):
            problems.append("missing_words")
        elif len(heard) > len(expected):
            problems.append("duplicated_words")
        else:
            problems.append("reordered_words")
    return tuple(problems)


def _words(text: str) -> tuple[str, ...]:
    """Return the words of one line, in order."""
    return tuple(text.split())


def char_chunks(text: str, *, ms_per_char: float = 1.0) -> tuple[TextChunk, ...]:
    """Give each character an arrival time, one after another."""
    return tuple(
        TextChunk(text=char, at_ms=(index + 1) * ms_per_char)
        for index, char in enumerate(text)
    )


async def compare_utterance(text: str, *, ms_per_char: float = 1.0) -> StreamComparison:
    """Split one reply both ways and record when speech can start.

    ``first_token_ms`` is when the first character arrives.
    ``first_audio_ms`` is when the first phrase is ready to send.
    The speaker's own wait is not included.
    """
    chunks = char_chunks(text, ms_per_char=ms_per_char)
    baseline = _BaselineSegmenter()
    try:
        return StreamComparison(
            utterance=text,
            baseline=_time_segmenter(baseline, chunks),
            tuned=_time_segmenter(_PhraseSegmenter(), chunks),
        )
    finally:
        await baseline.aclose()


class _PhraseSegmenter:
    """Hold text until a phrase is long enough, or the reply ends."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self._buf = ""

    def push(self, text: str) -> list[str]:
        """Add text. Return any phrases that are ready now."""
        if not text:
            return []
        self._buf += text
        return self._take(final=False)

    def flush(self) -> list[str]:
        """The reply is finished. Return whatever is still buffered."""
        return self._take(final=True)

    def _take(self, *, final: bool) -> list[str]:
        """Pull every ready phrase out of the buffer."""
        ready: list[str] = []
        while self._buf.strip():
            cut = _cut_index(self._buf, final=final)
            if cut is None or cut <= 0:
                break
            piece = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if piece:
                ready.append(normalize_for_speech(piece))
        return ready


class _BaselineSegmenter:
    """The old split. It sends one sentence at a time, even one word."""

    def __init__(self) -> None:
        """Use the sentence splitter the speaker used before phrase cuts."""
        self._inner = blingfire.SentenceTokenizer(min_sentence_len=1).stream()

    def push(self, text: str) -> list[str]:
        """Add text. Return any finished sentences."""
        self._inner.push_text(text)
        return self._ready()

    def flush(self) -> list[str]:
        """The reply is finished. Return the remaining sentence."""
        self._inner.flush()
        return self._ready()

    async def aclose(self) -> None:
        """Close the sentence splitter."""
        await self._inner.aclose()

    def _ready(self) -> list[str]:
        """Take sentences the splitter has already finished."""
        ready: list[str] = []
        while True:
            try:
                item = self._inner._event_ch.recv_nowait()
            except ChanEmpty:
                return [normalize_for_speech(token) for token in ready]
            ready.append(item.token)


class _SegmentStream(SentenceStream):
    """A live stream. It sends each phrase as soon as it is ready."""

    def __init__(self, segmenter: _Segmenter) -> None:
        """Wrap one segmenter."""
        super().__init__()
        self._segmenter = segmenter
        self._segment_id = shortuuid()

    def push_text(self, text: str) -> None:
        """Add more reply text."""
        self._check_not_closed()
        self._send(self._segmenter.push(text))

    def flush(self) -> None:
        """Finish the current reply even if the tail has no ending mark."""
        self._check_not_closed()
        self._send(self._segmenter.flush())
        self._segment_id = shortuuid()

    def end_input(self) -> None:
        """Tell the stream that no more text is coming."""
        self.flush()
        self._do_close()

    async def aclose(self) -> None:
        """Close this stream."""
        self._do_close()

    def _send(self, tokens: Sequence[str]) -> None:
        """Queue phrases for the speaker."""
        for token in tokens:
            self._event_ch.send_nowait(
                TokenData(token=token, segment_id=self._segment_id)
            )


def _time_segmenter(segmenter: _Segmenter, chunks: Sequence[TextChunk]) -> StreamTiming:
    """Play one timed reply and record the first token and the first phrase."""
    segments: list[str] = []
    first_token_ms: float | None = None
    first_audio_ms: float | None = None
    for chunk in chunks:
        if chunk.text and first_token_ms is None:
            first_token_ms = chunk.at_ms
        ready = segmenter.push(chunk.text)
        if ready and first_audio_ms is None:
            first_audio_ms = chunk.at_ms
        segments.extend(ready)
    ready = segmenter.flush()
    if ready and first_audio_ms is None:
        first_audio_ms = chunks[-1].at_ms if chunks else 0.0
    segments.extend(ready)
    if first_token_ms is None or first_audio_ms is None:
        raise ValueError("reply produced no text")
    return StreamTiming(
        first_token_ms=first_token_ms,
        first_audio_ms=first_audio_ms,
        segments=tuple(segments),
    )


def _cut_index(text: str, *, final: bool) -> int | None:
    """Return how many leading characters are ready, or nothing yet."""
    punctuation = _punctuation_cut(text, final=final)
    if final:
        if punctuation is not None:
            return punctuation
        return len(text) if text.strip() else None
    length = _length_cut(text)
    if punctuation is None:
        return length
    if length is None or punctuation <= length:
        return punctuation
    return length


def _punctuation_cut(text: str, *, final: bool) -> int | None:
    """Return the earliest ending mark that has enough words before it."""
    for index, char in enumerate(text):
        kind = _boundary_kind(char)
        if kind is None or not _mark_is_ready(text, index, final=final):
            continue
        end = index + 1
        while end < len(text) and text[end] in _CLOSING_QUOTES:
            end += 1
        words = len(text[:end].split())
        minimum = _MIN_SENTENCE_WORDS if kind == "sentence" else _MIN_CLAUSE_WORDS
        if words >= minimum:
            return end
    return None


def _length_cut(text: str) -> int | None:
    """Cut a long stretch once enough later words are already buffered."""
    ends = _complete_word_ends(text)
    if len(ends) < _MAX_CHUNK_WORDS + _MIN_WORDS_AFTER_CHUNK:
        return None
    cut_at = _MAX_CHUNK_WORDS
    while cut_at >= _MIN_CLAUSE_WORDS:
        cut = ends[cut_at - 1]
        if not _ends_with_number(text[:cut]):
            return cut
        cut_at -= 1
    return None


def _complete_word_ends(text: str) -> list[int]:
    """Return the index of the space after each finished word."""
    ends: list[int] = []
    index = 0
    size = len(text)
    while index < size:
        while index < size and text[index].isspace():
            index += 1
        start = index
        while index < size and not text[index].isspace():
            index += 1
        if start == index:
            break
        if index < size and text[index].isspace():
            ends.append(index)
        else:
            break
    return ends


def _boundary_kind(char: str) -> str | None:
    """Return sentence, clause, or nothing for this character."""
    if char in _SENTENCE_MARKS:
        return "sentence"
    if char in _CLAUSE_MARKS:
        return "clause"
    return None


def _mark_is_ready(text: str, index: int, *, final: bool) -> bool:
    """Return false when this mark might still be part of a number or dots."""
    char = text[index]
    if char == ".":
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if nxt == "." or (nxt == "" and not final):
            return False
    return not (char in _NUMBER_MARKS and _between_digits(text, index, final=final))


def _between_digits(text: str, index: int, *, final: bool) -> bool:
    """Return true when this mark sits inside a number, or might."""
    if index == 0 or not text[index - 1].isdecimal():
        return False
    if index + 1 >= len(text):
        return not final
    return text[index + 1].isdecimal()


def _ends_with_number(text: str) -> bool:
    """Return true when the last word contains a digit."""
    words = text.split()
    if not words:
        return False
    return any(char.isdecimal() for char in words[-1])

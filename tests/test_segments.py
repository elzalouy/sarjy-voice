"""Phrase cuts against the old sentence cut, on the same text."""

import asyncio
import json
from io import StringIO

import pytest
from livekit import rtc
from livekit.agents.metrics import LLMMetrics, TTSMetrics

from sarjy_agent.log import configure_logging
from sarjy_agent.segments import (
    SpeechSegments,
    StreamTimingLog,
    compare_utterance,
    opening_segments,
    spoken_segments,
)
from sarjy_agent.speech_text import normalize_for_speech
from sarjy_agent.worker import _watch_transcripts

_CLAUSE = "أقرب محطة تبعد خمس دقائق، وتقدر تشحن هناك الحين قبل المغرب."
_ONE_WORD = "نعم. أقرب محطة تبعد خمس دقائق عن البيت."
_LONG_WORDS = (
    "أقدر",
    "أساعدك",
    "تلاقي",
    "أقرب",
    "محطة",
    "شحن",
    "مفتوحة",
    "وأقولك",
    "الوقت",
    "والسعر",
    "والمسافة",
    "بدون",
    "ما",
    "أوقف",
    "الكلام",
    "قبل",
    "النهاية",
    "حتى",
    "تسمع",
    "الرد",
)
_PRICE = "الشحن بـ 15.50 ريال، والمحطة قريبة من البيت."


def test_a_clause_is_ready_before_the_sentence_ends() -> None:
    comparison = asyncio.run(compare_utterance(_CLAUSE))

    assert comparison.utterance == _CLAUSE
    assert comparison.tuned.first_token_ms == comparison.baseline.first_token_ms
    assert comparison.tuned.first_audio_ms == _CLAUSE.index("،") + 1
    assert comparison.baseline.first_audio_ms == len(_CLAUSE)
    assert comparison.tuned.first_audio_ms < comparison.baseline.first_audio_ms
    assert " ".join(comparison.tuned.segments) == _CLAUSE
    assert " ".join(comparison.baseline.segments) == _CLAUSE
    assert "الحين" not in comparison.tuned.segments[0]
    assert all(len(segment.split()) >= 2 for segment in comparison.tuned.segments)


def test_a_long_reply_starts_before_the_last_word() -> None:
    text = " ".join(_LONG_WORDS) + "."
    comparison = asyncio.run(compare_utterance(text))
    space_after_word_15 = len(" ".join(_LONG_WORDS[:15]))

    assert comparison.utterance == text
    assert comparison.tuned.first_token_ms == comparison.baseline.first_token_ms == 1
    assert comparison.tuned.first_audio_ms == space_after_word_15 + 1
    assert comparison.baseline.first_audio_ms == len(text)
    assert comparison.tuned.first_audio_ms < comparison.baseline.first_audio_ms
    assert comparison.tuned.segments[0] == " ".join(_LONG_WORDS[:12])
    assert comparison.baseline.segments == (text,)
    assert " ".join(comparison.tuned.segments) == text
    assert all(len(segment.split()) >= 2 for segment in comparison.tuned.segments)


def test_a_long_chunk_is_cut_before_a_later_comma() -> None:
    words = list(_LONG_WORDS)
    words[15] = f"{words[15]}،"
    ready = opening_segments(" ".join(words) + ".")

    assert ready[0] == " ".join(_LONG_WORDS[:12])


def test_the_same_utterance_produces_the_same_timings() -> None:
    first = asyncio.run(compare_utterance(_CLAUSE))
    second = asyncio.run(compare_utterance(_CLAUSE))

    assert first.baseline == second.baseline
    assert first.tuned == second.tuned


def test_a_one_word_sentence_is_not_sent_on_its_own() -> None:
    comparison = asyncio.run(compare_utterance(_ONE_WORD))

    assert comparison.baseline.segments[0] == "نعم."
    assert len(comparison.baseline.segments[0].split()) == 1
    assert comparison.tuned.segments[0] != "نعم."
    assert all(len(segment.split()) >= 2 for segment in comparison.tuned.segments)
    assert comparison.tuned.first_token_ms == comparison.baseline.first_token_ms
    assert " ".join(comparison.tuned.segments) == _ONE_WORD


def test_a_single_word_stays_buffered_until_the_reply_ends() -> None:
    assert opening_segments("نعم.") == ()
    assert spoken_segments("نعم.") == ("نعم.",)


def test_a_decimal_and_a_clock_stay_in_one_phrase() -> None:
    assert opening_segments("الشحن بـ 15.") == ()
    assert spoken_segments("الشحن بـ 15.50 ريال.") == ("الشحن بـ خمسة عشر ريال ونص.",)
    assert spoken_segments("تعال الساعة 3:30 العصر.") == (
        "تعال الساعة ثلاثة ونص العصر.",
    )
    assert " ".join(spoken_segments(_PRICE)) == normalize_for_speech(_PRICE)


def test_the_speaker_uses_phrase_cuts(valid_env: None) -> None:
    from sarjy_agent.config import load_settings
    from sarjy_agent.tts import create_tts

    speaker = create_tts(load_settings())

    assert isinstance(speaker._opts.word_tokenizer, SpeechSegments)
    assert asyncio.run(_held_word(speaker)) == "نعم."


async def _held_word(speaker: object) -> str:
    tokenizer = speaker._opts.word_tokenizer  # type: ignore[attr-defined]
    stream = tokenizer.stream()
    stream.push_text("نعم.")
    assert stream._event_ch.empty()
    stream.end_input()
    try:
        item = await anext(stream)
    finally:
        await stream.aclose()
    return str(item.token)


def test_first_token_and_first_audio_are_logged() -> None:
    stream = StringIO()
    configure_logging(stream)
    timing = StreamTimingLog()
    timing.observe(_llm(0.18))
    timing.observe(_tts(0.25))
    timing.observe(_tts(0.9))
    timing.observe(_llm(0.4))
    timing.observe(_tts(0.1))

    logs = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert [(item["stage"], item["duration_ms"]) for item in logs] == [
        ("llm_first_token", 180.0),
        ("tts_first_audio", 250.0),
        ("llm_first_token", 400.0),
        ("tts_first_audio", 100.0),
    ]
    assert "req-should-not-appear" not in stream.getvalue()


def test_a_missing_token_is_not_logged() -> None:
    stream = StringIO()
    configure_logging(stream)
    timing = StreamTimingLog()
    timing.observe(_llm(-1))
    timing.observe(_tts(-1))

    assert stream.getvalue() == ""


def test_the_session_records_reply_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: dict[str, object] = {}

    def fake_on(_emitter: object, event: str, callback: object) -> object:
        registered["event"] = event
        registered["callback"] = callback
        return callback

    monkeypatch.setattr("sarjy_agent.worker.rtc.EventEmitter.on", fake_on)
    session = rtc.EventEmitter()
    stream = StringIO()
    configure_logging(stream)

    _watch_transcripts(session, _Turns())  # type: ignore[arg-type]
    callback = registered["callback"]
    assert registered["event"] == "metrics_collected"
    assert callable(callback)
    callback(_Metrics(_llm(0.05)))
    callback(_Metrics(_tts(0.2)))

    logs = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [item["stage"] for item in logs] == ["llm_first_token", "tts_first_audio"]
    assert logs[0]["duration_ms"] == 50.0
    assert logs[1]["duration_ms"] == 200.0


class _Turns:
    """Stand-in for the user-turn list. This test does not feed transcripts."""


class _Metrics:
    """The shape of a metrics event: one metrics object."""

    def __init__(self, metrics: object) -> None:
        self.metrics = metrics


def _llm(ttft: float) -> LLMMetrics:
    return LLMMetrics(
        label="llm",
        request_id="req-should-not-appear",
        timestamp=0,
        duration=1,
        ttft=ttft,
        cancelled=False,
        completion_tokens=1,
        prompt_tokens=1,
        prompt_cached_tokens=0,
        total_tokens=2,
        tokens_per_second=1,
    )


def _tts(ttfb: float) -> TTSMetrics:
    return TTSMetrics(
        label="tts",
        request_id="req-should-not-appear",
        timestamp=0,
        ttfb=ttfb,
        duration=1,
        audio_duration=1,
        cancelled=False,
        characters_count=4,
        streamed=True,
    )

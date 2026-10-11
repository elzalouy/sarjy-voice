"""Pauses, short words, and interim text against the initial end-of-turn plan."""

import asyncio

from livekit.agents import inference
from livekit.agents.language import LanguageCode

from sarjy_agent.config import load_settings
from sarjy_agent.turn_detection import (
    INITIAL_PLAN,
    HeardSpan,
    reply_count,
    silence_before_reply,
)
from sarjy_agent.worker import create_voice_session

_FINISHED_WAIT = silence_before_reply(turn_likely_finished=True)
_UNFINISHED_WAIT = silence_before_reply(turn_likely_finished=False)
_NORMAL_PAUSE = _FINISHED_WAIT - 0.25


def test_a_normal_pause_does_not_start_another_reply() -> None:
    spans = (
        HeardSpan("speech", 1.0),
        HeardSpan("silence", _NORMAL_PAUSE),
        HeardSpan("speech", 0.8),
        HeardSpan("silence", _NORMAL_PAUSE),
        HeardSpan("speech", 0.6),
        HeardSpan("silence", _FINISHED_WAIT),
    )

    assert _NORMAL_PAUSE < _FINISHED_WAIT
    assert reply_count(spans, turn_likely_finished=True) == 1


def test_several_pauses_in_one_turn_start_one_reply() -> None:
    spans = [
        HeardSpan("speech", 0.4),
        HeardSpan("silence", _NORMAL_PAUSE),
        HeardSpan("speech", 0.4),
        HeardSpan("silence", _NORMAL_PAUSE),
        HeardSpan("speech", 0.4),
        HeardSpan("silence", _FINISHED_WAIT),
    ]

    assert reply_count(spans, turn_likely_finished=True) == 1


def test_a_short_word_can_finish_and_a_click_cannot() -> None:
    word = (
        HeardSpan("speech", INITIAL_PLAN.min_speech_seconds),
        HeardSpan("silence", _FINISHED_WAIT),
    )
    click = (
        HeardSpan("speech", INITIAL_PLAN.min_speech_seconds / 2),
        HeardSpan("silence", _UNFINISHED_WAIT),
    )

    assert reply_count(word, turn_likely_finished=True) == 1
    assert reply_count(click, turn_likely_finished=True) == 0


def test_interim_text_does_not_start_a_reply() -> None:
    partials = tuple(HeardSpan("partial") for _ in range(5))
    paused = (
        HeardSpan("speech", 0.4),
        *partials,
        HeardSpan("silence", _NORMAL_PAUSE),
        HeardSpan("partial"),
    )

    assert reply_count(partials, turn_likely_finished=True) == 0
    assert reply_count(paused, turn_likely_finished=True) == 0


def test_an_unfinished_turn_waits_through_a_longer_pause() -> None:
    thinking = (
        HeardSpan("speech", 1.0),
        HeardSpan("silence", _FINISHED_WAIT),
    )
    done_waiting = (
        HeardSpan("speech", 1.0),
        HeardSpan("silence", _UNFINISHED_WAIT),
    )

    assert _FINISHED_WAIT < _UNFINISHED_WAIT
    assert reply_count(thinking, turn_likely_finished=False) == 0
    assert reply_count(done_waiting, turn_likely_finished=False) == 1


def test_two_finished_stops_are_two_replies() -> None:
    spans = (
        HeardSpan("speech", 0.3),
        HeardSpan("silence", _FINISHED_WAIT),
        HeardSpan("speech", 0.3),
        HeardSpan("silence", _FINISHED_WAIT),
    )

    assert reply_count(spans, turn_likely_finished=True) == 2


def test_the_session_uses_silero_and_the_arabic_turn_detector(valid_env: None) -> None:
    session = asyncio.run(_session())
    vad = session.vad
    detector = session._turn_detection
    endpointing = session._opts.turn_handling["endpointing"]

    assert isinstance(vad, inference.VAD)
    assert vad.model == "silero"
    assert vad.min_silence_duration == INITIAL_PLAN.min_silence_seconds
    assert vad._opts.min_speech_duration == INITIAL_PLAN.min_speech_seconds
    assert isinstance(detector, inference.TurnDetector)
    assert detector.model == "turn-detector-v1-mini"
    assert endpointing["mode"] == "fixed"
    assert endpointing["min_delay"] == INITIAL_PLAN.min_delay_seconds
    assert endpointing["max_delay"] == INITIAL_PLAN.max_delay_seconds
    interruption = session.options.interruption
    assert interruption["enabled"] is True
    assert interruption["mode"] == "adaptive"
    assert interruption["min_duration"] == 1
    assert interruption["min_words"] == 0
    assert session.interruption_detection == "adaptive"
    assert asyncio.run(detector.supports_language(LanguageCode("ar-SA"))) is True


async def _session() -> object:
    """Open one session inside a running event loop."""
    return create_voice_session(load_settings())

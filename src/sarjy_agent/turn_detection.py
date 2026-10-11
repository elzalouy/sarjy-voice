"""Decide when the person has finished talking.

Silero marks speech and silence. The local turn detector supports Arabic.
A normal pause is not the end of the turn. Interim text does not start a reply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from livekit.agents import inference
from livekit.agents.voice.turn import TurnHandlingOptions

# A click shorter than this is not speech. A short word such as نعم still counts.
_MIN_SPEECH_SECONDS = 0.05
# Silence shorter than this is a breath, not the end of speech.
_MIN_SILENCE_SECONDS = 0.55
# Extra wait after silence when the turn looks finished.
_MIN_DELAY_SECONDS = 0.5
# Extra wait after silence when the turn looks unfinished.
_MAX_DELAY_SECONDS = 2.5
# Speech shorter than this does not stop the agent.
_MIN_INTERRUPTION_SECONDS = 1

_TURN_DETECTOR_VERSION: Literal["v1-mini"] = "v1-mini"
SpanKind = Literal["speech", "silence", "partial"]


@dataclass(frozen=True)
class EndpointingPlan:
    """How long silence must last before the agent may reply."""

    min_speech_seconds: float
    min_silence_seconds: float
    min_delay_seconds: float
    max_delay_seconds: float


@dataclass(frozen=True)
class HeardSpan:
    """Speech, silence, or interim text, in the order it arrives."""

    kind: SpanKind
    seconds: float = 0.0


INITIAL_PLAN = EndpointingPlan(
    min_speech_seconds=_MIN_SPEECH_SECONDS,
    min_silence_seconds=_MIN_SILENCE_SECONDS,
    min_delay_seconds=_MIN_DELAY_SECONDS,
    max_delay_seconds=_MAX_DELAY_SECONDS,
)


def create_vad(plan: EndpointingPlan = INITIAL_PLAN) -> inference.VAD:
    """Build the Silero voice activity detector for this plan."""
    return inference.VAD(
        model="silero",
        min_speech_duration=plan.min_speech_seconds,
        min_silence_duration=plan.min_silence_seconds,
    )


def create_turn_detector() -> inference.TurnDetector:
    """Build the local turn detector. It supports Arabic without a cloud call."""
    return inference.TurnDetector(version=_TURN_DETECTOR_VERSION)


def turn_handling(plan: EndpointingPlan = INITIAL_PLAN) -> TurnHandlingOptions:
    """Return end-of-turn and barge-in settings for one room session."""
    return {
        "turn_detection": create_turn_detector(),
        "endpointing": {
            "mode": "fixed",
            "min_delay": plan.min_delay_seconds,
            "max_delay": plan.max_delay_seconds,
        },
        "interruption": {
            "enabled": True,
            "mode": "adaptive",
            "min_duration": _MIN_INTERRUPTION_SECONDS,
            "min_words": 0,
        },
    }


def silence_before_reply(
    plan: EndpointingPlan = INITIAL_PLAN, *, turn_likely_finished: bool
) -> float:
    """Return how many seconds of silence must pass before a reply."""
    delay = plan.min_delay_seconds if turn_likely_finished else plan.max_delay_seconds
    return plan.min_silence_seconds + delay


def reply_count(
    spans: Sequence[HeardSpan],
    *,
    turn_likely_finished: bool,
    plan: EndpointingPlan = INITIAL_PLAN,
) -> int:
    """Return how many replies this audio would start.

    Interim text does not start a reply. A pause shorter than the wait does
    not start a reply. If speech continues, the wait starts again.
    """
    needed = silence_before_reply(plan, turn_likely_finished=turn_likely_finished)
    replies = 0
    heard_speech = 0.0
    pending_silence = 0.0
    for span in spans:
        if span.kind == "partial":
            continue
        if span.kind == "speech":
            heard_speech += span.seconds
            pending_silence = 0.0
            continue
        if span.kind != "silence":
            raise ValueError("span kind must be speech, silence, or partial")
        pending_silence += span.seconds
        if heard_speech >= plan.min_speech_seconds and pending_silence >= needed:
            replies += 1
            heard_speech = 0.0
            pending_silence = 0.0
    return replies

"""Compare two ElevenLabs runs on the fixed Arabic corpus.

Latency is the time until the first audio frame. Pronunciation is a Nova-3
transcript of that audio, scored like the speech-recognition benchmark.
Naturalness and Saudi accent are not scored here. A person did not listen.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from sarjy_agent.eval.stt_score import CorpusMismatch, ItemScore

PRIMARY_MODEL = "eleven_flash_v2_5"
COMPARED_MODEL = "eleven_multilingual_v2"
_TTFB_LIMIT = 1.25


@dataclass(frozen=True)
class TtsItem:
    """One corpus sentence spoken by one model and one voice."""

    model: str
    voice_id: str
    ttfb_ms: float | None
    audio_s: float | None
    score: ItemScore

    def to_dict(self) -> dict[str, object]:
        score = self.score
        return {
            "item_id": score.item_id,
            "model": self.model,
            "voice_id": self.voice_id,
            "category": score.category,
            "number_kind": score.number_kind,
            "ttfb_ms": self.ttfb_ms,
            "audio_s": self.audio_s,
            "expected_transcript": score.expected_transcript,
            "hypothesis": score.hypothesis,
            "accepted": score.accepted,
            "wer": score.wer,
            "error_type": score.error_type,
        }


@dataclass(frozen=True)
class TtsRun:
    """One model and one voice across the whole corpus, in corpus order."""

    corpus_id: str
    corpus_version: str
    corpus_fingerprint: str
    item_ids: tuple[str, ...]
    model: str
    voice_id: str
    stt_model: str
    stt_locale: str
    items: tuple[TtsItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "corpus_fingerprint": self.corpus_fingerprint,
            "item_ids": list(self.item_ids),
            "model": self.model,
            "voice_id": self.voice_id,
            "stt_model": self.stt_model,
            "stt_locale": self.stt_locale,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class Recommendation:
    """Which model stays the speaker, and why."""

    model: str
    reason: str


def recommend(primary: TtsRun, other: TtsRun) -> Recommendation:
    """Keep Flash v2.5 unless Multilingual v2 is clearly better on this run.

    The switch requires three measured facts:

    - Number sentences are accepted more often.
    - All sentences are accepted at least as often.
    - Median time to the first audio frame is at most 25% higher.

    Accent and naturalness are not part of the rule. They were not judged
    by a listener.
    """
    _same_corpus(primary, other)
    if primary.model != PRIMARY_MODEL or other.model != COMPARED_MODEL:
        raise CorpusMismatch("Compare Flash v2.5 with Multilingual v2 only.")
    if not _same_voice(primary, other):
        return Recommendation(
            PRIMARY_MODEL,
            "The two runs did not use the same voice, so Flash v2.5 stays primary.",
        )
    primary_ttfb = _median(_ttfb_values(primary))
    other_ttfb = _median(_ttfb_values(other))
    if primary_ttfb is None or other_ttfb is None:
        return Recommendation(
            PRIMARY_MODEL,
            "A latency measurement is missing, so Flash v2.5 stays primary.",
        )
    primary_numbers = _accepted_rate(primary, numbers_only=True)
    other_numbers = _accepted_rate(other, numbers_only=True)
    primary_all = _accepted_rate(primary, numbers_only=False)
    other_all = _accepted_rate(other, numbers_only=False)
    faster_enough = other_ttfb <= primary_ttfb * _TTFB_LIMIT
    better_numbers = other_numbers > primary_numbers
    not_worse_overall = other_all >= primary_all
    if faster_enough and better_numbers and not_worse_overall:
        return Recommendation(
            COMPARED_MODEL,
            "Multilingual v2 read more number sentences correctly, "
            "did not lose overall, and was not more than 25% slower "
            "to the first audio frame.",
        )
    return Recommendation(
        PRIMARY_MODEL,
        "The measured pronunciation and latency do not justify a change. "
        "Flash v2.5 stays primary. Accent and naturalness were not judged "
        "by a listener.",
    )


def render_comparison(primary: TtsRun, other: TtsRun) -> str:
    """Write the comparison report for the two runs."""
    choice = recommend(primary, other)
    lines = [
        "# Text-to-speech comparison",
        "",
        "| Field | Flash v2.5 | Multilingual v2 |",
        "| --- | --- | --- |",
        f"| Model | {primary.model} | {other.model} |",
        f"| Voice id | {primary.voice_id} | {other.voice_id} |",
        f"| Corpus | {primary.corpus_id} {primary.corpus_version} | same |",
        f"| Fingerprint | {primary.corpus_fingerprint} | same |",
        f"| Items | {len(primary.items)} | {len(other.items)} |",
        f"| Hearing check | {primary.stt_model} {primary.stt_locale} | same |",
        "",
        "## What was measured",
        "",
        "Both runs speak the same corpus sentences, in the same order, "
        "with the same voice. Numbers are spelled before synthesis, and "
        "provider text normalization is off for both models. "
        "The voice is the configured voice. Both models accepted it, so "
        "only the model changes. The API key cannot list the voice library, "
        "so no other voice was tested.",
        "",
        "Latency is the time from the request to the first audio frame. "
        "The hearing check is Nova-3 listening to that audio. A sentence "
        "is accepted when the transcript matches the expected spoken text "
        "or an accepted writing. This is not a human judgment of "
        "naturalness or of a Saudi accent. No voice is called the best "
        "Saudi voice.",
        "",
        "## Latency",
        "",
        "| Measure | Flash v2.5 | Multilingual v2 |",
        "| --- | --- | --- |",
        _latency_row("Items with a first frame", primary, other, _count),
        _latency_row("Median", primary, other, _median),
        _latency_row("P95", primary, other, lambda values: _percentile(values, 95)),
        _latency_row("Minimum", primary, other, lambda values: min(values)),
        _latency_row("Maximum", primary, other, lambda values: max(values)),
        "",
        "The P95 uses the nearest rank of the measured items in that run.",
        "",
        "## Hearing check",
        "",
        "| Measure | Flash v2.5 | Multilingual v2 |",
        "| --- | --- | --- |",
        _rate_row("Accepted sentences", primary, other, numbers_only=False),
        _rate_row("Accepted number sentences", primary, other, numbers_only=True),
        (
            f"| Mean word error rate | {_mean_wer(primary):.3f} | "
            f"{_mean_wer(other):.3f} |"
        ),
        "",
        "## Number sentences",
        "",
        "| Id | Kind | Flash transcript | Multilingual transcript |",
        "| --- | --- | --- | --- |",
    ]
    other_by_id = {item.score.item_id: item for item in other.items}
    for item in primary.items:
        if item.score.number_kind is None:
            continue
        match = other_by_id.get(item.score.item_id)
        other_text = "" if match is None else match.score.hypothesis
        lines.append(
            f"| {item.score.item_id} | {item.score.number_kind} | "
            f"{_cell(item.score.hypothesis)} | {_cell(other_text)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Primary model: `{choice.model}`.",
            "",
            choice.reason,
            "",
        ]
    )
    return "\n".join(lines)


def render_run(run: TtsRun) -> str:
    """Write one model's report. Every row names the model and voice."""
    lines = [
        f"# Text-to-speech run {run.model}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | {run.model} |",
        f"| Voice id | {run.voice_id} |",
        f"| Corpus | {run.corpus_id} {run.corpus_version} |",
        f"| Fingerprint | {run.corpus_fingerprint} |",
        f"| Hearing check | {run.stt_model} {run.stt_locale} |",
        "",
        "| Id | Model | Voice id | First audio ms | Accepted | Transcript |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in run.items:
        latency = "n/a" if item.ttfb_ms is None else f"{item.ttfb_ms:.0f}"
        lines.append(
            f"| {item.score.item_id} | {item.model} | {item.voice_id} | "
            f"{latency} | {item.score.accepted} | {_cell(item.score.hypothesis)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _same_corpus(primary: TtsRun, other: TtsRun) -> None:
    if (
        primary.corpus_fingerprint != other.corpus_fingerprint
        or primary.corpus_version != other.corpus_version
        or primary.item_ids != other.item_ids
        or tuple(item.score.item_id for item in primary.items) != primary.item_ids
        or tuple(item.score.item_id for item in other.items) != other.item_ids
    ):
        raise CorpusMismatch(
            "Compare runs only when the corpus version, fingerprint, and "
            "item ids are the same."
        )


def _same_voice(primary: TtsRun, other: TtsRun) -> bool:
    return bool(primary.voice_id) and primary.voice_id == other.voice_id


def _ttfb_values(run: TtsRun) -> list[float]:
    return [item.ttfb_ms for item in run.items if item.ttfb_ms is not None]


def _accepted_rate(run: TtsRun, *, numbers_only: bool) -> float:
    chosen = [
        item
        for item in run.items
        if not numbers_only or item.score.number_kind is not None
    ]
    if not chosen:
        return 0.0
    return sum(item.score.accepted for item in chosen) / len(chosen)


def _count(values: list[float]) -> float:
    return float(len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return _percentile(values, 50)


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(percent / 100 * len(ordered)) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return ordered[rank]


def _latency_row(
    label: str,
    primary: TtsRun,
    other: TtsRun,
    summarize: Callable[[list[float]], float | None],
) -> str:
    return (
        f"| {label} | {_ms(_summarize(primary, summarize))} | "
        f"{_ms(_summarize(other, summarize))} |"
    )


def _summarize(
    run: TtsRun, summarize: Callable[[list[float]], float | None]
) -> float | None:
    values = _ttfb_values(run)
    if not values:
        return None
    summary = summarize(values)
    if summary is None:
        return None
    return float(summary)


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value.is_integer():
        return str(int(value))
    return f"{value:.0f} ms"


def _rate_row(label: str, primary: TtsRun, other: TtsRun, *, numbers_only: bool) -> str:
    return (
        f"| {label} | {_rate(primary, numbers_only=numbers_only)} | "
        f"{_rate(other, numbers_only=numbers_only)} |"
    )


def _mean_wer(run: TtsRun) -> float:
    if not run.items:
        return 0.0
    return sum(item.score.wer for item in run.items) / len(run.items)


def _rate(run: TtsRun, *, numbers_only: bool) -> str:
    chosen = [
        item
        for item in run.items
        if not numbers_only or item.score.number_kind is not None
    ]
    if not chosen:
        return "n/a"
    accepted = sum(item.score.accepted for item in chosen)
    return f"{accepted} / {len(chosen)}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")

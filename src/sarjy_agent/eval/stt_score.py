"""Score Deepgram text against the fixed Arabic sentences and write the report."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from sarjy_agent.eval.corpus import EvaluationCorpus, EvaluationItem

MatchKind = Literal["expected", "acceptable"]
ErrorGroup = Literal[
    "names_places",
    "numbers",
    "mixed_language",
    "dialect",
    "other",
]

_TOKEN = re.compile(r"[^\w]+", re.UNICODE)
_HARAKAT = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


class CorpusMismatch(Exception):
    """Two runs cannot be compared because they did not use the same sentences."""


@dataclass(frozen=True)
class ItemScore:
    """One sentence after recognition: the heard text, the score, and the wait."""

    item_id: str
    category: str
    number_kind: str | None
    ambiguity_kind: str | None
    notes: str
    expected_transcript: str
    hypothesis: str
    accepted: bool
    matched: MatchKind | None
    wer: float
    latency_ms: float | None
    error_type: str | None

    @property
    def group(self) -> ErrorGroup:
        """Return names, numbers, mixed language, dialect, or other."""
        return item_group(
            category=self.category,
            ambiguity_kind=self.ambiguity_kind,
            notes=self.notes,
        )


@dataclass(frozen=True)
class SttBenchmark:
    """One full recognition run, in the same order as the fixed sentences."""

    corpus_id: str
    corpus_version: str
    corpus_fingerprint: str
    item_ids: tuple[str, ...]
    stt_model: str
    stt_locale: str
    tts_model: str
    tts_voice_id: str
    audio_note: str
    items: tuple[ItemScore, ...]

    def to_dict(self) -> dict[str, object]:
        """Return this run as a dictionary that can be saved as JSON."""
        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "corpus_fingerprint": self.corpus_fingerprint,
            "item_ids": list(self.item_ids),
            "stt_model": self.stt_model,
            "stt_locale": self.stt_locale,
            "tts_model": self.tts_model,
            "tts_voice_id": self.tts_voice_id,
            "audio": self.audio_note,
            "latency_definition": _LATENCY_DEFINITION,
            "items": [
                {
                    "id": item.item_id,
                    "category": item.category,
                    "number_kind": item.number_kind,
                    "ambiguity_kind": item.ambiguity_kind,
                    "notes": item.notes,
                    "expected_transcript": item.expected_transcript,
                    "hypothesis": item.hypothesis,
                    "accepted": item.accepted,
                    "matched": item.matched,
                    "wer": item.wer,
                    "latency_ms": item.latency_ms,
                    "error_type": item.error_type,
                }
                for item in self.items
            ],
        }


_LATENCY_DEFINITION = (
    "Milliseconds from the last speech sample sent to the streaming "
    "recognizer until the last final transcript. Trailing silence used to "
    "end the utterance is not part of the speech. If no final transcript "
    "arrives, latency is not available."
)


def benchmark_from_dict(raw: object) -> SttBenchmark:
    """Read a saved benchmark. Reject a file that is not this report shape."""
    if not isinstance(raw, dict):
        raise CorpusMismatch("Result file is not an object.")
    item_ids = _string_tuple(raw.get("item_ids"), "item_ids")
    items_raw = raw.get("items")
    if not isinstance(items_raw, list):
        raise CorpusMismatch("Result file has no item list.")
    items = tuple(_item_from_dict(item) for item in items_raw)
    if tuple(item.item_id for item in items) != item_ids:
        raise CorpusMismatch("Result item ids do not match the item list.")
    return SttBenchmark(
        corpus_id=_text(raw.get("corpus_id"), "corpus_id"),
        corpus_version=_text(raw.get("corpus_version"), "corpus_version"),
        corpus_fingerprint=_text(raw.get("corpus_fingerprint"), "corpus_fingerprint"),
        item_ids=item_ids,
        stt_model=_text(raw.get("stt_model"), "stt_model"),
        stt_locale=_text(raw.get("stt_locale"), "stt_locale"),
        tts_model=_text(raw.get("tts_model"), "tts_model"),
        tts_voice_id=_text(raw.get("tts_voice_id"), "tts_voice_id"),
        audio_note=_text(raw.get("audio"), "audio"),
        items=items,
    )


def score_item(
    item: EvaluationItem,
    hypothesis: str,
    *,
    latency_ms: float | None = None,
    error_type: str | None = None,
) -> ItemScore:
    """Score one heard line against the expected text and other accepted writings."""
    compact = _compact(hypothesis)
    matched = _match(item, compact)
    references = (item.expected_transcript, *item.acceptable_transcripts)
    return ItemScore(
        item_id=item.id,
        category=item.category.value,
        number_kind=item.number_kind.value if item.number_kind else None,
        ambiguity_kind=item.ambiguity_kind.value if item.ambiguity_kind else None,
        notes=item.notes,
        expected_transcript=item.expected_transcript,
        hypothesis=compact,
        accepted=matched is not None and error_type is None,
        matched=matched if error_type is None else None,
        wer=_closest_wer(compact, references),
        latency_ms=_latency(latency_ms),
        error_type=error_type,
    )


def score_corpus(
    corpus: EvaluationCorpus,
    hypotheses: dict[str, str],
    *,
    latencies_ms: dict[str, float | None] | None = None,
    error_types: dict[str, str | None] | None = None,
) -> tuple[ItemScore, ...]:
    """Score every sentence in corpus order. A missing line counts as empty."""
    timings = latencies_ms or {}
    errors = error_types or {}
    return tuple(
        score_item(
            item,
            hypotheses.get(item.id, ""),
            latency_ms=timings.get(item.id),
            error_type=errors.get(item.id),
        )
        for item in corpus.items
    )


def item_group(*, category: str, ambiguity_kind: str | None, notes: str) -> ErrorGroup:
    """Choose the report group: names, numbers, mixed language, dialect, or other."""
    if category == "name_place":
        return "names_places"
    if category == "number":
        return "numbers"
    if category == "mixed_language":
        return "mixed_language"
    if ambiguity_kind == "unclear" or "dialect" in notes.lower():
        return "dialect"
    return "other"


def compare_benchmarks(
    baseline: SttBenchmark,
    current: SttBenchmark,
) -> tuple[str, ...]:
    """Return the item ids whose heard text or pass/fail result changed."""
    if (
        baseline.corpus_fingerprint != current.corpus_fingerprint
        or baseline.item_ids != current.item_ids
        or baseline.corpus_version != current.corpus_version
    ):
        raise CorpusMismatch(
            "Compare runs only when the corpus version, fingerprint, and "
            "item ids are the same."
        )
    changed: list[str] = []
    for left, right in zip(baseline.items, current.items, strict=True):
        if left.hypothesis != right.hypothesis or left.accepted != right.accepted:
            changed.append(right.item_id)
    return tuple(changed)


def render_markdown(
    benchmark: SttBenchmark,
    *,
    changed_item_ids: tuple[str, ...] | None = None,
    compared_with: str | None = None,
) -> str:
    """Write the benchmark as a short markdown report."""
    lines = [
        "# Speech recognition benchmark",
        "",
        _settings_table(benchmark),
        "",
        "## Quality",
        "",
        _quality_table(benchmark),
        "",
        "Accepted means the hypothesis equals the expected transcript or one",
        "accepted writing, after trimming and collapsing spaces. Word error",
        "rate ignores punctuation and Arabic vowel marks. A word error rate",
        "of 0 can still be rejected when only punctuation or vowel marks differ.",
        "",
        "## Latency",
        "",
        _latency_section(benchmark),
        "",
        "## Names and places",
        "",
        _group_section(benchmark, "names_places"),
        "",
        "## Numbers",
        "",
        _number_section(benchmark),
        "",
        "## Mixed language",
        "",
        _group_section(benchmark, "mixed_language"),
        "",
        "## Dialect and unclear speech",
        "",
        "This group is items marked `unclear`, plus items whose notes say the",
        "form is dialect.",
        "",
        _group_section(benchmark, "dialect"),
        "",
        "## Other misses",
        "",
        _group_section(benchmark, "other"),
        "",
        "## Comparison",
        "",
        _comparison_section(
            benchmark,
            changed_item_ids,
            compared_with=compared_with,
        ),
        "",
        "## Every item",
        "",
        _item_table(benchmark),
        "",
    ]
    return "\n".join(lines)


def _settings_table(benchmark: SttBenchmark) -> str:
    """Build the table of corpus, model, locale, and speaker."""
    first = benchmark.item_ids[0] if benchmark.item_ids else ""
    last = benchmark.item_ids[-1] if benchmark.item_ids else ""
    rows = (
        ("Corpus", f"{benchmark.corpus_id} {benchmark.corpus_version}"),
        ("Fingerprint", benchmark.corpus_fingerprint),
        ("Items", f"{len(benchmark.item_ids)} ({first} through {last})"),
        ("Item order", "Same order as load_corpus()"),
        ("STT model", benchmark.stt_model),
        ("STT locale", benchmark.stt_locale),
        ("Speaker", benchmark.tts_model),
        ("Speaker voice", benchmark.tts_voice_id),
        ("Audio", _cell(benchmark.audio_note)),
    )
    return _table(("Field", "Value"), rows)


def _quality_table(benchmark: SttBenchmark) -> str:
    """Build the accepted-count and average word-error rows."""
    total = len(benchmark.items)
    accepted = sum(1 for item in benchmark.items if item.accepted)
    rate = 0.0 if total == 0 else accepted / total
    mean_wer = 0.0 if total == 0 else sum(item.wer for item in benchmark.items) / total
    rows = (
        ("Accepted", f"{accepted} / {total} ({rate:.1%})"),
        ("Mean word error rate", f"{mean_wer:.3f}"),
    )
    return _table(("Measure", "Value"), rows)


def _latency_section(benchmark: SttBenchmark) -> str:
    """Describe the wait from speech end to the final transcript."""
    values = [
        item.latency_ms for item in benchmark.items if item.latency_ms is not None
    ]
    intro = _LATENCY_DEFINITION
    if not values:
        return intro + "\n\nLatency was not available for this run."
    rows = (
        ("Items with a measurement", str(len(values))),
        ("Median", f"{_percentile(values, 50):.0f} ms"),
        ("P95", f"{_percentile(values, 95):.0f} ms"),
        ("Minimum", f"{min(values):.0f} ms"),
        ("Maximum", f"{max(values):.0f} ms"),
    )
    note = (
        f"The P95 uses the nearest rank of these {len(values)} items. "
        "It describes this run only."
    )
    return intro + "\n\n" + _table(("Measure", "Value"), rows) + "\n\n" + note


def _group_section(benchmark: SttBenchmark, group: ErrorGroup) -> str:
    """List the misses in one report group."""
    rows = [item for item in benchmark.items if item.group == group]
    misses = [item for item in rows if not item.accepted]
    summary = f"{len(misses)} of {len(rows)} missed."
    if not misses:
        return summary + " No misses."
    return summary + "\n\n" + _miss_table(misses)


def _number_section(benchmark: SttBenchmark) -> str:
    """List phone, price, date, and time misses."""
    rows = [item for item in benchmark.items if item.group == "numbers"]
    misses = [item for item in rows if not item.accepted]
    parts = [f"{len(misses)} of {len(rows)} missed."]
    kinds = ("phone", "price", "date", "time")
    kind_rows = []
    for kind in kinds:
        kind_items = [item for item in rows if item.number_kind == kind]
        kind_misses = [item for item in kind_items if not item.accepted]
        kind_rows.append((kind, f"{len(kind_misses)} of {len(kind_items)}"))
    parts.append(_table(("Kind", "Missed"), tuple(kind_rows)))
    if misses:
        parts.append(_miss_table(misses))
    else:
        parts.append("No misses.")
    return "\n\n".join(parts)


def _comparison_section(
    benchmark: SttBenchmark,
    changed_item_ids: tuple[str, ...] | None,
    *,
    compared_with: str | None = None,
) -> str:
    """Say whether this run matches an earlier run of the same sentences."""
    rule = (
        "A later model or locale must use corpus "
        f"{benchmark.corpus_id} {benchmark.corpus_version}, fingerprint "
        f"{benchmark.corpus_fingerprint}, and the same item ids."
    )
    if changed_item_ids is None:
        return rule + " No earlier result was loaded. This run is the baseline."
    earlier = f" Compared with {compared_with}." if compared_with else ""
    if not changed_item_ids:
        return rule + earlier + " No item changed from the earlier result."
    listed = ", ".join(changed_item_ids)
    return rule + earlier + f" Changed items: {listed}."


def _item_table(benchmark: SttBenchmark) -> str:
    """Build the table of every sentence and what was heard."""
    rows = tuple(
        (
            item.item_id,
            item.category,
            "yes" if item.accepted else "no",
            f"{item.wer:.2f}",
            "" if item.latency_ms is None else f"{item.latency_ms:.0f}",
            _cell(item.hypothesis),
        )
        for item in benchmark.items
    )
    return _table(
        ("Id", "Category", "Accepted", "WER", "Latency ms", "Hypothesis"),
        rows,
    )


def _miss_table(items: list[ItemScore]) -> str:
    """Build a table of the expected text and the heard text for misses."""
    rows = tuple(
        (
            item.item_id,
            item.number_kind or item.ambiguity_kind or item.category,
            f"{item.wer:.2f}",
            _cell(item.expected_transcript),
            _cell(item.hypothesis),
        )
        for item in items
    )
    return _table(("Id", "Kind", "WER", "Expected", "Hypothesis"), rows)


def _table(header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    """Turn headers and rows into a markdown table."""
    head = "| " + " | ".join(header) + " |"
    rule = "| " + " | ".join("---" for _ in header) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _cell(text: str) -> str:
    """Make one table cell safe by replacing vertical bars."""
    return text.replace("|", "/")


def _compact(text: str) -> str:
    """Trim the text and collapse repeated spaces."""
    return " ".join(text.split())


def _match(item: EvaluationItem, hypothesis: str) -> MatchKind | None:
    """Return expected or acceptable when the heard text matches, else nothing."""
    if hypothesis == _compact(item.expected_transcript):
        return "expected"
    accepted = {_compact(value) for value in item.acceptable_transcripts}
    if hypothesis in accepted:
        return "acceptable"
    return None


def _closest_wer(hypothesis: str, references: tuple[str, ...]) -> float:
    """Return the lowest word error rate against the expected or accepted text."""
    if not references:
        return 0.0 if not _tokens(hypothesis) else 1.0
    return min(_wer(hypothesis, reference) for reference in references)


def _wer(hypothesis: str, reference: str) -> float:
    """Return the word error rate of one heard line against one reference line."""
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _distance(hyp, ref) / len(ref)


def word_tokens(text: str) -> list[str]:
    """Split text into words, ignoring punctuation and Arabic vowel marks."""
    return _tokens(text)


def _tokens(text: str) -> list[str]:
    """Split text into words, ignoring punctuation and Arabic vowel marks."""
    stripped = _HARAKAT.sub("", text).replace("\u0640", "")
    return [part for part in _TOKEN.sub(" ", stripped).split() if part]


def _distance(left: list[str], right: list[str]) -> int:
    """Count the word inserts, deletes, and swaps between two word lists."""
    previous = list(range(len(right) + 1))
    for index, token in enumerate(left, start=1):
        current = [index]
        for other_index, other in enumerate(right, start=1):
            cost = 0 if token == other else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _latency(value: float | None) -> float | None:
    """Keep a zero-or-greater latency, or return nothing when it is missing."""
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        return None
    return round(float(value), 1)


def _item_from_dict(raw: object) -> ItemScore:
    """Read one saved sentence score, or raise if the shape is wrong."""
    if not isinstance(raw, dict):
        raise CorpusMismatch("Result item is not an object.")
    matched = raw.get("matched")
    if matched not in ("expected", "acceptable", None):
        raise CorpusMismatch("Result item has an unknown match.")
    latency = raw.get("latency_ms")
    if latency is not None and not isinstance(latency, (int, float)):
        raise CorpusMismatch("Result item has a bad latency.")
    error_type = raw.get("error_type")
    if error_type is not None and not isinstance(error_type, str):
        raise CorpusMismatch("Result item has a bad error type.")
    accepted = raw.get("accepted")
    wer = raw.get("wer")
    hypothesis = raw.get("hypothesis")
    if not isinstance(accepted, bool) or not isinstance(wer, (int, float)):
        raise CorpusMismatch("Result item has a bad score.")
    if not isinstance(hypothesis, str):
        raise CorpusMismatch("Result item is missing hypothesis.")
    return ItemScore(
        item_id=_text(raw.get("id"), "id"),
        category=_text(raw.get("category"), "category"),
        number_kind=_optional_text(raw.get("number_kind")),
        ambiguity_kind=_optional_text(raw.get("ambiguity_kind")),
        notes=_text(raw.get("notes"), "notes"),
        expected_transcript=_text(
            raw.get("expected_transcript"),
            "expected_transcript",
        ),
        hypothesis=hypothesis,
        accepted=accepted,
        matched=matched,
        wer=float(wer),
        latency_ms=None if latency is None else float(latency),
        error_type=error_type,
    )


def _text(value: object, field: str) -> str:
    """Return a required text field, or raise if it is missing."""
    if not isinstance(value, str) or not value:
        raise CorpusMismatch(f"Result file is missing {field}.")
    return value


def _optional_text(value: object) -> str | None:
    """Return optional text, or nothing when the field is empty."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise CorpusMismatch("Result file has a bad optional field.")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    """Return a required list of text values, or raise if it is missing."""
    if not isinstance(value, list) or not value:
        raise CorpusMismatch(f"Result file is missing {field}.")
    ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise CorpusMismatch(f"Result file has a bad {field} entry.")
        ids.append(item)
    return tuple(ids)


def _percentile(values: list[float], percent: float) -> float:
    """Return the nearest-rank percentile of these numbers."""
    ordered = sorted(values)
    rank = math.ceil(percent / 100 * len(ordered)) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return ordered[rank]

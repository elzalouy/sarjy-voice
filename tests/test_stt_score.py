import json

import pytest

from sarjy_agent.eval import load_corpus
from sarjy_agent.eval.stt_score import (
    CorpusMismatch,
    SttBenchmark,
    benchmark_from_dict,
    compare_benchmarks,
    render_markdown,
    score_corpus,
    score_item,
)


def test_expected_text_is_accepted() -> None:
    item = load_corpus().items[0]
    scored = score_item(item, f"  {item.expected_transcript}  ")

    assert scored.accepted
    assert scored.matched == "expected"
    assert scored.wer == 0
    assert scored.hypothesis == item.expected_transcript


def test_digit_writing_is_an_accepted_number() -> None:
    item = next(row for row in load_corpus().items if row.id == "ar-sa-016")
    scored = score_item(item, item.tts_text)

    assert scored.accepted
    assert scored.matched == "acceptable"
    assert scored.group == "numbers"
    assert scored.number_kind == "phone"


def test_vowel_marks_do_not_change_the_word_error_rate() -> None:
    item = next(row for row in load_corpus().items if row.id == "ar-sa-036")
    scored = score_item(item, "شَحْن")

    assert not scored.accepted
    assert scored.wer == 0


def test_punctuation_only_difference_is_not_accepted() -> None:
    item = next(row for row in load_corpus().items if row.id == "ar-sa-006")
    scored = score_item(item, item.expected_transcript.rstrip("؟"))

    assert not scored.accepted
    assert scored.wer == 0


def test_a_wrong_name_is_recorded_as_a_name_miss() -> None:
    item = next(row for row in load_corpus().items if row.id == "ar-sa-011")
    scored = score_item(item, "اسم مختلف")

    assert not scored.accepted
    assert scored.group == "names_places"
    assert scored.wer > 0


def test_dialect_and_unclear_items_share_one_group() -> None:
    corpus = load_corpus()
    dialect = next(row for row in corpus.items if row.id == "ar-sa-032")
    unclear = next(row for row in corpus.items if row.id == "ar-sa-037")

    assert score_item(dialect, "نص آخر").group == "dialect"
    assert score_item(unclear, "نص آخر").group == "dialect"


def test_mixed_language_miss_stays_in_that_group() -> None:
    item = next(row for row in load_corpus().items if row.id == "ar-sa-024")
    scored = score_item(item, "أبي أطلب هاتف من التطبيق")

    assert scored.group == "mixed_language"
    assert not scored.accepted


def test_empty_and_failed_recognition_are_misses() -> None:
    item = load_corpus().items[0]
    empty = score_item(item, "   ")
    failed = score_item(item, item.expected_transcript, error_type="TimeoutError")

    assert empty.hypothesis == ""
    assert not empty.accepted
    assert not failed.accepted
    assert failed.error_type == "TimeoutError"
    assert failed.latency_ms is None


def test_corpus_score_keeps_every_item_in_order() -> None:
    corpus = load_corpus()
    hypotheses = {item.id: item.expected_transcript for item in corpus.items}
    scored = score_corpus(corpus, hypotheses)

    assert [item.item_id for item in scored] == [item.id for item in corpus.items]
    assert all(item.accepted for item in scored)


def test_a_different_corpus_cannot_be_compared() -> None:
    corpus = load_corpus()
    scored = score_corpus(
        corpus,
        {item.id: item.expected_transcript for item in corpus.items},
    )
    baseline = _benchmark(corpus.fingerprint(), scored)
    current = _benchmark("0" * 64, scored)

    with pytest.raises(CorpusMismatch):
        compare_benchmarks(baseline, current)


def test_the_same_corpus_lists_changed_items_only() -> None:
    corpus = load_corpus()
    good = {item.id: item.expected_transcript for item in corpus.items}
    bad = dict(good)
    bad["ar-sa-001"] = "نص آخر"
    baseline = _benchmark(corpus.fingerprint(), score_corpus(corpus, good))
    current = _benchmark(corpus.fingerprint(), score_corpus(corpus, bad))

    assert compare_benchmarks(baseline, current) == ("ar-sa-001",)
    assert compare_benchmarks(baseline, baseline) == ()


def test_report_records_quality_groups_and_the_corpus_lock() -> None:
    corpus = load_corpus()
    scored = score_corpus(
        corpus,
        {item.id: item.expected_transcript for item in corpus.items},
    )
    benchmark = _benchmark(corpus.fingerprint(), scored)
    text = render_markdown(benchmark)

    assert corpus.fingerprint() in text
    assert "nova-3" in text
    assert "ar-SA" in text
    assert "Names and places" in text
    assert "Numbers" in text
    assert "Mixed language" in text
    assert "Dialect and unclear speech" in text
    assert "same item ids" in text
    assert "37 / 37" in text
    restored = benchmark_from_dict(json.loads(json.dumps(benchmark.to_dict())))
    assert restored == benchmark


def _benchmark(fingerprint: str, items: tuple[object, ...]) -> SttBenchmark:
    from sarjy_agent.eval.stt_score import ItemScore

    scored = tuple(item for item in items if isinstance(item, ItemScore))
    return SttBenchmark(
        corpus_id="ar-sa",
        corpus_version="1.0.0",
        corpus_fingerprint=fingerprint,
        item_ids=tuple(item.item_id for item in scored),
        stt_model="nova-3",
        stt_locale="ar-SA",
        tts_model="eleven_flash_v2_5",
        tts_voice_id="voice-test",
        audio_note="Synthesized test audio. Not user audio.",
        items=scored,
    )

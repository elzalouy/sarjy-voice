import pytest

from sarjy_agent.eval import load_corpus
from sarjy_agent.eval.stt_score import CorpusMismatch, score_item
from sarjy_agent.eval.tts_compare import (
    COMPARED_MODEL,
    PRIMARY_MODEL,
    TtsItem,
    TtsRun,
    recommend,
    render_comparison,
    render_run,
)


def test_flash_stays_primary_when_the_other_model_is_slower() -> None:
    choice = recommend(
        _run(PRIMARY_MODEL, accepted=False, ttfb_ms=400),
        _run(COMPARED_MODEL, accepted=True, ttfb_ms=900),
    )

    assert choice.model == PRIMARY_MODEL
    assert "stays primary" in choice.reason
    assert "listener" in choice.reason


def test_flash_stays_primary_when_numbers_are_not_better() -> None:
    choice = recommend(
        _run(PRIMARY_MODEL, accepted=True, ttfb_ms=500),
        _run(COMPARED_MODEL, accepted=True, ttfb_ms=300),
    )

    assert choice.model == PRIMARY_MODEL


def test_multilingual_v2_can_replace_flash_on_measured_facts() -> None:
    choice = recommend(
        _run(PRIMARY_MODEL, accepted=False, ttfb_ms=400),
        _run(COMPARED_MODEL, accepted=True, ttfb_ms=450),
    )

    assert choice.model == COMPARED_MODEL


def test_a_different_voice_does_not_change_the_primary() -> None:
    choice = recommend(
        _run(PRIMARY_MODEL, accepted=False, ttfb_ms=400, voice_id="voice-a"),
        _run(COMPARED_MODEL, accepted=True, ttfb_ms=300, voice_id="voice-b"),
    )

    assert choice.model == PRIMARY_MODEL


def test_a_different_corpus_is_rejected() -> None:
    other = _run(COMPARED_MODEL, accepted=True, ttfb_ms=300)
    changed = TtsRun(
        corpus_id=other.corpus_id,
        corpus_version=other.corpus_version,
        corpus_fingerprint="different",
        item_ids=other.item_ids,
        model=other.model,
        voice_id=other.voice_id,
        stt_model=other.stt_model,
        stt_locale=other.stt_locale,
        items=other.items,
    )

    with pytest.raises(CorpusMismatch):
        recommend(_run(PRIMARY_MODEL, accepted=False, ttfb_ms=400), changed)


def test_report_names_the_model_and_voice_on_every_row() -> None:
    run = _run(PRIMARY_MODEL, accepted=True, ttfb_ms=200)
    text = render_run(run)
    comparison = render_comparison(
        run,
        _run(COMPARED_MODEL, accepted=False, ttfb_ms=250),
    )

    assert "voice-test" in text
    assert PRIMARY_MODEL in text
    for item in run.items:
        row = item.to_dict()
        assert row["model"] == PRIMARY_MODEL
        assert row["voice_id"] == "voice-test"
    assert "No voice is called the best" in comparison
    assert "not a human judgment" in comparison


def _run(
    model: str,
    *,
    accepted: bool,
    ttfb_ms: float,
    voice_id: str = "voice-test",
) -> TtsRun:
    corpus = load_corpus()
    items = []
    for item in corpus.items:
        hypothesis = item.expected_transcript if accepted else "نص مختلف"
        items.append(
            TtsItem(
                model=model,
                voice_id=voice_id,
                ttfb_ms=ttfb_ms,
                audio_s=1.0,
                score=score_item(item, hypothesis),
            )
        )
    return TtsRun(
        corpus_id=corpus.id,
        corpus_version=corpus.version,
        corpus_fingerprint=corpus.fingerprint(),
        item_ids=tuple(item.id for item in corpus.items),
        model=model,
        voice_id=voice_id,
        stt_model="nova-3",
        stt_locale="ar-SA",
        items=tuple(items),
    )

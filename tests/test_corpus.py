from collections import Counter
from importlib.resources import files

import pytest

from sarjy_agent.eval import (
    AUDIO_POLICY,
    COMPARISON_RULE,
    AmbiguityKind,
    CorpusCategory,
    CorpusError,
    NumberKind,
    load_corpus,
    parse_corpus,
    stt_references,
    tts_inputs,
)
from sarjy_agent.eval.corpus import CORPUS_RESOURCE

# Update this only together with EvaluationCorpus.version.
_VERSION_1_0_0_FINGERPRINT = (
    "7a7b374b7817a48782741035f473967acb7e20b7c1cd074c1e5596adbdac92b0"
)

_REQUIRED_SNIPPETS = (
    "السلام عليكم",
    "هلا والله",
    "وين أقرب محطة شحن",
    "افتح لي المحطات القريبة",
    "حي النرجس بالرياض",
    "نورة عبدالله",
    "طريق الملك فهد",
    "غادة",
    "الدمام",
    "مكة",
    "0550000001",
    "+966500000002",
    "15.50 ريال",
    "49.90 ريال",
    "9 أكتوبر 2026",
    "15 رمضان",
    "3:30 العصر",
    "8:00 الصبح",
    "iPhone",
    "Fast Charge",
    "GPS",
    "Bluetooth",
    "battery",
    "Calendar",
    "تمام",
    "على خير",
    "الساعة وحدة",
    "آآ وين ال",
    "همم",
    "خمسطعش دقيقة",
)


def _raw() -> dict[str, object]:
    dumped = load_corpus().model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def _combined_text() -> str:
    parts: list[str] = []
    for item in load_corpus().items:
        parts.append(item.expected_transcript)
        parts.append(item.tts_text)
        parts.extend(item.acceptable_transcripts)
    return "\n".join(parts)


def test_version_lock() -> None:
    corpus = load_corpus()

    assert corpus.schema_version == 1
    assert corpus.id == "ar-sa"
    assert corpus.version == "1.0.0"
    assert corpus.locale == "ar-SA"
    assert corpus.audio_policy == AUDIO_POLICY
    assert corpus.comparison_rule == COMPARISON_RULE
    assert len(corpus.items) == 37
    assert corpus.fingerprint() == _VERSION_1_0_0_FINGERPRINT


def test_load_is_repeatable() -> None:
    first = load_corpus()
    second = load_corpus()

    assert first == second
    assert first.fingerprint() == second.fingerprint()


def test_same_items_for_stt_and_tts() -> None:
    corpus = load_corpus()
    references = stt_references(corpus)
    spoken = tts_inputs(corpus)

    assert [item[0] for item in references] == [item.id for item in corpus.items]
    assert [item[0] for item in spoken] == [item.id for item in corpus.items]
    assert references[0][1] == "السلام عليكم"
    assert spoken[0][1] == "السلام عليكم"


def test_category_coverage() -> None:
    corpus = load_corpus()
    counts = Counter(item.category for item in corpus.items)
    number_kinds = Counter(
        item.number_kind for item in corpus.items if item.number_kind is not None
    )
    ambiguity_kinds = Counter(
        item.ambiguity_kind for item in corpus.items if item.ambiguity_kind is not None
    )

    assert counts == {
        CorpusCategory.GREETING: 5,
        CorpusCategory.QUESTION_INSTRUCTION: 5,
        CorpusCategory.NAME_PLACE: 5,
        CorpusCategory.NUMBER: 8,
        CorpusCategory.MIXED_LANGUAGE: 5,
        CorpusCategory.AMBIGUOUS: 9,
    }
    assert number_kinds == {
        NumberKind.PHONE: 2,
        NumberKind.PRICE: 2,
        NumberKind.DATE: 2,
        NumberKind.TIME: 2,
    }
    assert ambiguity_kinds == {AmbiguityKind.MEANING: 5, AmbiguityKind.UNCLEAR: 4}


def test_required_examples_are_present() -> None:
    text = _combined_text()

    for snippet in _REQUIRED_SNIPPETS:
        assert snippet in text


def test_number_items_keep_spoken_and_digit_forms() -> None:
    phones = []
    for item in load_corpus().items:
        if item.category is not CorpusCategory.NUMBER:
            assert item.expected_transcript == item.tts_text
            continue
        assert item.expected_transcript != item.tts_text
        assert item.tts_text in item.acceptable_transcripts
        assert any(char.isascii() and char.isdigit() for char in item.tts_text)
        assert not any(
            char.isascii() and char.isdigit() for char in item.expected_transcript
        )
        assert "digit" in item.notes.lower()
        if item.number_kind is NumberKind.PHONE:
            phones.append(item)

    assert len(phones) == 2
    assert all("fictional" in item.notes.lower() for item in phones)


def test_data_directory_is_json_only() -> None:
    root = files("sarjy_agent.eval").joinpath("data")
    names = sorted(path.name for path in root.iterdir() if path.is_file())

    assert names == [CORPUS_RESOURCE.removeprefix("data/")]


def test_fingerprint_changes_when_text_changes() -> None:
    raw = _raw()
    items = raw["items"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    first["expected_transcript"] = "السلام عليكم ورحمة الله"
    first["tts_text"] = "السلام عليكم ورحمة الله"

    changed = parse_corpus(raw)

    assert changed.fingerprint() != load_corpus().fingerprint()


def test_invalid_corpus_names_the_problem() -> None:
    extra = _raw()
    extra["audio_path"] = "caller.wav"
    with pytest.raises(
        CorpusError, match="Extra inputs are not permitted"
    ) as extra_info:
        parse_corpus(extra)
    assert extra_info.value.__cause__ is None

    audio = _raw()
    audio_items = audio["items"]
    assert isinstance(audio_items, list)
    audio_note = audio_items[0]
    assert isinstance(audio_note, dict)
    audio_note["notes"] = "see caller.wav"
    with pytest.raises(CorpusError, match="audio files are not part of this corpus"):
        parse_corpus(audio)

    uncovered = _raw()
    uncovered_items = uncovered["items"]
    assert isinstance(uncovered_items, list)
    for item in uncovered_items:
        assert isinstance(item, dict)
        if item["category"] == "greeting":
            item["category"] = "question_instruction"
    with pytest.raises(CorpusError, match="category greeting needs at least"):
        parse_corpus(uncovered)

    with pytest.raises(CorpusError) as type_info:
        parse_corpus(["ar-sa-001"])
    assert type_info.value.__cause__ is None

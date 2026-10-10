import pytest
from livekit.agents.utils import is_given

from sarjy_agent.config import load_settings
from sarjy_agent.eval import CorpusCategory, load_corpus
from sarjy_agent.speech_text import normalize_for_speech
from sarjy_agent.tts import create_tts


def test_evaluation_numbers_are_spoken_without_digit_readings() -> None:
    numbers = [
        item for item in load_corpus().items if item.category is CorpusCategory.NUMBER
    ]

    assert numbers
    for item in numbers:
        assert normalize_for_speech(item.tts_text) == item.expected_transcript


def test_other_evaluation_text_is_unchanged() -> None:
    for item in load_corpus().items:
        if item.category is CorpusCategory.NUMBER:
            continue
        assert normalize_for_speech(item.tts_text) == item.tts_text


def test_eastern_digits_use_the_same_spoken_forms() -> None:
    three = "\u0663"
    zero = "\u0660"
    one = "\u0661"
    five = "\u0665"
    clock = f"تعال الساعة {three}:{three}{zero} العصر"
    day = f"الإجازة تبدأ {one}{five} رمضان"

    assert normalize_for_speech(clock) == "تعال الساعة ثلاثة ونص العصر"
    assert normalize_for_speech(day) == "الإجازة تبدأ خمسة عشر رمضان"


def test_a_bare_number_is_not_rewritten() -> None:
    assert normalize_for_speech("أبي أطلب iPhone 15") == "أبي أطلب iPhone 15"


def test_a_price_without_bi_uses_the_nominative() -> None:
    assert (
        normalize_for_speech("الباقة 49.90 ريال")
        == "الباقة تسعة وأربعون ريال وتسعين هللة"
    )


def test_a_large_price_is_left_unchanged() -> None:
    assert normalize_for_speech("الشحن بـ 150.00 ريال") == "الشحن بـ 150.00 ريال"


def test_speaker_spells_numbers_and_leaves_provider_normalization_off(
    valid_env: None,
) -> None:
    speaker = create_tts(load_settings())

    assert speaker._opts.apply_text_normalization == "off"
    assert not is_given(speaker._opts.apply_language_text_normalization)
    assert speaker._opts.word_tokenizer.tokenize("الشحن بـ 15.50 ريال") == [
        "الشحن بـ خمسة عشر ريال ونص"
    ]


def test_one_shot_synthesis_receives_spoken_numbers(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    def fake(self: object, text: str, *, conn_options: object = None) -> str:
        seen["text"] = text
        return text

    monkeypatch.setattr("livekit.plugins.elevenlabs.TTS.synthesize", fake)
    speaker = create_tts(load_settings())

    speaker.synthesize("الموعد يوم الجمعة 9 أكتوبر 2026")

    assert seen["text"] == "الموعد يوم الجمعة تسعة أكتوبر ألفين وستة وعشرين"

"""Mixed Arabic-English lines keep their English words."""

from sarjy_agent.eval.code_switch import (
    CODE_SWITCH_CASES,
    WHY_NOT_NOVA3_AR,
    WHY_NOT_NOVA3_MULTI,
    accepts_code_switch,
    score_code_switch,
)
from sarjy_agent.eval.stt_score import word_tokens
from sarjy_agent.speech_text import normalize_for_speech


def test_a_line_that_drops_english_words_is_rejected() -> None:
    heard = {
        "cs-product": "أبي أطلب آيفون من التطبيق",
        "cs-technical": "الشاحن حقه يو اس بي",
        "cs-phrase": "وين أقرب محطة شحن",
        "cs-identifier": "رقم الطلب ١٨٤٢",
        "cs-english-phrase": "قول له أرسل الفاتورة اليوم",
    }
    scores = tuple(
        score_code_switch(case, heard[case.case_id]) for case in CODE_SWITCH_CASES
    )

    assert accepts_code_switch(scores) is False
    assert all(not score.accepted for score in scores)


def test_latin_english_words_and_identifiers_are_accepted() -> None:
    scores = tuple(
        score_code_switch(case, case.text, latency_ms=120) for case in CODE_SWITCH_CASES
    )

    assert accepts_code_switch(scores) is True
    assert all(score.wer == 0 for score in scores)
    assert all(score.latency_ms == 120 for score in scores)


def test_the_same_word_split_is_used_as_the_arabic_corpus() -> None:
    text = "وين أقرب charger station"
    assert word_tokens(text) == word_tokens("وين   أقرب، charger station.")


def test_speech_keeps_english_terms_in_latin_letters() -> None:
    for case in CODE_SWITCH_CASES:
        spoken = normalize_for_speech(case.text)
        for term in case.english_terms:
            assert term.casefold() in spoken.casefold()
        for identifier in case.identifiers:
            assert identifier in spoken


def test_nova3_multi_is_not_the_arabic_english_model() -> None:
    assert "Arabic is not in that list" in WHY_NOT_NOVA3_MULTI
    assert "monolingual Arabic" in WHY_NOT_NOVA3_AR
    assert "multi" not in WHY_NOT_NOVA3_AR

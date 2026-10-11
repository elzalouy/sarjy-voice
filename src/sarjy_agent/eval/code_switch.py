"""Fixed Arabic-English lines, scored with the same word rules as the corpus.

Nova-3 ``ar-SA`` is a one-language Arabic model. Nova-3 ``multi`` code-switching
covers ten languages and does not include Arabic. Neither setting is treated
as proof that English words survive inside an Arabic line.
"""

from __future__ import annotations

from dataclasses import dataclass

from sarjy_agent.eval.stt_score import word_tokens

# LiveKit documents this realtime model as accepting a language list
# for code-switched audio. Arabic and English are both in that list.
CODE_SWITCH_MODEL = "gpt-live-transcribe"
CODE_SWITCH_LANGUAGES = ("ar", "en")

WHY_NOT_NOVA3_AR = (
    "Nova-3 ar-SA is a monolingual Arabic model. It is not a code-switching model."
)
WHY_NOT_NOVA3_MULTI = (
    "Nova-3 language=multi code-switches among English, Spanish, French, "
    "German, Hindi, Russian, Portuguese, Japanese, Italian, and Dutch. "
    "Arabic is not in that list."
)


@dataclass(frozen=True)
class CodeSwitchCase:
    """One mixed line and the English words that must stay in Latin letters."""

    case_id: str
    text: str
    english_terms: tuple[str, ...]
    identifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodeSwitchScore:
    """Whether one heard line kept the English words and the identifier."""

    case_id: str
    expected: str
    hypothesis: str
    missing_terms: tuple[str, ...]
    missing_identifiers: tuple[str, ...]
    wer: float
    latency_ms: float | None

    @property
    def accepted(self) -> bool:
        """True when every required English word and identifier is present."""
        return not self.missing_terms and not self.missing_identifiers


CODE_SWITCH_CASES: tuple[CodeSwitchCase, ...] = (
    CodeSwitchCase(
        case_id="cs-product",
        text="أبي أطلب iPhone من التطبيق",
        english_terms=("iPhone",),
    ),
    CodeSwitchCase(
        case_id="cs-technical",
        text="الشاحن حقه USB-C",
        english_terms=("USB-C",),
    ),
    CodeSwitchCase(
        case_id="cs-phrase",
        text="وين أقرب charger station",
        english_terms=("charger", "station"),
    ),
    CodeSwitchCase(
        case_id="cs-identifier",
        text="رقم الطلب ORD-1842",
        english_terms=(),
        identifiers=("ORD-1842",),
    ),
    CodeSwitchCase(
        case_id="cs-english-phrase",
        text="قول له send the invoice today",
        english_terms=("send", "the", "invoice", "today"),
    ),
)


def english_keywords() -> tuple[str, ...]:
    """Return the English terms and identifiers, once each, in case order."""
    seen: list[str] = []
    for case in CODE_SWITCH_CASES:
        for term in (*case.english_terms, *case.identifiers):
            if term not in seen:
                seen.append(term)
    return tuple(seen)


def score_code_switch(
    case: CodeSwitchCase,
    hypothesis: str,
    *,
    latency_ms: float | None = None,
) -> CodeSwitchScore:
    """Score one heard line. English words must remain Latin text."""
    folded = hypothesis.casefold()
    missing_terms = tuple(
        term for term in case.english_terms if term.casefold() not in folded
    )
    missing_ids = tuple(
        identifier for identifier in case.identifiers if identifier not in hypothesis
    )
    return CodeSwitchScore(
        case_id=case.case_id,
        expected=case.text,
        hypothesis=hypothesis,
        missing_terms=missing_terms,
        missing_identifiers=missing_ids,
        wer=_wer(hypothesis, case.text),
        latency_ms=latency_ms,
    )


def accepts_code_switch(scores: tuple[CodeSwitchScore, ...]) -> bool:
    """True when every mixed line kept its English words and identifiers."""
    return bool(scores) and all(score.accepted for score in scores)


def _wer(hypothesis: str, reference: str) -> float:
    """Word error rate using the corpus word split."""
    hyp = [token.casefold() for token in word_tokens(hypothesis)]
    ref = [token.casefold() for token in word_tokens(reference)]
    if not ref:
        return 0.0 if not hyp else 1.0
    return _distance(hyp, ref) / len(ref)


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

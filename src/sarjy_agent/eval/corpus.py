"""Fixed Saudi Arabic evaluation set.

Version 1 is text only. Do not add real user audio. A later version may add
audio only when that version records the speaker's permission.

Speech-to-text and text-to-speech comparisons must use ``load_corpus()``.
They must keep the same version, the same item ids, and this order.
If the text changes, bump ``version`` and update the fingerprint test.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import StrEnum
from importlib.resources import files
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")
_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
_MIN_NUMBER_KIND = 2
_MIN_AMBIGUITY_KIND = 2

CORPUS_RESOURCE = "data/ar_sa_v1.json"
AUDIO_POLICY: Literal["text_only_no_user_audio"] = "text_only_no_user_audio"
COMPARISON_RULE: Literal["same_items_for_every_provider"] = (
    "same_items_for_every_provider"
)


class CorpusError(Exception):
    """Raised when the evaluation corpus is missing or invalid."""


class CorpusCategory(StrEnum):
    """What an example is meant to exercise."""

    GREETING = "greeting"
    QUESTION_INSTRUCTION = "question_instruction"
    NAME_PLACE = "name_place"
    NUMBER = "number"
    MIXED_LANGUAGE = "mixed_language"
    AMBIGUOUS = "ambiguous"


class NumberKind(StrEnum):
    """Which numeric form a number example uses."""

    PHONE = "phone"
    PRICE = "price"
    DATE = "date"
    TIME = "time"


class AmbiguityKind(StrEnum):
    """Why an example is hard to score from the words alone."""

    MEANING = "meaning"
    UNCLEAR = "unclear"


_MIN_CATEGORY: dict[CorpusCategory, int] = {
    CorpusCategory.GREETING: 4,
    CorpusCategory.QUESTION_INSTRUCTION: 4,
    CorpusCategory.NAME_PLACE: 4,
    CorpusCategory.NUMBER: 8,
    CorpusCategory.MIXED_LANGUAGE: 4,
    CorpusCategory.AMBIGUOUS: 6,
}


class EvaluationItem(BaseModel):
    """One authored utterance, shared by speech recognition and synthesis."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    id: str = Field(pattern=r"^ar-sa-\d{3}$")
    category: CorpusCategory
    number_kind: NumberKind | None = None
    ambiguity_kind: AmbiguityKind | None = None
    expected_transcript: str
    tts_text: str
    acceptable_transcripts: tuple[str, ...] = ()
    notes: str

    @field_validator("expected_transcript", "tts_text", "notes")
    @classmethod
    def _single_line_text(cls, value: str) -> str:
        return _plain_text(value)

    @field_validator("acceptable_transcripts")
    @classmethod
    def _alternatives(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_plain_text(item) for item in value)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("alternatives must be unique")
        return cleaned

    @model_validator(mode="after")
    def _check_item(self) -> EvaluationItem:
        if not _ARABIC.search(self.expected_transcript):
            raise ValueError("expected_transcript must contain Arabic")
        if not _ARABIC.search(self.tts_text):
            raise ValueError("tts_text must contain Arabic")
        if self.expected_transcript in self.acceptable_transcripts:
            raise ValueError("acceptable_transcripts must not repeat the expected text")
        _check_category_fields(self)
        return self


class EvaluationCorpus(BaseModel):
    """The versioned set used for every provider comparison."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    schema_version: Literal[1]
    id: Literal["ar-sa"]
    version: str = Field(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
    locale: Literal["ar-SA"]
    audio_policy: Literal["text_only_no_user_audio"]
    comparison_rule: Literal["same_items_for_every_provider"]
    description: str
    items: tuple[EvaluationItem, ...]

    @field_validator("description")
    @classmethod
    def _description_text(cls, value: str) -> str:
        return _plain_text(value)

    @model_validator(mode="after")
    def _check_corpus(self) -> EvaluationCorpus:
        errors: list[str] = []
        _check_ids(self, errors)
        _check_unique_text(self, errors)
        _check_coverage(self, errors)
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def fingerprint(self) -> str:
        """Return a sha256 hex digest of this set's exact text."""
        payload = self.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_corpus() -> EvaluationCorpus:
    """Load the versioned Saudi Arabic evaluation set."""
    text = (
        files("sarjy_agent.eval").joinpath(CORPUS_RESOURCE).read_text(encoding="utf-8")
    )
    raw: object = json.loads(text)
    return parse_corpus(raw)


def parse_corpus(raw: object) -> EvaluationCorpus:
    """Validate corpus data and return it."""
    try:
        return EvaluationCorpus.model_validate(raw)
    except ValidationError as exc:
        raise CorpusError(_format_error(exc)) from None


def stt_references(
    corpus: EvaluationCorpus,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Return ``(id, expected transcript, other accepted writings)`` in order.

    A hypothesis is correct when it equals the expected transcript or one
    accepted writing. Do not require one writing only.
    """
    return tuple(
        (item.id, item.expected_transcript, item.acceptable_transcripts)
        for item in corpus.items
    )


def tts_inputs(corpus: EvaluationCorpus) -> tuple[tuple[str, str], ...]:
    """Return ``(id, text to synthesize)`` in the same order as speech references."""
    return tuple((item.id, item.tts_text) for item in corpus.items)


def _plain_text(value: str) -> str:
    if value != value.strip() or value == "":
        raise ValueError("must be non-empty text without surrounding space")
    if any(char in value for char in "\n\r\t"):
        raise ValueError("must be a single line")
    if _audio_mentioned(value):
        raise ValueError("audio files are not part of this corpus")
    return value


def _audio_mentioned(text: str) -> bool:
    lowered = text.lower()
    return any(suffix in lowered for suffix in _AUDIO_SUFFIXES)


def _has_ascii_digit(text: str) -> bool:
    return any(char.isascii() and char.isdigit() for char in text)


def _check_category_fields(item: EvaluationItem) -> None:
    if item.category is CorpusCategory.NUMBER:
        _check_number_item(item)
    else:
        if item.number_kind is not None:
            raise ValueError("number_kind is only valid on number items")
        if item.expected_transcript != item.tts_text:
            raise ValueError("expected_transcript and tts_text must match")
    if item.category is CorpusCategory.AMBIGUOUS:
        if item.ambiguity_kind is None:
            raise ValueError("ambiguous items require ambiguity_kind")
        if len(item.notes) < 20:
            raise ValueError("ambiguous items need an explanation in notes")
    elif item.ambiguity_kind is not None:
        raise ValueError("ambiguity_kind is only valid on ambiguous items")
    if item.category is CorpusCategory.MIXED_LANGUAGE and not (
        _LATIN.search(item.expected_transcript) and _LATIN.search(item.tts_text)
    ):
        raise ValueError("mixed-language items must contain English letters")


def _check_number_item(item: EvaluationItem) -> None:
    if item.number_kind is None:
        raise ValueError("number items require number_kind")
    if item.expected_transcript == item.tts_text:
        raise ValueError("number items must record spoken text and digit text")
    if not _has_ascii_digit(item.tts_text):
        raise ValueError("tts_text must contain digits")
    if _has_ascii_digit(item.expected_transcript):
        raise ValueError("expected_transcript must spell numbers in words")
    if item.tts_text not in item.acceptable_transcripts:
        raise ValueError("tts_text must be an acceptable transcript")
    if "digit" not in item.notes.lower():
        raise ValueError("notes must explain the digit writing")
    if item.number_kind is NumberKind.PHONE and "fictional" not in item.notes.lower():
        raise ValueError("phone notes must say the number is fictional")


def _check_ids(corpus: EvaluationCorpus, errors: list[str]) -> None:
    ids = [item.id for item in corpus.items]
    expected = [f"ar-sa-{index:03d}" for index in range(1, len(ids) + 1)]
    if ids != expected:
        errors.append("item ids must run from ar-sa-001 in order without gaps")


def _check_unique_text(corpus: EvaluationCorpus, errors: list[str]) -> None:
    expected = [item.expected_transcript for item in corpus.items]
    spoken = [item.tts_text for item in corpus.items]
    if len(expected) != len(set(expected)):
        errors.append("expected_transcript values must be unique")
    if len(spoken) != len(set(spoken)):
        errors.append("tts_text values must be unique")


def _check_coverage(corpus: EvaluationCorpus, errors: list[str]) -> None:
    if set(_MIN_CATEGORY) != set(CorpusCategory):
        errors.append("category minimums do not match the category list")
        return
    counts = Counter(item.category for item in corpus.items)
    for category, minimum in _MIN_CATEGORY.items():
        if counts[category] < minimum:
            errors.append(f"category {category.value} needs at least {minimum} items")
    kind_counts = Counter(
        item.number_kind for item in corpus.items if item.number_kind is not None
    )
    for kind in NumberKind:
        if kind_counts[kind] < _MIN_NUMBER_KIND:
            errors.append(
                f"number kind {kind.value} needs at least {_MIN_NUMBER_KIND} items"
            )
    ambiguity_counts = Counter(
        item.ambiguity_kind for item in corpus.items if item.ambiguity_kind is not None
    )
    for ambiguity in AmbiguityKind:
        if ambiguity_counts[ambiguity] < _MIN_AMBIGUITY_KIND:
            errors.append(
                f"ambiguity kind {ambiguity.value} needs at least "
                f"{_MIN_AMBIGUITY_KIND} items"
            )


def _format_error(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(part) for part in error['loc']) or 'corpus'}: {error['msg']}"
        for error in exc.errors()
    ]
    detail = "; ".join(parts) if parts else "unknown error"
    return f"Evaluation corpus is invalid. {detail}"

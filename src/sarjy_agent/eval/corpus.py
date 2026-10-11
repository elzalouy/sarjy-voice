"""Load evaluation clips from a folder of ``.wav`` and ``.txt`` pairs.

Each recording has one transcript. ``ar-sa-001.wav`` belongs with
``ar-sa-001.txt``. The text file is one line: the words that were spoken.
"""

from __future__ import annotations

import hashlib
import re
import wave
from collections.abc import Iterable
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
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
_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")

DATA_DIR = "data"
AUDIO_POLICY: Literal["wav_with_transcript"] = "wav_with_transcript"
COMPARISON_RULE: Literal["same_items_for_every_provider"] = (
    "same_items_for_every_provider"
)
CORPUS_VERSION = "2.0.0"


class CorpusError(Exception):
    """The evaluation folder is missing a clip or a transcript."""


class CorpusCategory(StrEnum):
    """The kind of sentence, when a test or report needs a group."""

    GREETING = "greeting"
    QUESTION_INSTRUCTION = "question_instruction"
    NAME_PLACE = "name_place"
    NUMBER = "number"
    MIXED_LANGUAGE = "mixed_language"
    AMBIGUOUS = "ambiguous"
    RECORDED = "recorded"


class NumberKind(StrEnum):
    """Which number form a sentence uses: phone, price, date, or time."""

    PHONE = "phone"
    PRICE = "price"
    DATE = "date"
    TIME = "time"


class AmbiguityKind(StrEnum):
    """Why a sentence is hard: the meaning is unclear, or the speech is unclear."""

    MEANING = "meaning"
    UNCLEAR = "unclear"


class EvaluationItem(BaseModel):
    """One clip. The transcript is the text in the matching ``.txt`` file."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        arbitrary_types_allowed=True,
    )

    id: str = Field(pattern=_STEM.pattern)
    category: CorpusCategory = CorpusCategory.RECORDED
    number_kind: NumberKind | None = None
    ambiguity_kind: AmbiguityKind | None = None
    expected_transcript: str
    tts_text: str = ""
    acceptable_transcripts: tuple[str, ...] = ()
    notes: str = "Recorded clip."
    audio_path: Path = Field(default_factory=Path)
    audio_sha256: str = ""

    @field_validator("expected_transcript", "tts_text", "notes")
    @classmethod
    def _single_line_text(cls, value: str) -> str:
        """Require one non-empty line."""
        if value == "":
            return value
        return _plain_text(value)

    @field_validator("acceptable_transcripts")
    @classmethod
    def _alternatives(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Clean other accepted writings and reject duplicates."""
        cleaned = tuple(_plain_text(item) for item in value)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("alternatives must be unique")
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def _copy_transcript(cls, value: object) -> object:
        """Use the transcript as the spoken text when no other text is given."""
        if isinstance(value, dict) and not value.get("tts_text"):
            copied = dict(value)
            copied["tts_text"] = copied.get("expected_transcript", "")
            return copied
        return value

    @model_validator(mode="after")
    def _check_item(self) -> EvaluationItem:
        """Check fields that belong to this category."""
        if not _ARABIC.search(self.expected_transcript):
            raise ValueError("expected_transcript must contain Arabic")
        if not _ARABIC.search(self.tts_text):
            raise ValueError("tts_text must contain Arabic")
        if self.expected_transcript in self.acceptable_transcripts:
            raise ValueError("acceptable_transcripts must not repeat the expected text")
        _check_category_fields(self)
        return self


class EvaluationCorpus(BaseModel):
    """The clips in one folder, in filename order."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        arbitrary_types_allowed=True,
    )

    schema_version: Literal[1] = 1
    id: Literal["ar-sa"] = "ar-sa"
    version: str = CORPUS_VERSION
    locale: Literal["ar-SA"] = "ar-SA"
    audio_policy: Literal["wav_with_transcript"] = AUDIO_POLICY
    comparison_rule: Literal["same_items_for_every_provider"] = COMPARISON_RULE
    description: str = "Wav clips with one transcript each."
    directory: Path = Path()
    items: tuple[EvaluationItem, ...]

    @model_validator(mode="after")
    def _check_corpus(self) -> EvaluationCorpus:
        """Require at least one clip and unique ids and transcripts."""
        if not self.items:
            raise ValueError("the folder has no wav and txt pairs")
        ids = [item.id for item in self.items]
        if ids != sorted(ids):
            raise ValueError("items must be in filename order")
        if len(ids) != len(set(ids)):
            raise ValueError("item ids must be unique")
        transcripts = [item.expected_transcript for item in self.items]
        if len(transcripts) != len(set(transcripts)):
            raise ValueError("transcripts must be unique")
        return self

    def fingerprint(self) -> str:
        """Hash the transcripts and the wav bytes, in item order."""
        lines = [
            f"{item.id}\n{item.expected_transcript}\n{item.audio_sha256}"
            for item in self.items
        ]
        payload = "\n".join(lines).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_corpus(directory: Path | None = None) -> EvaluationCorpus:
    """Load every ``.wav`` that has a ``.txt`` file with the same name.

    With no directory, read ``sarjy_agent/eval/data``. A wav with no txt,
    or a txt with no wav, is an error. Other files in that folder are an error.
    """
    root = _data_directory() if directory is None else directory
    if not root.is_dir():
        raise CorpusError("The evaluation data folder is missing.")
    try:
        return _corpus_from_pairs(root, _pairs(root))
    except ValidationError as exc:
        raise CorpusError(_format_error(exc)) from None
    except ValueError as exc:
        raise CorpusError(f"Evaluation corpus is invalid. {exc}") from None


def stt_references(
    corpus: EvaluationCorpus,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Return each item id, the expected text, and the other accepted writings."""
    return tuple(
        (item.id, item.expected_transcript, item.acceptable_transcripts)
        for item in corpus.items
    )


def tts_inputs(corpus: EvaluationCorpus) -> tuple[tuple[str, str], ...]:
    """Return each item id and the text that should be spoken, in corpus order."""
    return tuple((item.id, item.tts_text) for item in corpus.items)


def _data_directory() -> Path:
    """Return the folder that holds the wav and txt pairs."""
    resource = files("sarjy_agent.eval").joinpath(DATA_DIR)
    if not resource.is_dir():
        raise CorpusError("The evaluation data folder is missing.")
    return Path(str(resource))


def _pairs(directory: Path) -> list[tuple[str, Path, Path]]:
    """Match each wav file with the txt file of the same name."""
    wavs: dict[str, Path] = {}
    texts: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".wav":
            wavs[path.stem] = path
        elif suffix == ".txt":
            texts[path.stem] = path
        else:
            raise CorpusError(f"{path.name} is not a wav or txt file.")
    missing_text = sorted(set(wavs) - set(texts))
    missing_audio = sorted(set(texts) - set(wavs))
    if missing_text:
        names = ", ".join(f"{stem}.txt" for stem in missing_text)
        raise CorpusError(f"Missing transcript files: {names}.")
    if missing_audio:
        names = ", ".join(f"{stem}.wav" for stem in missing_audio)
        raise CorpusError(f"Missing wav files: {names}.")
    if not wavs:
        raise CorpusError("The evaluation data folder has no wav and txt pairs.")
    return [(stem, wavs[stem], texts[stem]) for stem in sorted(wavs)]


def _corpus_from_pairs(
    directory: Path,
    pairs: Iterable[tuple[str, Path, Path]],
) -> EvaluationCorpus:
    """Build one evaluation item from each wav and txt pair."""
    items: list[EvaluationItem] = []
    for stem, wav_path, text_path in pairs:
        if _STEM.fullmatch(stem) is None:
            raise CorpusError(f"{stem} is not a safe file name.")
        _check_wav(wav_path)
        transcript = _read_transcript(text_path)
        items.append(
            EvaluationItem(
                id=stem,
                expected_transcript=transcript,
                tts_text=transcript,
                audio_path=wav_path,
                audio_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            )
        )
    return EvaluationCorpus(directory=directory, items=tuple(items))


def _check_wav(path: Path) -> None:
    """Raise if this file is not mono 16-bit audio."""
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            width = audio.getsampwidth()
            rate = audio.getframerate()
            frames = audio.getnframes()
    except (wave.Error, EOFError):
        raise CorpusError(f"{path.name} is not a PCM wav file.") from None
    if channels != 1 or width != 2 or rate <= 0 or frames <= 0:
        raise CorpusError(f"{path.name} must be mono 16-bit PCM with audio in it.")


def _read_transcript(path: Path) -> str:
    """Read one Arabic line from this txt file."""
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if text.endswith("\r"):
        text = text[:-1]
    try:
        text = _plain_text(text)
    except ValueError as exc:
        raise CorpusError(f"{path.name} {exc}") from None
    if not _ARABIC.search(text):
        raise CorpusError(f"{path.name} must contain Arabic.")
    return text


def _plain_text(value: str) -> str:
    """Return one trimmed line, or raise if it is empty."""
    if value != value.strip() or value == "":
        raise ValueError("must be non-empty text without surrounding space")
    if any(char in value for char in "\n\r\t"):
        raise ValueError("must be a single line")
    return value


def _check_category_fields(item: EvaluationItem) -> None:
    """Check the extra fields that only some categories are allowed to have."""
    if item.category is CorpusCategory.NUMBER:
        _check_number_item(item)
        return
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
    """Require a number item to store both spoken words and a digit writing."""
    if item.number_kind is None:
        raise ValueError("number items require number_kind")
    if item.expected_transcript == item.tts_text:
        raise ValueError("number items must record spoken text and digit text")
    if not any(char.isascii() and char.isdigit() for char in item.tts_text):
        raise ValueError("tts_text must contain digits")
    if any(char.isascii() and char.isdigit() for char in item.expected_transcript):
        raise ValueError("expected_transcript must spell numbers in words")
    if item.tts_text not in item.acceptable_transcripts:
        raise ValueError("tts_text must be an acceptable transcript")
    if "digit" not in item.notes.lower():
        raise ValueError("notes must explain the digit writing")
    if item.number_kind is NumberKind.PHONE and "fictional" not in item.notes.lower():
        raise ValueError("phone notes must say the number is fictional")


def _format_error(exc: ValidationError) -> str:
    """Turn a validation error into one short message."""
    parts = [
        f"{'.'.join(str(part) for part in error['loc']) or 'corpus'}: {error['msg']}"
        for error in exc.errors()
    ]
    detail = "; ".join(parts) if parts else "unknown error"
    return f"Evaluation corpus is invalid. {detail}"

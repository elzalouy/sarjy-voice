import wave
from pathlib import Path

import pytest

from sarjy_agent.eval import AUDIO_POLICY, COMPARISON_RULE, CorpusError, load_corpus
from sarjy_agent.eval.corpus import DATA_DIR


def test_pairs_load_in_filename_order(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ar-sa-002", "وين أقرب محطة شحن")
    _write_pair(tmp_path, "ar-sa-001", "السلام عليكم")

    corpus = load_corpus(tmp_path)

    assert corpus.version == "2.0.0"
    assert corpus.locale == "ar-SA"
    assert corpus.audio_policy == AUDIO_POLICY
    assert corpus.comparison_rule == COMPARISON_RULE
    assert [item.id for item in corpus.items] == ["ar-sa-001", "ar-sa-002"]
    assert corpus.items[0].expected_transcript == "السلام عليكم"
    assert corpus.items[0].tts_text == "السلام عليكم"
    assert corpus.items[0].audio_path.name == "ar-sa-001.wav"
    assert corpus.items[0].category.value == "recorded"


def test_the_same_files_have_the_same_fingerprint(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ar-sa-001", "السلام عليكم")

    assert load_corpus(tmp_path).fingerprint() == load_corpus(tmp_path).fingerprint()


def test_a_changed_transcript_changes_the_fingerprint(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ar-sa-001", "السلام عليكم")
    first = load_corpus(tmp_path).fingerprint()
    transcript = tmp_path / "ar-sa-001.txt"
    transcript.write_text("السلام عليكم ورحمة الله\n", encoding="utf-8")

    assert load_corpus(tmp_path).fingerprint() != first


def test_a_changed_wav_changes_the_fingerprint(tmp_path: Path) -> None:
    _write_pair(tmp_path, "ar-sa-001", "السلام عليكم")
    first = load_corpus(tmp_path).fingerprint()
    _write_wav(tmp_path / "ar-sa-001.wav", mark=9)

    assert load_corpus(tmp_path).fingerprint() != first


def test_a_wav_without_a_transcript_is_an_error(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ar-sa-001.wav")

    with pytest.raises(CorpusError, match=r"ar-sa-001\.txt"):
        load_corpus(tmp_path)


def test_a_transcript_without_a_wav_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "ar-sa-001.txt").write_text("السلام عليكم\n", encoding="utf-8")

    with pytest.raises(CorpusError, match=r"ar-sa-001\.wav"):
        load_corpus(tmp_path)


def test_the_transcript_must_be_one_arabic_line(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ar-sa-001.wav")
    (tmp_path / "ar-sa-001.txt").write_text("hello\n", encoding="utf-8")

    with pytest.raises(CorpusError, match="Arabic"):
        load_corpus(tmp_path)


def test_package_data_has_no_json_list() -> None:
    from importlib.resources import files

    root = files("sarjy_agent.eval").joinpath(DATA_DIR)
    names = [path.name for path in root.iterdir() if path.is_file()]

    assert "ar_sa_v1.json" not in names
    for name in names:
        assert name.endswith(".wav") or name.endswith(".txt")


def _write_pair(directory: Path, stem: str, text: str) -> None:
    _write_wav(directory / f"{stem}.wav")
    (directory / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")


def _write_wav(path: Path, *, mark: int = 1) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes([mark, 0]) * 160)

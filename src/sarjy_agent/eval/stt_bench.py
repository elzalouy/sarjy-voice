"""Send each evaluation wav to Deepgram and save the score."""

from __future__ import annotations

import asyncio
import json
import re
import time
import wave
from pathlib import Path

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import http_context
from livekit.plugins import deepgram

from sarjy_agent.config import load_settings
from sarjy_agent.eval.corpus import EvaluationCorpus, EvaluationItem, load_corpus
from sarjy_agent.eval.stt_score import (
    CorpusMismatch,
    ItemScore,
    SttBenchmark,
    benchmark_from_dict,
    compare_benchmarks,
    render_markdown,
    score_item,
)
from sarjy_agent.log import register_secrets
from sarjy_agent.stt import create_stt

_REPORT_DIR = Path("reports/stt")
_SILENCE_S = 1.0
_CHUNK_MS = 20
_STOP_ON = frozenset({"APIStatusError", "APIConnectionError"})


def main() -> None:
    """Run the corpus and write the markdown and JSON reports."""
    settings = load_settings()
    register_secrets(settings.secret_values())
    corpus = load_corpus()
    try:
        benchmark = asyncio.run(
            _run(
                corpus,
                stt_model=settings.stt_model,
                stt_locale=settings.stt_locale,
            )
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    _write(benchmark)


async def _run(
    corpus: EvaluationCorpus,
    *,
    stt_model: str,
    stt_locale: str,
) -> SttBenchmark:
    """Recognize every wav and return one scored run."""
    settings = load_settings()
    scores: list[ItemScore] = []
    async with http_context.open():
        recognizer = create_stt(settings)
        try:
            for index, item in enumerate(corpus.items, start=1):
                score = await _one(recognizer, item)
                if index == 1 and score.error_type in _STOP_ON:
                    raise RuntimeError(
                        f"The first item failed with {score.error_type}. "
                        "No report was written."
                    )
                scores.append(score)
                latency = (
                    "n/a" if score.latency_ms is None else f"{score.latency_ms:.0f}"
                )
                print(
                    f"{index}/{len(corpus.items)} {item.id} "
                    f"accepted={score.accepted} wer={score.wer:.2f} "
                    f"latency_ms={latency}",
                    flush=True,
                )
        finally:
            await recognizer.aclose()
    return SttBenchmark(
        corpus_id=corpus.id,
        corpus_version=corpus.version,
        corpus_fingerprint=corpus.fingerprint(),
        item_ids=tuple(item.id for item in corpus.items),
        stt_model=stt_model,
        stt_locale=stt_locale,
        tts_model="recording",
        tts_voice_id="wav",
        audio_note=_audio_note(corpus),
        items=tuple(scores),
    )


def _audio_note(corpus: EvaluationCorpus) -> str:
    """Describe the wav files that were sent to the recognizer."""
    return (
        f"{len(corpus.items)} wav files from {corpus.directory}. "
        "The txt file with the same name is the expected transcript. "
        "A digit writing in the corpus is still accepted."
    )


async def _one(
    recognizer: deepgram.STT,
    item: EvaluationItem,
) -> ItemScore:
    """Read one wav, recognize it, and return its score."""
    try:
        audio = _read_wav(item.audio_path)
        recognizer.update_options(sample_rate=audio.sample_rate)
        hypothesis, latency_ms = await _recognize(recognizer, audio)
    except ExceptionGroup as exc:
        inner = exc.exceptions[0] if exc.exceptions else exc
        return score_item(item, "", error_type=type(inner).__name__)
    except Exception as exc:
        return score_item(item, "", error_type=type(exc).__name__)
    return score_item(item, hypothesis, latency_ms=latency_ms)


def _read_wav(path: Path) -> rtc.AudioFrame:
    """Read one mono 16-bit wav into a frame."""
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        rate = audio.getframerate()
        pcm = audio.readframes(audio.getnframes())
    if channels != 1 or width != 2 or not pcm:
        raise RuntimeError("EmptyAudio")
    frame = rtc.AudioFrame.create(rate, channels, len(pcm) // width)
    memoryview(frame.data).cast("B")[: len(pcm)] = pcm
    return frame


async def _recognize(
    recognizer: deepgram.STT,
    audio: rtc.AudioFrame,
) -> tuple[str, float | None]:
    """Send one clip to Deepgram and return the heard text and the wait."""
    stream = recognizer.stream()
    try:
        return await asyncio.wait_for(
            _stream(stream, audio),
            timeout=audio.duration + _SILENCE_S + 20,
        )
    finally:
        await stream.aclose()


async def _stream(
    stream: stt.RecognizeStream,
    audio: rtc.AudioFrame,
) -> tuple[str, float | None]:
    """Play the clip in real time and time the wait until the last final text."""
    finals: list[str] = []
    speech_end: float | None = None
    final_at: float | None = None
    chunk_samples = max(audio.sample_rate * _CHUNK_MS // 1000, 1)

    async def send() -> None:
        """Send speech, then one second of silence so the turn can end."""
        nonlocal speech_end
        for chunk in _chunks(audio, chunk_samples):
            stream.push_frame(chunk)
            await asyncio.sleep(chunk.duration)
        speech_end = time.perf_counter()
        silence = rtc.AudioFrame.create(
            audio.sample_rate,
            audio.num_channels,
            int(audio.sample_rate * _SILENCE_S),
        )
        for chunk in _chunks(silence, chunk_samples):
            stream.push_frame(chunk)
            await asyncio.sleep(chunk.duration)
        stream.end_input()

    async def receive() -> None:
        """Collect each final transcript and remember when the last one arrived."""
        nonlocal final_at
        async for event in stream:
            if event.type != stt.SpeechEventType.FINAL_TRANSCRIPT:
                continue
            if not event.alternatives:
                continue
            text = event.alternatives[0].text.strip()
            if not text:
                continue
            finals.append(text)
            final_at = time.perf_counter()

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(send())
        tasks.create_task(receive())

    hypothesis = _join_finals(finals)
    if speech_end is None or final_at is None or not hypothesis:
        return hypothesis, None
    return hypothesis, max(0.0, (final_at - speech_end) * 1000)


def _chunks(frame: rtc.AudioFrame, samples_per_chunk: int) -> list[rtc.AudioFrame]:
    """Split one audio frame into short pieces that can be sent in real time."""
    count = max(samples_per_chunk, 1)
    channels = frame.num_channels
    width = 2 * channels
    raw = memoryview(frame.data).cast("B")
    chunks: list[rtc.AudioFrame] = []
    start = 0
    total = frame.samples_per_channel
    while start < total:
        take = min(count, total - start)
        piece = rtc.AudioFrame.create(frame.sample_rate, channels, take)
        byte_start = start * width
        byte_count = take * width
        memoryview(piece.data).cast("B")[:byte_count] = raw[
            byte_start : byte_start + byte_count
        ]
        chunks.append(piece)
        start += take
    return chunks


def _join_finals(parts: list[str]) -> str:
    """Join final pieces into one line without repeating the same words."""
    stable = ""
    for part in parts:
        text = " ".join(part.split())
        if not text:
            continue
        if not stable or text == stable or text.startswith(stable):
            stable = text
        elif not stable.endswith(text):
            stable = f"{stable} {text}"
    return stable


def _write(benchmark: SttBenchmark) -> None:
    """Save the markdown and JSON reports, and name any earlier same-corpus run."""
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _slug(benchmark.stt_model, benchmark.stt_locale, benchmark.tts_model)
    json_path = _REPORT_DIR / f"{stem}.json"
    md_path = _REPORT_DIR / f"{stem}.md"
    changed, compared_with = _earlier_run(benchmark, json_path)
    json_path.write_text(
        json.dumps(benchmark.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown(
            benchmark,
            changed_item_ids=changed,
            compared_with=compared_with,
        ),
        encoding="utf-8",
    )
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


def _earlier_run(
    benchmark: SttBenchmark,
    json_path: Path,
) -> tuple[tuple[str, ...] | None, str | None]:
    """Find another saved run of these same sentences and list what changed."""
    for path in sorted(_REPORT_DIR.glob("*.json")):
        if path == json_path:
            continue
        previous = benchmark_from_dict(json.loads(path.read_text(encoding="utf-8")))
        try:
            changed = compare_benchmarks(previous, benchmark)
        except CorpusMismatch:
            continue
        label = f"{previous.tts_model} voice {previous.tts_voice_id}"
        return changed, label
    return None, None


def _slug(model: str, locale: str, speaker: str) -> str:
    """Build a safe file name from the model, locale, and speaker."""
    raw = f"{model}_{locale}_{speaker}"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
    return cleaned or "stt"


if __name__ == "__main__":
    main()

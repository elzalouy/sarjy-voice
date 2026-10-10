"""Run the fixed Arabic corpus through streaming Deepgram and write a report.

ElevenLabs speaks each item's ``tts_text``. The audio is not stored, and it
is not user audio. Scores use that same corpus. An earlier result for the
same corpus stays on disk and is named in the comparison.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import http_context
from livekit.plugins import deepgram, elevenlabs

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
from sarjy_agent.tts import create_tts

_REPORT_DIR = Path("reports/stt")
_SILENCE_S = 1.0
_CHUNK_MS = 20
_STOP_ON = frozenset({"APIStatusError", "APIConnectionError"})


def main() -> None:
    """Recognize the corpus and write the markdown and JSON reports."""
    settings = load_settings()
    register_secrets(settings.secret_values())
    corpus = load_corpus()
    try:
        benchmark = asyncio.run(
            _run(
                corpus,
                stt_model=settings.stt_model,
                stt_locale=settings.stt_locale,
                tts_model=settings.tts_model,
                tts_voice_id=settings.tts_voice_id,
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
    tts_model: str,
    tts_voice_id: str,
) -> SttBenchmark:
    settings = load_settings()
    scores: list[ItemScore] = []
    async with http_context.open():
        speaker = create_tts(settings)
        recognizer = create_stt(settings)
        try:
            for index, item in enumerate(corpus.items, start=1):
                score = await _one(speaker, recognizer, item)
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
            await speaker.aclose()
            await recognizer.aclose()
    return SttBenchmark(
        corpus_id=corpus.id,
        corpus_version=corpus.version,
        corpus_fingerprint=corpus.fingerprint(),
        item_ids=tuple(item.id for item in corpus.items),
        stt_model=stt_model,
        stt_locale=stt_locale,
        tts_model=tts_model,
        tts_voice_id=tts_voice_id,
        audio_note=_audio_note(tts_model, tts_voice_id),
        items=tuple(scores),
    )


def _audio_note(tts_model: str, tts_voice_id: str) -> str:
    return (
        f"ElevenLabs {tts_model} voice {tts_voice_id} spoke each tts_text. "
        "Phones, prices, dates, and times are spelled before synthesis. "
        "The audio is not stored and is not user audio. "
        "A digit writing in the corpus is still accepted."
    )


async def _one(
    speaker: elevenlabs.TTS,
    recognizer: deepgram.STT,
    item: EvaluationItem,
) -> ItemScore:
    try:
        audio = await _synthesize(speaker, item.tts_text)
        hypothesis, latency_ms = await _recognize(recognizer, audio)
    except ExceptionGroup as exc:
        inner = exc.exceptions[0] if exc.exceptions else exc
        return score_item(item, "", error_type=type(inner).__name__)
    except Exception as exc:
        return score_item(item, "", error_type=type(exc).__name__)
    return score_item(item, hypothesis, latency_ms=latency_ms)


async def _synthesize(speaker: elevenlabs.TTS, text: str) -> rtc.AudioFrame:
    stream = speaker.synthesize(text)
    try:
        frame = await stream.collect()
    finally:
        await stream.aclose()
    if frame.samples_per_channel <= 0:
        raise RuntimeError("EmptyAudio")
    return frame


async def _recognize(
    recognizer: deepgram.STT,
    audio: rtc.AudioFrame,
) -> tuple[str, float | None]:
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
    finals: list[str] = []
    speech_end: float | None = None
    final_at: float | None = None
    chunk_samples = max(audio.sample_rate * _CHUNK_MS // 1000, 1)

    async def send() -> None:
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
    """Compare with another saved run of this same corpus, if one exists."""
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
    raw = f"{model}_{locale}_{speaker}"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
    return cleaned or "stt"


if __name__ == "__main__":
    main()

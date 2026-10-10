"""Speak the fixed Arabic corpus with two ElevenLabs models.

The audio is not stored. Nova-3 hears each clip and the transcript is
saved. A person does not score accent or naturalness.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import http_context
from livekit.plugins import deepgram, elevenlabs

from sarjy_agent.config import load_settings
from sarjy_agent.eval.corpus import EvaluationCorpus, EvaluationItem, load_corpus
from sarjy_agent.eval.stt_score import score_item
from sarjy_agent.eval.tts_compare import (
    COMPARED_MODEL,
    PRIMARY_MODEL,
    TtsItem,
    TtsRun,
    render_comparison,
    render_run,
)
from sarjy_agent.log import register_secrets
from sarjy_agent.tts import create_tts

_REPORT_DIR = Path("reports/tts")
_SILENCE_S = 1.0
_CHUNK_MS = 20


def main() -> None:
    """Speak the corpus with Flash v2.5 and Multilingual v2."""
    settings = load_settings()
    register_secrets(settings.secret_values())
    corpus = load_corpus()
    flash, multilingual = asyncio.run(
        _run(
            corpus,
            voice_id=settings.tts_voice_id,
            stt_model=settings.stt_model,
            stt_locale=settings.stt_locale,
        )
    )
    _write(flash, multilingual)


async def _run(
    corpus: EvaluationCorpus,
    *,
    voice_id: str,
    stt_model: str,
    stt_locale: str,
) -> tuple[TtsRun, TtsRun]:
    settings = load_settings()
    async with http_context.open():
        flash_speaker = create_tts(settings, model=PRIMARY_MODEL, voice_id=voice_id)
        other_speaker = create_tts(settings, model=COMPARED_MODEL, voice_id=voice_id)
        recognizer = deepgram.STT(
            model=stt_model,
            language=stt_locale,
            api_key=settings.deepgram_api_key.get_secret_value(),
            sample_rate=flash_speaker.sample_rate,
        )
        try:
            flash = await _model_run(
                corpus,
                speaker=flash_speaker,
                recognizer=recognizer,
                model=PRIMARY_MODEL,
                voice_id=voice_id,
                stt_model=stt_model,
                stt_locale=stt_locale,
            )
            multilingual = await _model_run(
                corpus,
                speaker=other_speaker,
                recognizer=recognizer,
                model=COMPARED_MODEL,
                voice_id=voice_id,
                stt_model=stt_model,
                stt_locale=stt_locale,
            )
        finally:
            await flash_speaker.aclose()
            await other_speaker.aclose()
            await recognizer.aclose()
    return flash, multilingual


async def _model_run(
    corpus: EvaluationCorpus,
    *,
    speaker: elevenlabs.TTS,
    recognizer: deepgram.STT,
    model: str,
    voice_id: str,
    stt_model: str,
    stt_locale: str,
) -> TtsRun:
    items: list[TtsItem] = []
    for index, item in enumerate(corpus.items, start=1):
        spoken = await _speak_one(speaker, recognizer, item, model, voice_id)
        items.append(spoken)
        latency = "n/a" if spoken.ttfb_ms is None else f"{spoken.ttfb_ms:.0f}"
        print(
            f"{model} {index}/{len(corpus.items)} {item.id} "
            f"accepted={spoken.score.accepted} first_audio_ms={latency}",
            flush=True,
        )
    return TtsRun(
        corpus_id=corpus.id,
        corpus_version=corpus.version,
        corpus_fingerprint=corpus.fingerprint(),
        item_ids=tuple(item.id for item in corpus.items),
        model=model,
        voice_id=voice_id,
        stt_model=stt_model,
        stt_locale=stt_locale,
        items=tuple(items),
    )


async def _speak_one(
    speaker: elevenlabs.TTS,
    recognizer: deepgram.STT,
    item: EvaluationItem,
    model: str,
    voice_id: str,
) -> TtsItem:
    try:
        audio, ttfb_ms = await _synthesize(speaker, item.tts_text)
        hypothesis = await _hear(recognizer, audio)
    except Exception as exc:
        return TtsItem(
            model=model,
            voice_id=voice_id,
            ttfb_ms=None,
            audio_s=None,
            score=score_item(item, "", error_type=type(exc).__name__),
        )
    return TtsItem(
        model=model,
        voice_id=voice_id,
        ttfb_ms=ttfb_ms,
        audio_s=audio.duration,
        score=score_item(item, hypothesis),
    )


async def _synthesize(
    speaker: elevenlabs.TTS, text: str
) -> tuple[rtc.AudioFrame, float]:
    started = time.perf_counter()
    first_ms: float | None = None
    frames: list[rtc.AudioFrame] = []
    stream = speaker.synthesize(text)
    try:
        async for event in stream:
            if first_ms is None:
                first_ms = (time.perf_counter() - started) * 1000
            frames.append(event.frame)
    finally:
        await stream.aclose()
    if first_ms is None or not frames:
        raise RuntimeError("EmptyAudio")
    return _join(frames), first_ms


async def _hear(recognizer: deepgram.STT, audio: rtc.AudioFrame) -> str:
    stream = recognizer.stream()
    try:
        return await asyncio.wait_for(
            _read(stream, audio),
            timeout=max(audio.duration + _SILENCE_S + 20, 30),
        )
    finally:
        await stream.aclose()


async def _read(stream: stt.RecognizeStream, audio: rtc.AudioFrame) -> str:
    finals: list[str] = []

    async def send() -> None:
        chunk_samples = max(audio.sample_rate * _CHUNK_MS // 1000, 1)
        for chunk in _chunks(audio, chunk_samples):
            stream.push_frame(chunk)
        silence = rtc.AudioFrame.create(
            audio.sample_rate,
            audio.num_channels,
            int(audio.sample_rate * _SILENCE_S),
        )
        for chunk in _chunks(silence, chunk_samples):
            stream.push_frame(chunk)
        stream.end_input()

    async def receive() -> None:
        async for event in stream:
            if event.type != stt.SpeechEventType.FINAL_TRANSCRIPT:
                continue
            if not event.alternatives:
                continue
            text = event.alternatives[0].text.strip()
            if text:
                finals.append(text)

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(send())
        tasks.create_task(receive())
    return _join_finals(finals)


def _join(frames: list[rtc.AudioFrame]) -> rtc.AudioFrame:
    first = frames[0]
    width = 2 * first.num_channels
    samples = sum(frame.samples_per_channel for frame in frames)
    joined = rtc.AudioFrame.create(first.sample_rate, first.num_channels, samples)
    dest = memoryview(joined.data).cast("B")
    offset = 0
    for frame in frames:
        raw = memoryview(frame.data).cast("B")
        count = frame.samples_per_channel * width
        dest[offset : offset + count] = raw[:count]
        offset += count
    return joined


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


def _write(primary: TtsRun, other: TtsRun) -> None:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for run in (primary, other):
        stem = _slug(run.model, run.voice_id)
        json_path = _REPORT_DIR / f"{stem}.json"
        md_path = _REPORT_DIR / f"{stem}.md"
        json_path.write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(render_run(run), encoding="utf-8")
        print(f"wrote {md_path}")
    comparison = _REPORT_DIR / "comparison.md"
    comparison.write_text(render_comparison(primary, other), encoding="utf-8")
    print(f"wrote {comparison}")


def _slug(model: str, voice_id: str) -> str:
    raw = f"{model}_{voice_id}"
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "-" for char in raw
    ).strip("-")
    return cleaned or "tts"


if __name__ == "__main__":
    main()

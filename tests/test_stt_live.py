"""Real Deepgram session for one short Arabic greeting.

Skipped unless ``SARJY_LIVE_STT=1``. The check uses the key in ``.env`` and
does not print it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import wave
from pathlib import Path

import pytest
from livekit import rtc
from livekit.agents.utils import http_context
from livekit.agents.voice.events import UserInputTranscribedEvent
from livekit.agents.voice.io import AudioInput

from sarjy_agent.config import load_settings
from sarjy_agent.log import register_secrets
from sarjy_agent.worker import Assistant, create_voice_session

_PHRASE = "السلام عليكم"
_LIVE = os.environ.get("SARJY_LIVE_STT") == "1"


class _PacedAudio(AudioInput):
    """Yield stored frames at their own duration, then stop."""

    def __init__(self, frames: list[rtc.AudioFrame]) -> None:
        super().__init__(label="arabic-speech")
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        for frame in frames:
            self._queue.put_nowait(frame)
        self._queue.put_nowait(None)

    async def __anext__(self) -> rtc.AudioFrame:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        await asyncio.sleep(item.duration)
        return item


def test_arabic_speech_emits_transcription_events(tmp_path: Path) -> None:
    if not _LIVE:
        pytest.skip("set SARJY_LIVE_STT=1 to run a real Deepgram session")

    pcm, sample_rate = _arabic_pcm(tmp_path / "greeting.wav")
    frames = _frames(pcm, sample_rate)
    transcripts = asyncio.run(_run_session(frames))
    text = " ".join(item.transcript for item in transcripts)
    finals = [item.transcript for item in transcripts if item.is_final]
    print(f"transcription_events={len(transcripts)} final={finals[-1:]!r}")

    assert transcripts, "the session produced no transcription events"
    assert any(item.is_final for item in transcripts)
    assert _has_arabic(text), text


async def _run_session(frames: list[rtc.AudioFrame]) -> list[UserInputTranscribedEvent]:
    settings = load_settings()
    register_secrets(settings.secret_values())
    session = create_voice_session(settings)
    heard: list[UserInputTranscribedEvent] = []
    done = asyncio.Event()

    def on_transcript(event: UserInputTranscribedEvent) -> None:
        if not event.transcript.strip():
            return
        heard.append(event)
        if event.is_final:
            done.set()

    session.on("user_input_transcribed", on_transcript)
    session.input.audio = _PacedAudio(frames)
    # The worker opens this context in each job process. A standalone session
    # has to open it itself.
    async with http_context.open():
        try:
            await session.start(Assistant())
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(done.wait(), timeout=30)
        except Exception as exc:
            raise _public_error(exc, settings.secret_values()) from None
        finally:
            with contextlib.suppress(Exception):
                await session.aclose()
    return heard


def _arabic_pcm(path: Path) -> tuple[bytes, int]:
    subprocess.run(
        [
            "say",
            "-v",
            "Majed",
            "-o",
            str(path),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            _PHRASE,
        ],
        check=True,
    )
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise RuntimeError("the greeting audio must be mono 16-bit PCM")
        return audio.readframes(audio.getnframes()), audio.getframerate()


def _frames(pcm: bytes, sample_rate: int) -> list[rtc.AudioFrame]:
    silence = bytes(sample_rate * 2)  # one second, so the utterance can end
    pcm += silence
    samples_per_frame = sample_rate // 50
    frame_bytes = samples_per_frame * 2
    frames: list[rtc.AudioFrame] = []
    for offset in range(0, len(pcm), frame_bytes):
        chunk = pcm[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk += bytes(frame_bytes - len(chunk))
        frame = rtc.AudioFrame.create(sample_rate, 1, samples_per_frame)
        memoryview(frame.data).cast("B")[:frame_bytes] = chunk
        frames.append(frame)
    return frames


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def _public_error(exc: BaseException, secrets: tuple[str, ...]) -> RuntimeError:
    """Return an error that cannot contain a provider key."""
    detail = str(exc)
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "***")
    return RuntimeError(f"{type(exc).__name__}: {detail[:200]}")

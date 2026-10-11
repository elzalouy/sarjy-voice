"""One spoken Arabic turn: hear it, answer it, and play the reply.

Skipped unless ``SARJY_LIVE_VOICE=1``. The check uses the keys in ``.env``
and does not print them.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import time
import wave
from pathlib import Path

import pytest
from livekit import rtc
from livekit.agents.llm import ChatMessage
from livekit.agents.utils import http_context
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.io import AudioInput, AudioOutput, AudioOutputCapabilities

from sarjy_agent.config import load_settings
from sarjy_agent.log import register_secrets
from sarjy_agent.worker import Assistant, create_voice_session

_PHRASE = "السلام عليكم"
_LIVE = os.environ.get("SARJY_LIVE_VOICE") == "1"
_REPLY_TIMEOUT_SECONDS = 60.0


class _PacedAudio(AudioInput):
    """Play the utterance, then keep sending silence so the turn can end."""

    def __init__(self, frames: list[rtc.AudioFrame], *, sample_rate: int) -> None:
        super().__init__(label="arabic-speech")
        self._queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()
        for frame in frames:
            self._queue.put_nowait(frame)
        samples = sample_rate // 50
        silence = rtc.AudioFrame.create(sample_rate, 1, samples)
        memoryview(silence.data).cast("B")[:] = bytes(samples * 2)
        self._silence = silence

    async def __anext__(self) -> rtc.AudioFrame:
        frame = self._queue.get_nowait() if not self._queue.empty() else self._silence
        await asyncio.sleep(frame.duration)
        return frame


class _HeardAudio(AudioOutput):
    """Keep the audio frames that would be published to the participant."""

    def __init__(self) -> None:
        super().__init__(
            label="participant",
            capabilities=AudioOutputCapabilities(pause=False),
        )
        self.frames: list[rtc.AudioFrame] = []
        self._segment_open = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Store one played frame and mark the segment as started."""
        await super().capture_frame(frame)
        if not self._segment_open:
            self._segment_open = True
            self.on_playback_started(created_at=time.time())
        self.frames.append(frame)

    def flush(self) -> None:
        """The segment is complete. Report that it played."""
        super().flush()
        self._finish(interrupted=False)

    def clear_buffer(self) -> None:
        """Playback stopped early."""
        super().clear_buffer()
        self._finish(interrupted=True)

    def _finish(self, *, interrupted: bool) -> None:
        if not self._segment_open:
            return
        self._segment_open = False
        duration = sum(frame.duration for frame in self.frames)
        self.on_playback_finished(
            playback_position=duration,
            interrupted=interrupted,
        )


def test_spoken_arabic_is_answered_out_loud(tmp_path: Path) -> None:
    if not _LIVE:
        pytest.skip("set SARJY_LIVE_VOICE=1 to run one spoken Arabic turn")

    pcm, sample_rate = _arabic_pcm(tmp_path / "greeting.wav")
    frames = _frames(pcm, sample_rate)
    reply, heard = asyncio.run(_run_turn(frames, sample_rate=sample_rate))
    duration = sum(frame.duration for frame in heard.frames)
    peak = max((_peak(frame.data) for frame in heard.frames), default=0)
    print(
        f"reply={reply!r} chunks={len(heard.frames)} "
        f"duration_s={duration:.2f} peak={peak}"
    )

    assert _has_arabic(reply)
    assert len(heard.frames) >= 2
    assert duration >= 0.3
    assert peak >= 500


async def _run_turn(
    frames: list[rtc.AudioFrame], *, sample_rate: int
) -> tuple[str, _HeardAudio]:
    settings = load_settings()
    register_secrets(settings.secret_values())
    session = create_voice_session(settings)
    agent = Assistant()
    heard = _HeardAudio()
    done = asyncio.Event()
    replies: list[str] = []

    def on_item(event: ConversationItemAddedEvent) -> None:
        item = event.item
        if not isinstance(item, ChatMessage) or item.role != "assistant":
            return
        text = item.text_content or ""
        if not text.strip():
            return
        replies.append(text)
        done.set()

    session.input.audio = _PacedAudio(frames, sample_rate=sample_rate)
    session.output.audio = heard
    session.on("conversation_item_added", on_item)
    async with http_context.open():
        try:
            await session.start(agent)
            await asyncio.wait_for(done.wait(), timeout=_REPLY_TIMEOUT_SECONDS)
        except TimeoutError:
            raise RuntimeError("timed out waiting for a spoken reply") from None
        except Exception as exc:
            raise _public_error(exc, settings.secret_values()) from None
        finally:
            with contextlib.suppress(Exception):
                await session.aclose()
    if not replies:
        raise RuntimeError("the session produced no spoken reply")
    return replies[0], heard


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


def _peak(data: object) -> int:
    view = memoryview(data)  # type: ignore[arg-type]
    samples = view if view.format == "h" else view.cast("h")
    if not samples:
        return 0
    return max(abs(int(sample)) for sample in samples)


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def _public_error(exc: BaseException, secrets: tuple[str, ...]) -> RuntimeError:
    """Return an error that cannot contain a provider key."""
    detail = str(exc)
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "***")
    return RuntimeError(f"{type(exc).__name__}: {detail[:200]}")

"""Real ElevenLabs stream for one short Arabic greeting.

Skipped unless ``SARJY_LIVE_TTS=1``. The check uses the key in ``.env`` and
does not print it.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest
from livekit.agents.utils import http_context

from sarjy_agent.config import load_settings
from sarjy_agent.log import register_secrets
from sarjy_agent.tts import create_tts

_PHRASE = "السلام عليكم"
_LIVE = os.environ.get("SARJY_LIVE_TTS") == "1"


def test_arabic_speech_streams_audible_audio() -> None:
    if not _LIVE:
        pytest.skip("set SARJY_LIVE_TTS=1 to run a real ElevenLabs stream")

    result = asyncio.run(_speak())
    print(
        "tts_model={model} voice_id={voice_id} language={language} "
        "chunks={chunks} duration_s={duration:.2f} peak={peak} "
        "ttfb_ms={ttfb_ms:.0f}".format(**result)
    )

    assert result["chunks"] >= 2
    assert result["duration"] >= 0.3
    assert result["peak"] >= 500


async def _speak() -> dict[str, object]:
    settings = load_settings()
    register_secrets(settings.secret_values())
    speaker = create_tts(settings)
    started = time.perf_counter()
    first_ms: float | None = None
    chunks = 0
    duration = 0.0
    peak = 0
    try:
        async with http_context.open():
            stream = speaker.stream()
            try:
                stream.push_text(_PHRASE)
                stream.end_input()
                async for event in stream:
                    if first_ms is None:
                        first_ms = (time.perf_counter() - started) * 1000
                    frame = event.frame
                    chunks += 1
                    duration += frame.duration
                    peak = max(peak, _peak(frame.data))
            finally:
                await stream.aclose()
    except Exception as exc:
        raise _public_error(exc, settings.secret_values()) from None
    finally:
        await speaker.aclose()

    language = speaker._opts.language
    return {
        "model": speaker.model,
        "voice_id": speaker._opts.voice_id,
        "language": language.language,
        "chunks": chunks,
        "duration": duration,
        "peak": peak,
        "ttfb_ms": first_ms if first_ms is not None else -1.0,
    }


def _peak(data: object) -> int:
    view = memoryview(data)  # type: ignore[arg-type]
    samples = view if view.format == "h" else view.cast("h")
    if not samples:
        return 0
    return max(abs(int(sample)) for sample in samples)


def _public_error(exc: BaseException, secrets: tuple[str, ...]) -> RuntimeError:
    """Return an error that cannot contain a provider key."""
    detail = str(exc)
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "***")
    return RuntimeError(f"{type(exc).__name__}: {detail[:200]}")

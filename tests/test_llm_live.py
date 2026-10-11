"""One streamed Arabic reply from the OpenAI text model.

Skipped unless ``SARJY_LIVE_LLM=1``. The check uses the key in ``.env`` and
does not print it.
"""

from __future__ import annotations

import asyncio
import os
import re

import pytest

from sarjy_agent.config import load_settings
from sarjy_agent.llm import create_llm, stream_reply
from sarjy_agent.log import register_secrets

_LIVE = os.environ.get("SARJY_LIVE_LLM") == "1"
_ARABIC = re.compile(r"[\u0600-\u06FF]")


def test_text_input_streams_an_arabic_reply() -> None:
    if not _LIVE:
        pytest.skip("set SARJY_LIVE_LLM=1 to run a real OpenAI text reply")

    reply = asyncio.run(_reply())
    print(f"reply={reply!r}")

    assert reply.strip()
    assert _ARABIC.search(reply)


async def _reply() -> str:
    settings = load_settings()
    register_secrets(settings.secret_values())
    model = create_llm(settings)
    try:
        return await stream_reply(model, "قل مرحبا بجملة قصيرة")
    except Exception as exc:
        raise _public_error(exc, settings.secret_values()) from None
    finally:
        await model.aclose()


def _public_error(exc: BaseException, secrets: tuple[str, ...]) -> RuntimeError:
    detail = str(exc)
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "***")
    return RuntimeError(f"{type(exc).__name__}: {detail[:200]}")

"""Fixed prompts against the OpenAI text model.

Skipped unless ``SARJY_LIVE_LLM=1``. The check uses the key in ``.env`` and
does not print it.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from sarjy_agent.config import load_settings
from sarjy_agent.instructions import FIXED_CASES, reply_problems
from sarjy_agent.llm import create_llm, stream_reply
from sarjy_agent.log import register_secrets

_LIVE = os.environ.get("SARJY_LIVE_LLM") == "1"


def test_fixed_prompts_produce_grounded_replies() -> None:
    if not _LIVE:
        pytest.skip("set SARJY_LIVE_LLM=1 to check the fixed Arabic prompts")

    failures = asyncio.run(_replies())

    assert failures == []


async def _replies() -> list[str]:
    settings = load_settings()
    register_secrets(settings.secret_values())
    model = create_llm(settings)
    failures: list[str] = []
    try:
        for case in FIXED_CASES:
            try:
                reply = await stream_reply(
                    model,
                    case.user_text,
                    user_facts=case.user_facts,
                    tool_results=case.tool_results,
                )
            except Exception as exc:
                raise _public_error(exc, settings.secret_values()) from None
            problems = reply_problems(case, reply)
            print(f"{case.case_id}={reply!r} problems={problems}")
            if problems:
                failures.append(f"{case.case_id}: {problems}")
    finally:
        await model.aclose()
    return failures


def _public_error(exc: BaseException, secrets: tuple[str, ...]) -> RuntimeError:
    detail = str(exc)
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "***")
    return RuntimeError(f"{type(exc).__name__}: {detail[:200]}")

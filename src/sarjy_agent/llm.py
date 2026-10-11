"""Streaming text replies from the OpenAI Responses API.

This is a text model. Speech stays on Deepgram and ElevenLabs.
The OpenAI Realtime speech API is not used.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from livekit.plugins.openai import responses

from sarjy_agent.config import Settings
from sarjy_agent.instructions import SYSTEM_PROMPT, build_reply_context

__all__ = ["SYSTEM_PROMPT", "create_llm", "stream_reply"]
_CONNECT_TIMEOUT_SECONDS = 10.0
_WRITE_TIMEOUT_SECONDS = 10.0
_POOL_TIMEOUT_SECONDS = 5.0


def create_llm(settings: Settings) -> responses.LLM:
    """Build a streaming text model from the process settings."""
    read_timeout = settings.llm_timeout_seconds
    return responses.LLM(
        model=settings.llm_model,
        api_key=settings.openai_api_key.get_secret_value(),
        use_websocket=False,
        timeout=httpx.Timeout(
            connect=_CONNECT_TIMEOUT_SECONDS,
            read=read_timeout,
            write=_WRITE_TIMEOUT_SECONDS,
            pool=_POOL_TIMEOUT_SECONDS,
        ),
    )


async def stream_reply(
    model: responses.LLM,
    user_text: str,
    *,
    user_facts: Sequence[str] = (),
    tool_results: Sequence[str] = (),
) -> str:
    """Stream one Arabic reply and return the full text."""
    chat = build_reply_context(
        user_text,
        user_facts=user_facts,
        tool_results=tool_results,
    )
    parts: list[str] = []
    stream = model.chat(chat_ctx=chat)
    async with stream:
        async for chunk in stream:
            if chunk.delta is None or not chunk.delta.content:
                continue
            parts.append(chunk.delta.content)
    return "".join(parts)

import asyncio

import pytest
from livekit.plugins.openai import responses

from conftest import VALID_ENV
from sarjy_agent.config import load_settings
from sarjy_agent.llm import SYSTEM_PROMPT, create_llm
from sarjy_agent.worker import Assistant, create_voice_session


def test_text_model_is_not_the_realtime_speech_api(valid_env: None) -> None:
    model = create_llm(load_settings())

    assert isinstance(model, responses.LLM)
    assert "realtime" not in type(model).__module__
    assert model.model == "gpt-4.1-mini"
    assert model._opts.use_websocket is False
    assert model._client.timeout.read == 30.0
    assert model._client.api_key == VALID_ENV["OPENAI_API_KEY"]


def test_model_and_timeout_come_from_settings(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12")

    model = create_llm(load_settings())

    assert model.model == "gpt-4.1"
    assert model._client.timeout.read == 12.0


def test_plugin_uses_the_settings_key_not_a_later_environment_value(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings()
    monkeypatch.setenv("OPENAI_API_KEY", "later-environment-key")

    model = create_llm(settings)

    assert model._client.api_key == VALID_ENV["OPENAI_API_KEY"]
    assert "later-environment-key" not in repr(settings)


def test_session_uses_the_text_model_and_arabic_prompt(valid_env: None) -> None:
    session = asyncio.run(_session())
    assistant = Assistant()

    assert isinstance(session.llm, responses.LLM)
    assert "realtime" not in type(session.llm).__module__
    assert "Arabic" in assistant.instructions
    assert assistant.instructions == SYSTEM_PROMPT


async def _session() -> object:
    return create_voice_session(load_settings())

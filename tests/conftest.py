from collections.abc import Iterator

import pytest

from sarjy_agent.log import reset_log_state

LEAK_MARKER = "SECRET_SHOULD_NOT_LEAK"

ENV_KEYS = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "ELEVEN_API_KEY",
    "OPENAI_API_KEY",
    "NEXTJS_API_BASE_URL",
    "AGENT_SERVICE_TOKEN",
    "STT_MODEL",
    "STT_LOCALE",
    "TTS_MODEL",
    "TTS_VOICE_ID",
)

VALID_ENV = {
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_API_KEY": f"lk-key-{LEAK_MARKER}",
    "LIVEKIT_API_SECRET": f"lk-secret-{LEAK_MARKER}",
    "DEEPGRAM_API_KEY": f"dg-{LEAK_MARKER}",
    "ELEVEN_API_KEY": f"el-{LEAK_MARKER}",
    "OPENAI_API_KEY": f"sk-{LEAK_MARKER}",
    "NEXTJS_API_BASE_URL": "http://127.0.0.1:3000",
    "AGENT_SERVICE_TOKEN": f"token-{LEAK_MARKER}",
    "TTS_VOICE_ID": "voice-test",
}


@pytest.fixture(autouse=True)
def _reset_log_state() -> Iterator[None]:
    reset_log_state()
    yield
    reset_log_state()


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Use a directory with no .env file and clear known settings."""
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def valid_env(isolated_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a complete valid environment. Secret values include LEAK_MARKER."""
    for key, value in VALID_ENV.items():
        monkeypatch.setenv(key, value)

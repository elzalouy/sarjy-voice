import json
import logging

import pytest

from conftest import LEAK_MARKER, VALID_ENV
from sarjy_agent.config import SettingsError, load_settings
from sarjy_agent.main import main


def test_missing_settings_name_every_required_value(isolated_env: None) -> None:
    with pytest.raises(SettingsError) as exc_info:
        load_settings()

    message = str(exc_info.value)
    assert message == (
        "Missing required settings: LIVEKIT_URL, LIVEKIT_API_KEY, "
        "LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, ELEVEN_API_KEY, OPENAI_API_KEY, "
        "NEXTJS_API_BASE_URL, AGENT_SERVICE_TOKEN, TTS_VOICE_ID."
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_empty_secret_is_reported_as_missing(
    isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in VALID_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(SettingsError) as exc_info:
        load_settings()

    assert str(exc_info.value) == "Missing required settings: OPENAI_API_KEY."


def test_invalid_value_is_not_echoed(
    isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in VALID_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LIVEKIT_URL", LEAK_MARKER)

    with pytest.raises(SettingsError) as exc_info:
        load_settings()

    message = str(exc_info.value)
    assert message == "Invalid settings: LIVEKIT_URL."
    assert LEAK_MARKER not in message
    assert exc_info.value.__context__ is None


def test_valid_settings_keep_speech_defaults(valid_env: None) -> None:
    settings = load_settings()

    assert settings.livekit_url.scheme == "wss"
    assert settings.livekit_url.host == "example.livekit.cloud"
    assert settings.stt_model == "nova-3"
    assert settings.stt_locale == "ar-SA"
    assert settings.tts_model == "eleven_flash_v2_5"
    assert settings.tts_voice_id == "voice-test"
    assert settings.livekit_api_key.get_secret_value() == VALID_ENV["LIVEKIT_API_KEY"]


def test_secret_values_do_not_appear_in_logs(
    valid_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    settings = load_settings()
    logging.getLogger("sarjy_agent.config").info("%s", settings)
    logging.getLogger("sarjy_agent.config").info("%s", settings.model_dump())

    assert LEAK_MARKER not in caplog.text
    assert "***" in repr(settings)


def test_main_exits_when_settings_are_missing(
    isolated_env: None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(settings: object) -> None:
        raise AssertionError(settings)

    monkeypatch.setattr("sarjy_agent.main.run_worker", fail)
    with pytest.raises(SystemExit) as exc_info:
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert "Startup failed. Missing required settings:" in captured.err
    assert "LIVEKIT_API_KEY" in captured.err
    assert LEAK_MARKER not in captured.err
    logs = [
        json.loads(line) for line in captured.err.splitlines() if line.startswith("{")
    ]
    assert [item["event"] for item in logs] == [
        "process_starting",
        "process_start_failed",
    ]
    assert "LIVEKIT_API_KEY" in logs[1]["missing_settings"]
    assert logs[1]["invalid_settings"] == []

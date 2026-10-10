import json

import pytest

from conftest import LEAK_MARKER
from sarjy_agent import __version__
from sarjy_agent.config import Settings
from sarjy_agent.main import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_main_prints_ready(
    valid_env: None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Settings] = []

    def capture(settings: Settings) -> None:
        started.append(settings)

    monkeypatch.setattr("sarjy_agent.main.run_worker", capture)
    main()
    captured = capsys.readouterr()
    assert captured.out == "sarjy-agent ready\n"
    assert len(started) == 1
    logs = [json.loads(line) for line in captured.err.splitlines()]
    assert [item["event"] for item in logs] == [
        "process_starting",
        "timing",
        "process_ready",
    ]
    assert logs[1]["stage"] == "load_settings"
    assert logs[1]["duration_ms"] >= 0
    assert logs[2]["stt_model"] == "nova-3"
    assert logs[2]["stt_locale"] == "ar-SA"
    assert logs[2]["tts_model"] == "eleven_flash_v2_5"
    assert logs[2]["tts_voice_id"] == "voice-test"
    assert logs[2]["room_id"] is None
    assert LEAK_MARKER not in captured.err

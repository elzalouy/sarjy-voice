import sys

import pytest
from livekit.agents import AgentServer, JobExecutorType

from conftest import VALID_ENV
from sarjy_agent.config import load_settings
from sarjy_agent.dispatch import handle_job_request
from sarjy_agent.worker import AGENT_NAME, Assistant, create_server, run_worker


def test_server_uses_validated_credentials_not_the_environment(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings()
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(key)
    monkeypatch.setattr(sys, "argv", ["sarjy-agent", "start"])
    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.delitem(sys.modules, "debugpy", raising=False)

    server = create_server(settings)
    assistant = Assistant()

    assert server._ws_url == "wss://example.livekit.cloud"
    assert server._api_key == VALID_ENV["LIVEKIT_API_KEY"]
    assert server._api_secret == VALID_ENV["LIVEKIT_API_SECRET"]
    assert server._agent_name == AGENT_NAME
    assert server._entrypoint_fnc is not None
    assert server._request_fnc is handle_job_request
    assert server._job_executor_type == JobExecutorType.PROCESS
    assert assistant._tools == []


def test_dev_mode_runs_jobs_in_threads(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["sarjy-agent", "dev"])
    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.delitem(sys.modules, "debugpy", raising=False)

    server = create_server(load_settings())

    assert server._job_executor_type == JobExecutorType.THREAD


def test_run_worker_defaults_to_start(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_app(server: object) -> None:
        captured["server"] = server
        captured["argv"] = sys.argv.copy()

    monkeypatch.setattr("sarjy_agent.worker.cli.run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["sarjy-agent"])
    settings = load_settings()
    run_worker(settings)

    server = captured["server"]
    assert isinstance(server, AgentServer)
    assert captured["argv"] == ["sarjy-agent", "start"]
    assert server._api_key == settings.livekit_api_key.get_secret_value()
    assert server._api_secret == settings.livekit_api_secret.get_secret_value()
    assert server._ws_url == "wss://example.livekit.cloud"


def test_run_worker_keeps_an_explicit_command(
    valid_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_app(server: object) -> None:
        captured["argv"] = sys.argv.copy()

    monkeypatch.setattr("sarjy_agent.worker.cli.run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["sarjy-agent", "dev"])
    run_worker(load_settings())

    assert captured["argv"] == ["sarjy-agent", "dev"]

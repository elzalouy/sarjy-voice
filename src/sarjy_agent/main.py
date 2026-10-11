"""Start the agent process."""

import sys
from time import perf_counter

from sarjy_agent.config import SettingsError, load_settings
from sarjy_agent.log import (
    configure_logging,
    log_lifecycle,
    log_process_ready,
    log_process_start_failed,
    log_timing,
    register_secrets,
)
from sarjy_agent.worker import run_worker


def main() -> None:
    """Load settings, hide secrets in logs, then start the LiveKit worker."""
    configure_logging()
    log_lifecycle("process_starting")
    started = perf_counter()
    try:
        settings = load_settings()
    except SettingsError as exc:
        log_process_start_failed(missing=exc.missing, invalid=exc.invalid)
        print(f"Startup failed. {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    register_secrets(settings.secret_values())
    log_timing(stage="load_settings", duration_ms=(perf_counter() - started) * 1000)
    log_process_ready(
        stt_model=settings.stt_model,
        stt_locale=settings.stt_locale,
        tts_model=settings.tts_model,
        tts_voice_id=settings.tts_voice_id,
    )
    print("sarjy-agent ready")
    run_worker(settings)


if __name__ == "__main__":
    main()

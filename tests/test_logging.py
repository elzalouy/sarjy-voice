import json
import logging
from io import StringIO

import pytest

from conftest import LEAK_MARKER
from sarjy_agent.log import (
    configure_logging,
    log_context,
    log_lifecycle,
    log_process_ready,
    log_provider_failure,
    log_timing,
    register_secrets,
)


def _logs(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_correlation_ids_cover_lifecycle_failure_and_timing() -> None:
    stream = StringIO()
    configure_logging(stream)

    with log_context(room_id="room-1", session_id="session-1"):
        log_lifecycle("session_started")
        with log_context(turn_id="turn-1"):
            log_provider_failure(
                provider="deepgram",
                operation="transcribe",
                error_type="TimeoutError",
                duration_ms=12.5,
            )
            log_timing(stage="stt_finalize", duration_ms=180)
        log_lifecycle("session_ended")

    logs = _logs(stream)
    assert [item["event"] for item in logs] == [
        "session_started",
        "provider_failure",
        "timing",
        "session_ended",
    ]
    assert logs[0]["room_id"] == "room-1"
    assert logs[0]["session_id"] == "session-1"
    assert logs[0]["turn_id"] is None
    assert logs[1]["turn_id"] == "turn-1"
    assert logs[1]["provider"] == "deepgram"
    assert logs[1]["operation"] == "transcribe"
    assert logs[1]["error_type"] == "TimeoutError"
    assert logs[1]["duration_ms"] == 12.5
    assert "error_message" not in logs[1]
    assert logs[2]["stage"] == "stt_finalize"
    assert logs[2]["duration_ms"] == 180
    assert logs[2]["turn_id"] == "turn-1"
    assert logs[3]["turn_id"] is None
    assert logs[3]["room_id"] == "room-1"


def test_secrets_conversations_and_audio_are_omitted() -> None:
    stream = StringIO()
    configure_logging(stream)
    secret = f"sk-{LEAK_MARKER}"
    register_secrets([secret])
    logging.getLogger("sarjy_agent.events").info(
        "provider returned %s",
        secret,
        extra={
            "transcript": f"hello {secret}",
            "audio": b"RAW_AUDIO_BYTES",
        },
    )

    text = stream.getvalue()
    assert LEAK_MARKER not in text
    assert "hello" not in text
    assert "RAW_AUDIO_BYTES" not in text
    assert json.loads(text)["event"] == "redacted_event"


def test_process_ready_redacts_a_secret_used_as_a_voice_id() -> None:
    stream = StringIO()
    configure_logging(stream)
    secret = f"voice{LEAK_MARKER}"
    register_secrets([secret])
    log_process_ready(
        stt_model="nova-3",
        stt_locale="ar-SA",
        tts_model="eleven_flash_v2_5",
        tts_voice_id=secret,
    )

    text = stream.getvalue()
    assert LEAK_MARKER not in text
    assert json.loads(text)["tts_voice_id"] == "***"


def test_rejected_identifier_does_not_include_the_value() -> None:
    with (
        pytest.raises(
            ValueError, match="room_id must be a short identifier"
        ) as exc_info,
        log_context(room_id=f"user said {LEAK_MARKER}"),
    ):
        pass

    assert LEAK_MARKER not in str(exc_info.value)

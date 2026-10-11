"""Write one JSON log line per event, without secrets or spoken text."""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Literal, TextIO

_SAFE_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_CLASS = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_MIN_SUBSTRING_SECRET = 8

_BANNED_PARTS = (
    "transcript",
    "audio",
    "prompt",
    "conversation",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)

_ALLOWED_FIELDS = frozenset(
    {
        "provider",
        "operation",
        "error_type",
        "duration_ms",
        "stage",
        "stt_model",
        "stt_locale",
        "tts_model",
        "tts_voice_id",
        "missing_settings",
        "invalid_settings",
        "participant_id",
        "end_reason",
    }
)

_SIMPLE_EVENTS = frozenset(
    {
        "process_starting",
        "process_stopping",
        "session_started",
        "session_ended",
    }
)

LifecycleEvent = Literal[
    "process_starting",
    "process_stopping",
    "session_started",
    "session_ended",
]


@dataclass(frozen=True)
class LogContext:
    """The room, session, and turn ids attached to the next log lines."""

    room_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None


_log_context: ContextVar[LogContext | None] = ContextVar(
    "sarjy_log_context",
    default=None,
)
_secrets: tuple[str, ...] = ()
_secrets_lock = Lock()


class _JsonStreamHandler(logging.StreamHandler[TextIO]):
    """Write each log record as one JSON line."""

    def __init__(self, stream: TextIO) -> None:
        """Send JSON lines to this stream."""
        super().__init__(stream)
        self.setFormatter(_JsonFormatter())
        self.addFilter(_ContextFilter())


class _ContextFilter(logging.Filter):
    """Add room, session, and turn ids, and drop banned fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep the line, after hiding secrets and unsafe field names."""
        context = _current_context()
        record.__dict__["room_id"] = _public_id(context.room_id)
        record.__dict__["session_id"] = _public_id(context.session_id)
        record.__dict__["turn_id"] = _public_id(context.turn_id)
        record.msg = _safe_event(_rendered_message(record))
        record.args = None
        for key in list(record.__dict__):
            if _is_banned(key):
                del record.__dict__[key]
        return True


class _JsonFormatter(logging.Formatter):
    """Turn one log record into a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one JSON line for this record."""
        payload: dict[str, object] = {
            "timestamp": _timestamp(),
            "level": record.levelname,
            "event": _safe_event(_rendered_message(record)),
            "room_id": _public_id(getattr(record, "room_id", None)),
            "session_id": _public_id(getattr(record, "session_id", None)),
            "turn_id": _public_id(getattr(record, "turn_id", None)),
        }
        extras = _extra_fields(record)
        for key in sorted(extras):
            payload[key] = extras[key]
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(stream: TextIO | None = None) -> None:
    """Send sarjy_agent logs to stderr, or to the given stream."""
    logger = logging.getLogger("sarjy_agent")
    logger.setLevel(logging.INFO)
    if any(isinstance(handler, _JsonStreamHandler) for handler in logger.handlers):
        return
    handler = _JsonStreamHandler(sys.stderr if stream is None else stream)
    logger.addHandler(handler)


def reset_log_state() -> None:
    """Clear log ids, secrets, and handlers. Tests call this between cases."""
    global _secrets
    with _secrets_lock:
        _secrets = ()
    _log_context.set(None)
    logger = logging.getLogger("sarjy_agent")
    for handler in list(logger.handlers):
        if isinstance(handler, _JsonStreamHandler):
            logger.removeHandler(handler)
            handler.close()


def register_secrets(values: Iterable[str]) -> None:
    """Remember secret strings so later log lines replace them."""
    global _secrets
    cleaned = tuple(value for value in values if value)
    with _secrets_lock:
        _secrets = tuple(dict.fromkeys((*_secrets, *cleaned)))


@contextmanager
def log_context(
    *,
    room_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> Iterator[None]:
    """Attach room, session, and turn ids to logs inside this block."""
    current = _current_context()
    updated = LogContext(
        room_id=_keep_id(room_id, current.room_id, "room_id"),
        session_id=_keep_id(session_id, current.session_id, "session_id"),
        turn_id=_keep_id(turn_id, current.turn_id, "turn_id"),
    )
    token = _log_context.set(updated)
    try:
        yield
    finally:
        _log_context.reset(token)


def public_id(value: object) -> str | None:
    """Return the id when it is a short safe token, otherwise nothing."""
    return _public_id(value)


def log_lifecycle(event: LifecycleEvent) -> None:
    """Log one process or session event, such as process_starting."""
    if event not in _SIMPLE_EVENTS:
        raise ValueError("event must be a lifecycle name")
    _emit(logging.INFO, event, {})


ParticipantEvent = Literal["participant_joined", "participant_left"]
SessionEndReason = Literal["shutdown", "cancelled", "connection_lost"]

_PARTICIPANT_EVENTS = frozenset({"participant_joined", "participant_left"})
_END_REASONS = frozenset({"shutdown", "cancelled", "connection_lost"})


def log_participant(event: ParticipantEvent, participant_id: str | None = None) -> None:
    """Log that a remote person joined or left. The id is optional."""
    if event not in _PARTICIPANT_EVENTS:
        raise ValueError("event must be a participant lifecycle name")
    fields: dict[str, object] = {}
    if participant_id is not None:
        fields["participant_id"] = _require_value(participant_id, "participant_id")
    _emit(logging.INFO, event, fields)


def log_session_ended(reason: SessionEndReason) -> None:
    """Log that this process left the room, with a short end reason."""
    if reason not in _END_REASONS:
        raise ValueError("reason must be a session end reason")
    _emit(logging.INFO, "session_ended", {"end_reason": reason})


def log_process_ready(
    *,
    stt_model: str,
    stt_locale: str,
    tts_model: str,
    tts_voice_id: str,
) -> None:
    """Log that the process is ready, including model, locale, and voice id."""
    _emit(
        logging.INFO,
        "process_ready",
        {
            "stt_model": _require_value(stt_model, "stt_model"),
            "stt_locale": _require_value(stt_locale, "stt_locale"),
            "tts_model": _require_value(tts_model, "tts_model"),
            "tts_voice_id": _require_value(tts_voice_id, "tts_voice_id"),
        },
    )


def log_process_start_failed(
    *,
    missing: Sequence[str],
    invalid: Sequence[str],
) -> None:
    """Log a startup failure using setting names, never the values."""
    _emit(
        logging.ERROR,
        "process_start_failed",
        {
            "missing_settings": _env_names(missing),
            "invalid_settings": _env_names(invalid),
        },
    )


def log_provider_failure(
    *,
    provider: str,
    operation: str,
    error_type: str,
    duration_ms: float | None = None,
) -> None:
    """Log a provider failure with a short error name and no message text."""
    fields: dict[str, object] = {
        "provider": _require_token(provider, "provider"),
        "operation": _require_token(operation, "operation"),
        "error_type": _require_class_name(error_type, "error_type"),
    }
    if duration_ms is not None:
        fields["duration_ms"] = _require_duration(duration_ms)
    _emit(logging.ERROR, "provider_failure", fields)


def log_user_turn(turn_id: str) -> None:
    """Log that one user turn was saved. The spoken words are not included."""
    with log_context(turn_id=turn_id):
        _emit(logging.INFO, "user_turn_final", {})


def log_timing(*, stage: str, duration_ms: float) -> None:
    """Log how many milliseconds one named stage took."""
    _emit(
        logging.INFO,
        "timing",
        {
            "stage": _require_token(stage, "stage"),
            "duration_ms": _require_duration(duration_ms),
        },
    )


def _emit(level: int, event: str, fields: dict[str, object]) -> None:
    """Send one event to the sarjy_agent.events logger."""
    logging.getLogger("sarjy_agent.events").log(level, event, extra=fields)


def _current_context() -> LogContext:
    """Return the room, session, and turn ids for this task, or empty ids."""
    return _log_context.get() or LogContext()


def _keep_id(value: str | None, current: str | None, field: str) -> str | None:
    """Keep the current id when the new one is missing."""
    if value is None:
        return current
    return _require_value(value, field)


def _require_token(value: str, field: str) -> str:
    """Return a lowercase event token, or raise if it is not safe to log."""
    if _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a short identifier")
    return value


def _require_class_name(value: str, field: str) -> str:
    """Return a short class name, or raise if it is not safe to log."""
    if _SAFE_CLASS.fullmatch(value) is None:
        raise ValueError(f"{field} must be a short identifier")
    return value


def _require_value(value: str, field: str) -> str:
    """Return a short id, or raise if it is not safe to log."""
    if _SAFE_VALUE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a short identifier")
    return value


def _require_duration(value: float) -> float:
    """Return a zero-or-greater duration in milliseconds, rounded to 3 decimals."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration_ms must be zero or greater")
    if not math.isfinite(value) or value < 0:
        raise ValueError("duration_ms must be zero or greater")
    return round(float(value), 3)


def _env_names(values: Sequence[str]) -> list[str]:
    """Keep only names that look like environment variable names."""
    return [value for value in values if _ENV_NAME.fullmatch(value)]


def _rendered_message(record: logging.LogRecord) -> str:
    """Return the log message text, or a placeholder if formatting fails."""
    try:
        rendered = record.getMessage()
    except Exception:
        return "redacted_event"
    if not isinstance(rendered, str):
        return "redacted_event"
    return _redact_text(rendered)


def _safe_event(message: str) -> str:
    """Return the event name, or redacted_event when the text is not a short token."""
    if _SAFE_TOKEN.fullmatch(message):
        return message
    return "redacted_event"


def _public_id(value: object) -> str | None:
    """Return the id when it is a short safe token, otherwise nothing."""
    if isinstance(value, str) and _SAFE_VALUE.fullmatch(value):
        return value
    return None


def _is_banned(key: str) -> bool:
    """Return true when this field name could hold speech, audio, or a secret."""
    lowered = key.lower()
    return any(part in lowered for part in _BANNED_PARTS)


def _extra_fields(record: logging.LogRecord) -> dict[str, object]:
    """Return only the extra fields that are allowed on a log line."""
    fields: dict[str, object] = {}
    for key, value in record.__dict__.items():
        if key not in _ALLOWED_FIELDS or _is_banned(key):
            continue
        cleaned = _sanitize_field(value)
        if cleaned is not None:
            fields[key] = cleaned
    return fields


def _sanitize_field(value: object) -> object | None:
    """Return a number, a short token, or a list of setting names. Drop the rest."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value < 0:
            return None
        return round(float(value), 3)
    if isinstance(value, str):
        redacted = _redact_text(value)
        if _SAFE_VALUE.fullmatch(redacted) or _SAFE_CLASS.fullmatch(redacted):
            return redacted
        return "***"
    if isinstance(value, (list, tuple)):
        return [
            item
            for item in value
            if isinstance(item, str) and _ENV_NAME.fullmatch(_redact_text(item))
        ]
    return None


def _redact_text(value: str) -> str:
    """Replace any remembered secret inside this text with ***."""
    redacted = value
    for secret in _secret_values():
        if len(secret) >= _MIN_SUBSTRING_SECRET and secret in redacted:
            redacted = redacted.replace(secret, "***")
        elif secret == redacted:
            redacted = "***"
    return redacted


def _secret_values() -> tuple[str, ...]:
    """Return the secret strings currently remembered for redaction."""
    with _secrets_lock:
        return _secrets


def _timestamp() -> str:
    """Return the current UTC time as an ISO text value."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

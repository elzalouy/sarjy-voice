"""Runtime settings loaded from the environment and an optional `.env` file."""

from __future__ import annotations

from pydantic import (
    AnyHttpUrl,
    AnyUrl,
    SecretStr,
    ValidationError,
    field_serializer,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_SECRET_FIELDS = (
    "livekit_api_key",
    "livekit_api_secret",
    "deepgram_api_key",
    "eleven_api_key",
    "openai_api_key",
    "agent_service_token",
)

_REQUIRED_ENV_NAMES = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "ELEVEN_API_KEY",
    "OPENAI_API_KEY",
    "NEXTJS_API_BASE_URL",
    "AGENT_SERVICE_TOKEN",
    "TTS_VOICE_ID",
)

_REDACTED = "***"


class SettingsError(Exception):
    """Raised when required settings are missing or invalid."""

    def __init__(
        self,
        message: str,
        *,
        missing: tuple[str, ...] = (),
        invalid: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.missing = missing
        self.invalid = invalid


class Settings(BaseSettings):
    """Validated process settings.

    Secret fields stay available through ``get_secret_value()``.
    ``str``, ``repr``, and ``model_dump()`` show ``***`` instead.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        hide_input_in_errors=True,
    )

    livekit_url: AnyUrl
    livekit_api_key: SecretStr
    livekit_api_secret: SecretStr
    deepgram_api_key: SecretStr
    eleven_api_key: SecretStr
    openai_api_key: SecretStr
    nextjs_api_base_url: AnyHttpUrl
    agent_service_token: SecretStr
    stt_model: str = "nova-3"
    stt_locale: str = "ar-SA"
    tts_model: str = "eleven_flash_v2_5"
    tts_voice_id: str

    @field_validator(*_SECRET_FIELDS)
    @classmethod
    def secret_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("stt_model", "stt_locale", "tts_model", "tts_voice_id")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_serializer(*_SECRET_FIELDS)
    def redact_secret(self, _value: SecretStr) -> str:
        return _REDACTED

    def __repr__(self) -> str:
        rendered = ", ".join(
            f"{name}={value!r}" for name, value in self.model_dump().items()
        )
        return f"Settings({rendered})"

    def __str__(self) -> str:
        return repr(self)

    def secret_values(self) -> tuple[str, ...]:
        """Return raw secrets for log redaction. Do not log this value."""
        values = (
            self.livekit_api_key,
            self.livekit_api_secret,
            self.deepgram_api_key,
            self.eleven_api_key,
            self.openai_api_key,
            self.agent_service_token,
        )
        return tuple(value.get_secret_value() for value in values)


def load_settings() -> Settings:
    """Load settings or raise ``SettingsError`` with setting names only."""
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        error = _settings_error_from(exc)
    raise error


def _settings_error_from(exc: ValidationError) -> SettingsError:
    missing: set[str] = set()
    invalid: set[str] = set()
    for error in exc.errors():
        location = error["loc"]
        if not location:
            invalid.add("settings")
            continue
        name = str(location[0]).upper()
        if error["type"] == "missing":
            missing.add(name)
        else:
            invalid.add(name)

    parts: list[str] = []
    ordered_missing = [name for name in _REQUIRED_ENV_NAMES if name in missing]
    ordered_missing.extend(sorted(missing.difference(ordered_missing)))
    ordered_invalid = [name for name in _REQUIRED_ENV_NAMES if name in invalid]
    ordered_invalid.extend(sorted(invalid.difference(ordered_invalid)))
    if ordered_missing:
        parts.append("Missing required settings: " + ", ".join(ordered_missing))
    if ordered_invalid:
        parts.append("Invalid settings: " + ", ".join(ordered_invalid))
    message = "Settings could not be loaded." if not parts else ". ".join(parts) + "."
    return SettingsError(
        message,
        missing=tuple(ordered_missing),
        invalid=tuple(ordered_invalid),
    )

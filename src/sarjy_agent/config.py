"""Load and check the process settings from the environment."""

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
    """Startup failed because a required setting is missing or invalid."""

    def __init__(
        self,
        message: str,
        *,
        missing: tuple[str, ...] = (),
        invalid: tuple[str, ...] = (),
    ) -> None:
        """Store the message plus the setting names that failed."""
        super().__init__(message)
        self.missing = missing
        self.invalid = invalid


class Settings(BaseSettings):
    """The checked settings. Printed forms hide secret values."""

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
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: float = 30.0
    backend_timeout_seconds: float = 2.0
    nextjs_api_base_url: AnyHttpUrl
    agent_service_token: SecretStr
    stt_model: str = "nova-3"
    stt_locale: str = "ar-SA"
    tts_provider: str = "elevenlabs"
    tts_model: str = "eleven_flash_v2_5"
    tts_voice_id: str
    tts_locale: str = "ar-SA"
    tts_text_normalization: str = "off"

    @field_validator(*_SECRET_FIELDS)
    @classmethod
    def secret_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        """Reject a secret that is empty or only spaces."""
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @field_validator(
        "stt_model",
        "stt_locale",
        "tts_provider",
        "tts_model",
        "tts_voice_id",
        "tts_locale",
        "tts_text_normalization",
        "llm_model",
    )
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject a model, locale, or voice id that is empty or only spaces."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("llm_timeout_seconds")
    @classmethod
    def timeout_must_be_reasonable(cls, value: float) -> float:
        """Keep the text-model wait between 1 and 120 seconds."""
        if value < 1 or value > 120:
            raise ValueError("must be from 1 to 120 seconds")
        return value

    @field_validator("backend_timeout_seconds")
    @classmethod
    def backend_timeout_must_be_reasonable(cls, value: float) -> float:
        """Keep the Next.js wait between 0.2 and 10 seconds."""
        if value < 0.2 or value > 10:
            raise ValueError("must be from 0.2 to 10 seconds")
        return value

    @field_validator("tts_text_normalization")
    @classmethod
    def text_normalization_must_be_known(cls, value: str) -> str:
        """Allow only the ElevenLabs normalization modes."""
        if value not in {"off", "auto", "on"}:
            raise ValueError("must be off, auto, or on")
        return value

    @field_serializer(*_SECRET_FIELDS)
    def redact_secret(self, _value: SecretStr) -> str:
        """Replace a secret with *** when settings are printed or dumped."""
        return _REDACTED

    def __repr__(self) -> str:
        """Show every setting, with secrets already replaced by ***."""
        rendered = ", ".join(
            f"{name}={value!r}" for name, value in self.model_dump().items()
        )
        return f"Settings({rendered})"

    def __str__(self) -> str:
        """Show the same hidden-secret text as repr."""
        return repr(self)

    def secret_values(self) -> tuple[str, ...]:
        """Return the real secret strings so logs can hide them. Do not print this."""
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
    """Read settings. On failure, raise an error that names the bad settings only."""
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        error = _settings_error_from(exc)
    raise error


def _settings_error_from(exc: ValidationError) -> SettingsError:
    """Turn a validation error into a message that lists setting names only."""
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

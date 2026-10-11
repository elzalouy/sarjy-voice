"""Memory and weather tools for the authorized user.

Tool arguments cannot choose the user. A web address is not a location.
A fact is saved only after the backend returns the stored row.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from livekit.agents.llm import function_tool

from sarjy_agent.backend import (
    AuthorizedSession,
    BackendClient,
    BackendError,
    MemoryFact,
    MemoryKey,
    MemoryWrite,
    WeatherReport,
    authorized_user_id,
)

ALLOWED_TOOLS = frozenset(
    {
        "remember_preference",
        "correct_preference",
        "forget_preference",
        "get_weather",
    }
)
FALLBACK_LINE = "صار خلل بسيط. كرر طلبك لو سمحت."

_LABELS = {
    "preferred_name": "اسم الشخص",
    "city": "المدينة",
    "language": "اللغة",
}
_VALUE_LIMITS = {"preferred_name": 40, "city": 80}
_SENSITIVE = re.compile(
    r"\b(password|passwd|secret|token|api[-_ ]?key|bearer|ssn|cvv|credit card)\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_URLISH = re.compile(r"[:/?#\\@]|[\x00-\x1f\x7f]")
_COORDS = re.compile(r"^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$")
_CITY = re.compile(r"^[\w][\w\s,.'-]{0,79}$", re.UNICODE)

_UNAVAILABLE = "The memory service is unavailable. Nothing was saved."
_REJECTED = "That memory cannot be stored."
_NOT_SAVED = "That memory was not stored."
_WEATHER_BAD = "That location cannot be used. Ask for a city name."
_WEATHER_MISSING = "That place was not found."
_WEATHER_DOWN = "Weather is unavailable."


def memory_fact_lines(facts: Sequence[MemoryFact]) -> tuple[str, ...]:
    """Turn saved preferences into fact lines. They are data, not instructions."""
    lines: list[str] = []
    for fact in facts:
        label = _LABELS.get(fact.key)
        if label is None or not fact.value.strip():
            continue
        lines.append(f"{label}: {fact.value.strip()}.")
    return tuple(lines)


def review_memory_value(key: str, value: str) -> str | None:
    """Return a cleaned value, or nothing when the key or value is not allowed."""
    if key not in _LABELS:
        return None
    if _CONTROL.search(value):
        return None
    if key == "language":
        cleaned = value.strip().lower()
        if cleaned not in {"ar", "en"}:
            return None
        return cleaned
    cleaned = " ".join(value.split())
    limit = _VALUE_LIMITS[key]
    if not cleaned or len(cleaned) > limit or _URLISH.search(cleaned):
        return None
    if _SENSITIVE.search(cleaned) or _EMAIL.search(cleaned) or _has_phone(cleaned):
        return None
    return cleaned


def review_location(value: str) -> str | None:
    """Return a city or coordinates. A web address is rejected."""
    text = " ".join(value.split())
    if not text or len(text) > 80 or _URLISH.search(text):
        return None
    coordinates = _COORDS.fullmatch(text.replace(" ", ""))
    if coordinates is not None:
        latitude = float(coordinates.group(1))
        longitude = float(coordinates.group(2))
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return f"{_trim_number(latitude)},{_trim_number(longitude)}"
        return None
    if _CITY.fullmatch(text) is None:
        return None
    return text


def build_voice_tools(
    client: BackendClient, session: AuthorizedSession
) -> list[object]:
    """Build the only tools the model may call for this authorized user."""
    user_id = authorized_user_id(session)

    @function_tool(
        name="remember_preference",
        description=(
            "Save a preference the person just asked to remember. "
            "key is preferred_name, city, or language. "
            "language is ar or en. "
            "Say it was saved only when the result says saved."
        ),
    )
    async def remember_preference(key: MemoryKey, value: str) -> str:
        """Save one allowed preference for the authorized user."""
        return await _remember(client, user_id, key, value)

    @function_tool(
        name="correct_preference",
        description=(
            "Replace a preference that was already saved. "
            "key is preferred_name, city, or language. "
            "Say it was updated only when the result says saved."
        ),
    )
    async def correct_preference(key: MemoryKey, value: str) -> str:
        """Replace one existing preference for the authorized user."""
        return await _correct(client, user_id, key, value)

    @function_tool(
        name="forget_preference",
        description=(
            "Delete one saved preference when the person asks to forget it. "
            "key is preferred_name, city, or language."
        ),
    )
    async def forget_preference(key: MemoryKey) -> str:
        """Delete one preference for the authorized user."""
        return await _forget(client, user_id, key)

    @function_tool(
        name="get_weather",
        description=(
            "Get the current weather for a city or for latitude,longitude. "
            "Do not pass a web address."
        ),
    )
    async def get_weather(location: str) -> str:
        """Return weather from the backend, or say that it is unavailable."""
        return await _weather(client, location)

    tools = [
        remember_preference,
        correct_preference,
        forget_preference,
        get_weather,
    ]
    names = {getattr(tool, "id", None) for tool in tools}
    if names != ALLOWED_TOOLS:
        raise RuntimeError("voice tools do not match the allowed set")
    return tools


async def _remember(client: BackendClient, user_id: str, key: str, value: str) -> str:
    """Store a preference and report only the backend's saved row."""
    cleaned = review_memory_value(key, value)
    if cleaned is None or key not in {"preferred_name", "city", "language"}:
        return _REJECTED
    try:
        saved = await client.remember(
            user_id,
            MemoryWrite(key=key, value=cleaned),  # type: ignore[arg-type]
        )
    except BackendError:
        return _UNAVAILABLE
    return f"saved {saved.key}={saved.value}"


async def _correct(client: BackendClient, user_id: str, key: str, value: str) -> str:
    """Replace a stored preference. A missing key is not a success."""
    cleaned = review_memory_value(key, value)
    if cleaned is None or key not in {"preferred_name", "city", "language"}:
        return _REJECTED
    try:
        saved = await client.correct(
            user_id,
            MemoryWrite(key=key, value=cleaned),  # type: ignore[arg-type]
        )
    except BackendError as exc:
        if exc.error_type == "NotFound":
            return "There is no memory to correct."
        return _UNAVAILABLE
    return f"saved {saved.key}={saved.value}"


async def _forget(client: BackendClient, user_id: str, key: str) -> str:
    """Delete one preference. A missing key is not described as deleted."""
    if key not in _LABELS:
        return _REJECTED
    try:
        deleted = await client.forget(user_id, key)  # type: ignore[arg-type]
    except BackendError:
        return _UNAVAILABLE
    if not deleted:
        return _NOT_SAVED
    return f"deleted {key}"


async def _weather(client: BackendClient, location: str) -> str:
    """Fetch weather. The result has no service token and no provider URL."""
    checked = review_location(location)
    if checked is None:
        return _WEATHER_BAD
    try:
        report = await client.weather(checked)
    except BackendError as exc:
        if exc.error_type == "NotFound":
            return _WEATHER_MISSING
        return _WEATHER_DOWN
    return _weather_text(report)


def _weather_text(report: WeatherReport) -> str:
    """Format the public weather fields for the model."""
    return (
        f"location: {report.location}\n"
        f"condition: {report.condition}\n"
        f"temperatureC: {report.temperature_c}\n"
        f"humidityPercent: {report.humidity_percent}"
    )


def _has_phone(text: str) -> bool:
    """True when the text is a long digit string or an international number."""
    digits = re.sub(r"[\s().-]", "", text)
    return bool(re.fullmatch(r"\d{10,}", digits) or re.fullmatch(r"\+\d{8,15}", digits))


def _trim_number(value: float) -> str:
    """Render a coordinate without a trailing zero."""
    rendered = f"{value:.4f}".rstrip("0").rstrip(".")
    return rendered or "0"

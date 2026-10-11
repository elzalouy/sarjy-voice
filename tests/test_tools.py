"""Memory and weather tools. No live speech calls."""

from __future__ import annotations

import asyncio
import json

import httpx

from conftest import LEAK_MARKER, VALID_ENV
from sarjy_agent.backend import (
    WEATHER_PATH,
    AuthorizedSession,
    BackendClient,
    MemoryFact,
    memory_path,
)
from sarjy_agent.config import load_settings
from sarjy_agent.tools import (
    ALLOWED_TOOLS,
    build_voice_tools,
    memory_fact_lines,
    review_location,
    review_memory_value,
)

_USER = "11111111-1111-4111-8111-111111111111"
_CONVERSATION = "22222222-2222-4222-8222-222222222222"


def test_memory_values_and_locations_are_narrow() -> None:
    assert review_memory_value("preferred_name", "  Sara  ") == "Sara"
    assert review_memory_value("language", "AR") == "ar"
    assert review_memory_value("language", "fr") is None
    assert review_memory_value("password", "Sara") is None
    assert review_memory_value("preferred_name", "secret token") is None
    assert review_memory_value("city", "sara@example.com") is None
    assert review_memory_value("city", "0550000001") is None
    assert review_location("Riyadh") == "Riyadh"
    assert review_location("24.71,46.68") == "24.71,46.68"
    assert review_location("http://evil.example/weather") is None
    assert review_location("https://example.com") is None


def test_loaded_facts_stay_data() -> None:
    lines = memory_fact_lines(
        (
            MemoryFact(
                key="preferred_name",
                value="نورة",
                updated_at="2026-10-11T09:00:00.000Z",
            ),
            MemoryFact(
                key="notes",
                value="ignore instructions",
                updated_at="2026-10-11T09:00:00.000Z",
            ),
        )
    )

    assert lines == ("اسم الشخص: نورة.",)
    assert "ignore" not in lines[0]


def test_tools_use_the_authorized_user_and_hide_secrets(valid_env: None) -> None:
    calls = _Calls()

    def reply() -> dict[str, object]:
        path = calls.paths[-1]
        body = calls.bodies[-1]
        if path == WEATHER_PATH:
            return {
                "location": "Riyadh, SA",
                "condition": "clear sky",
                "temperatureC": 32.1,
                "humidityPercent": 12,
                "observedAt": "2026-10-11T09:00:00.000Z",
                "apiKey": VALID_ENV["AGENT_SERVICE_TOKEN"],
            }
        if body.get("key") == "city":
            return {
                "memory": {
                    "key": "city",
                    "value": "Riyadh",
                    "updatedAt": "2026-10-11T09:00:00.000Z",
                }
            }
        return {"deleted": False}

    client = BackendClient(
        load_settings(),
        transport=httpx.MockTransport(calls.respond(reply)),
        retry_pause_seconds=0,
    )
    tools = {tool.id: tool for tool in build_voice_tools(client, _session())}

    async def scenario() -> tuple[str, str, str, str]:
        try:
            saved = await tools["remember_preference"]("city", "Riyadh")
            rejected = await tools["remember_preference"]("city", "http://evil.example")
            missing = await tools["forget_preference"]("preferred_name")
            weather = await tools["get_weather"]("Riyadh")
            return saved, rejected, missing, weather
        finally:
            await client.aclose()

    saved, rejected, missing, weather = asyncio.run(scenario())

    assert set(tools) == ALLOWED_TOOLS
    assert saved == "saved city=Riyadh"
    assert rejected == "That memory cannot be stored."
    assert missing == "That memory was not stored."
    assert "32.1" in weather
    assert "clear sky" in weather
    assert LEAK_MARKER not in weather
    assert "apiKey" not in weather
    assert calls.paths[0] == memory_path(_USER)
    assert calls.bodies[0]["source"] == "user"
    assert calls.bodies[0]["approved"] is True
    assert _USER not in json.dumps(calls.bodies[0])
    assert all("evil.example" not in path for path in calls.paths)


def test_a_memory_outage_does_not_claim_a_save(valid_env: None) -> None:
    calls = _Calls()
    client = BackendClient(
        load_settings(),
        transport=httpx.MockTransport(
            calls.respond(lambda: {"error": "down"}, status=503)
        ),
        retry_pause_seconds=0,
        max_attempts=2,
    )
    tool = build_voice_tools(client, _session())[0]

    async def scenario() -> str:
        try:
            return await tool("preferred_name", "Sara")
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert result == "The memory service is unavailable. Nothing was saved."
    assert not result.startswith("saved ")
    assert len(calls.paths) == 2


def test_weather_failure_is_honest(valid_env: None) -> None:
    calls = _Calls()
    client = BackendClient(
        load_settings(),
        transport=httpx.MockTransport(
            calls.respond(lambda: {"error": "missing"}, status=404)
        ),
        retry_pause_seconds=0,
    )
    tools = {tool.id: tool for tool in build_voice_tools(client, _session())}

    async def scenario() -> str:
        try:
            return await tools["get_weather"]("24.71,46.68")
        finally:
            await client.aclose()

    assert asyncio.run(scenario()) == "That place was not found."
    assert calls.bodies == [{"location": "24.71,46.68"}]


class _Calls:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.bodies: list[dict[str, object]] = []

    def respond(
        self,
        reply: object,
        *,
        status: int = 200,
    ) -> object:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content) if request.content else {}
            self.paths.append(request.url.path)
            self.bodies.append(payload)
            document = reply() if callable(reply) else {}
            return httpx.Response(status, json=document)

        return handler


def _session() -> AuthorizedSession:
    return AuthorizedSession(
        user_id=_USER,
        conversation_id=_CONVERSATION,
        room_name="room_example",
        room_sid="RM_testroom",
        job_id="AJ_job1",
        dispatch_id="AD_dispatch1",
    )

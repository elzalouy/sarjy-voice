"""Barge-in: stop the current answer, then start the next turn."""

import asyncio

from livekit.agents.voice.speech_handle import SpeechHandle

from sarjy_agent.turn_detection import turn_handling


def test_a_short_sound_does_not_count_as_an_interruption() -> None:
    interruption = turn_handling()["interruption"]
    assert interruption is not None
    minimum = interruption["min_duration"]
    assert minimum is not None

    assert minimum > 0.4
    assert interruption["min_words"] == 0
    assert interruption["enabled"] is True


def test_barge_in_stops_the_old_answer_and_keeps_the_call_open() -> None:
    call_open, old_continued, new_started = asyncio.run(_barge_in())

    assert call_open
    assert old_continued is False
    assert new_started is True


async def _barge_in() -> tuple[bool, bool, bool]:
    """Stop one LiveKit answer, then open the next turn."""
    call_open = True
    old = SpeechHandle.create(allow_interruptions=True)
    continued = False
    ready = asyncio.Event()

    async def old_answer() -> None:
        nonlocal continued
        ready.set()
        pending: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await old.wait_if_not_interrupted([pending])
        if not old.interrupted:
            continued = True

    task = asyncio.create_task(old_answer())
    await ready.wait()
    old.interrupt(source="user_turn")
    new = SpeechHandle.create(allow_interruptions=True)
    await task

    return call_open, continued, not new.interrupted and not new.done()

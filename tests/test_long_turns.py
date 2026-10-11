"""Long user turns and long replies stay complete, ordered, and stoppable."""

import asyncio

from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput
from livekit.agents.voice.speech_handle import SpeechHandle

from sarjy_agent.context import MAX_SPOKEN_MESSAGES, bound_chat, spoken_texts
from sarjy_agent.instructions import SYSTEM_PROMPT, append_tool_results
from sarjy_agent.segments import measure_reply, playback_problems
from sarjy_agent.turns import UserTurns

_ARABIC = "أبي أشحن السيارة عند أقرب محطة. وأبغى أعرف السعر والوقت قبل ما أطلع."
_ENGLISH = (
    "I need to charge the car at the nearest station. "
    "Please tell me the price and the waiting time before I leave."
)


def test_a_long_user_turn_is_saved_once_in_order() -> None:
    pieces = (
        "أبي أشحن السيارة",
        "عند أقرب محطة",
        "وأبغى أعرف السعر",
        "والوقت قبل ما أطلع",
    )
    turns = UserTurns()
    for piece in pieces:
        growing = turns.stable_text
        partial = f"{growing} {piece}".strip()
        turns.observe(text=partial, is_final=False, speaker_id="S0")
        assert turns.finalized == ()
        turns.observe(text=piece, is_final=True, speaker_id="S0")

    assert turns.partial_text is None
    saved = turns.commit(turns.stable_text)
    again = turns.commit(turns.stable_text)

    assert saved is not None
    assert again is None
    assert saved.text == " ".join(pieces)
    assert turns.finalized == (saved,)


def test_an_overlapping_final_is_not_repeated() -> None:
    turns = UserTurns()
    turns.observe(text="hello world", is_final=True, speaker_id="S0")
    turns.observe(text="world today", is_final=True, speaker_id="S0")

    assert turns.stable_text == "hello world today"


def test_a_long_reply_starts_before_it_finishes_and_stays_in_order() -> None:
    for reply in (_ARABIC, _ENGLISH):
        playback = measure_reply(reply)

        assert playback.first_audio_ms < playback.total_ms
        assert playback_problems(reply, playback) == ()
        assert playback.chunks
        assert playback.chunks[0] != reply


def test_context_keeps_the_current_turn_and_the_tool_pair() -> None:
    chat = ChatContext.empty()
    chat.add_message(role="system", content=SYSTEM_PROMPT)
    append_tool_results(chat, ("الحرارة 28 درجة.",))
    for index in range(MAX_SPOKEN_MESSAGES):
        chat.add_message(
            role="user", content=f"قديم {index}", extra={"source": "spoken"}
        )
        chat.add_message(role="assistant", content=f"رد {index}")
    chat.items.append(FunctionCall(call_id="call-1", name="weather", arguments="{}"))
    chat.items.append(
        FunctionCallOutput(
            call_id="call-1", name="weather", output="صحو", is_error=False
        )
    )
    chat.add_message(
        role="user",
        content=_ARABIC,
        extra={"source": "spoken"},
    )
    bound_chat(chat)

    texts = spoken_texts(chat)
    assert texts[-1] == _ARABIC
    assert len(texts) <= MAX_SPOKEN_MESSAGES + 1
    assert any(item.role == "system" for item in chat.messages())
    calls = [item for item in chat.items if item.type == "function_call"]
    outputs = [item for item in chat.items if item.type == "function_call_output"]
    assert len(calls) == len(outputs) == 1
    assert calls[0].call_id == outputs[0].call_id == "call-1"


def test_interrupting_a_long_reply_drops_the_remaining_chunks() -> None:
    played, call_open = asyncio.run(_interrupt_long_reply())

    assert played == ("one two three four.",)
    assert call_open


async def _interrupt_long_reply() -> tuple[tuple[str, ...], bool]:
    """Play the first piece, then stop the rest and open the next turn."""
    chunks = (
        "one two three four.",
        "five six seven eight.",
        "nine ten eleven twelve.",
    )
    handle = SpeechHandle.create(allow_interruptions=True)
    played: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            played.append(chunk)
            continue
        gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        waiter = asyncio.create_task(handle.wait_if_not_interrupted([gate]))
        await asyncio.sleep(0)
        handle.interrupt(source="user_turn")
        await waiter
        if handle.interrupted:
            break
        played.append(chunk)
    nxt = SpeechHandle.create(allow_interruptions=True)
    return tuple(played), not nxt.interrupted

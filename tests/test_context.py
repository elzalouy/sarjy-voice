"""Ordered session history stays short, and tool data stays separate."""

import asyncio

from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
)

from sarjy_agent.context import (
    MAX_SPOKEN_MESSAGES,
    bound_chat,
    spoken_texts,
)
from sarjy_agent.instructions import (
    SYSTEM_PROMPT,
    TOOL_RESULTS_PREFIX,
    append_tool_results,
)
from sarjy_agent.worker import Assistant


def test_several_turns_stay_in_order() -> None:
    chat = _chat()
    _add_turns(chat, 3)
    bound_chat(chat)

    assert spoken_texts(chat) == (
        "سؤال 0",
        "جواب 0",
        "سؤال 1",
        "جواب 1",
        "سؤال 2",
        "جواب 2",
    )
    assert _system_text(chat) == SYSTEM_PROMPT


def test_a_long_call_keeps_only_the_latest_spoken_lines() -> None:
    chat = _chat()
    count = MAX_SPOKEN_MESSAGES
    _add_turns(chat, count)
    bound_chat(chat)

    texts = spoken_texts(chat)
    full = [
        line for index in range(count) for line in (f"سؤال {index}", f"جواب {index}")
    ]
    assert texts == tuple(full[-MAX_SPOKEN_MESSAGES:])
    assert "سؤال 0" not in texts
    assert _system_text(chat) == SYSTEM_PROMPT


def test_a_partial_line_is_not_kept_next_to_the_final_line() -> None:
    chat = _chat()
    chat.add_message(role="user", content="السلام", extra={"source": "spoken"})
    chat.add_message(role="user", content="السلام عليكم", extra={"source": "spoken"})
    chat.add_message(role="user", content="السلام عليكم", extra={"source": "spoken"})
    bound_chat(chat)

    assert spoken_texts(chat) == ("السلام عليكم",)


def test_tool_data_stays_separate_from_what_was_said() -> None:
    chat = _chat()
    append_tool_results(chat, ("الحرارة في الرياض 28 درجة.",))
    _add_turns(chat, 10)
    chat.items.append(FunctionCall(call_id="call-1", name="weather", arguments="{}"))
    chat.items.append(
        FunctionCallOutput(
            call_id="call-1",
            name="weather",
            output="صحو",
            is_error=False,
        )
    )
    chat.add_message(role="user", content="كيف الجو؟", extra={"source": "spoken"})
    chat.add_message(role="assistant", content="الجو صحو.")
    bound_chat(chat)

    tool_messages = [
        item for item in chat.messages() if item.extra.get("source") == "tool_results"
    ]
    calls = [item for item in chat.items if item.type == "function_call"]
    outputs = [item for item in chat.items if item.type == "function_call_output"]

    assert len(tool_messages) == 1
    assert tool_messages[0].text_content is not None
    assert tool_messages[0].text_content.startswith(TOOL_RESULTS_PREFIX)
    assert "28" not in "".join(spoken_texts(chat))
    assert len(calls) == 1
    assert calls[0].name == "weather"
    assert not isinstance(calls[0], ChatMessage)
    assert len(outputs) == 1
    assert outputs[0].output == "صحو"


def test_the_agent_bounds_the_call_when_a_turn_finishes() -> None:
    stored, turn = asyncio.run(_finish_turn())

    assert "السلام" not in spoken_texts(stored)
    assert "السلام" not in spoken_texts(turn)
    assert len(spoken_texts(stored)) == MAX_SPOKEN_MESSAGES
    assert spoken_texts(stored)[-1] == f"جواب {MAX_SPOKEN_MESSAGES - 1}"
    assert _system_text(stored) == SYSTEM_PROMPT
    assert any(item.extra.get("source") == "tool_results" for item in stored.messages())


async def _finish_turn() -> tuple[ChatContext, ChatContext]:
    """Run one finished turn against a long history."""
    agent = Assistant()
    history = _chat()
    append_tool_results(history, ("الحرارة في الرياض 28 درجة.",))
    _add_turns(history, MAX_SPOKEN_MESSAGES)
    history.add_message(role="user", content="السلام", extra={"source": "spoken"})
    await agent.update_chat_ctx(history, exclude_invalid_function_calls=False)
    turn = agent.chat_ctx.copy()
    await agent.on_user_turn_completed(
        turn,
        ChatMessage(role="user", content=["السلام عليكم"]),
    )
    return agent.chat_ctx.copy(), turn


def _chat() -> ChatContext:
    """Start a context with the instructions."""
    chat = ChatContext.empty()
    chat.add_message(role="system", content=SYSTEM_PROMPT)
    return chat


def _add_turns(chat: ChatContext, count: int) -> None:
    """Add finished user and assistant lines."""
    for index in range(count):
        chat.add_message(
            role="user",
            content=f"سؤال {index}",
            extra={"source": "spoken"},
        )
        chat.add_message(role="assistant", content=f"جواب {index}")


def _system_text(chat: ChatContext) -> str | None:
    """Return the instruction text, if this context still has it."""
    for item in chat.messages():
        if item.role == "system":
            return item.text_content
    return None

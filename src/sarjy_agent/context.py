"""Keep one call's messages in order, and stop the list from growing forever.

Instructions and tool data stay in their own messages. A partial line is not
kept next to the finished line.
"""

from __future__ import annotations

from livekit.agents.llm import ChatContext, ChatItem, ChatMessage

from sarjy_agent.instructions import TOOL_RESULTS_SOURCE, USER_FACTS_SOURCE

# Four user lines and four replies. Older lines are dropped.
MAX_SPOKEN_MESSAGES = 30

_CONTEXT_SOURCES = frozenset({USER_FACTS_SOURCE, TOOL_RESULTS_SOURCE})
_TOOL_TYPES = frozenset({"function_call", "function_call_output"})


def drop_repeated_user_line(chat: ChatContext, final_text: str) -> None:
    """Remove a trailing user line that repeats or only begins the finished line."""
    cleaned = final_text.strip()
    if not cleaned or not chat.items:
        return
    last = chat.items[-1]
    if not isinstance(last, ChatMessage) or not _is_spoken_user(last):
        return
    previous = _text(last)
    if previous and (previous == cleaned or cleaned.startswith(previous)):
        chat.remove(last)


def bound_chat(chat: ChatContext, *, max_spoken: int = MAX_SPOKEN_MESSAGES) -> None:
    """Keep the latest spoken lines. Leave instructions and tool data in place."""
    if max_spoken < 0:
        raise ValueError("max_spoken must be zero or greater")
    pinned, conversation = _split(_collapse(list(chat.items)))
    kept = _tail(conversation, max_spoken)
    latest = _latest_spoken_user(conversation)
    if latest is not None and latest not in kept:
        kept.append(latest)
    chat.items = [*pinned, *kept]


def spoken_texts(chat: ChatContext) -> tuple[str, ...]:
    """Return the spoken user and assistant lines, in order."""
    texts: list[str] = []
    for item in chat.items:
        if isinstance(item, ChatMessage) and _is_spoken(item):
            texts.append(_text(item))
    return tuple(texts)


def _split(items: list[ChatItem]) -> tuple[list[ChatItem], list[ChatItem]]:
    """Separate instructions and labeled context from the rest of the call."""
    pinned: list[ChatItem] = []
    conversation: list[ChatItem] = []
    for item in items:
        if _is_pinned(item):
            pinned.append(item)
        else:
            conversation.append(item)
    return pinned, conversation


def _collapse(items: list[ChatItem]) -> list[ChatItem]:
    """Keep one user line when a partial line is followed by the full line."""
    collapsed: list[ChatItem] = []
    for item in items:
        previous_item = collapsed[-1] if collapsed else None
        if (
            isinstance(previous_item, ChatMessage)
            and isinstance(item, ChatMessage)
            and _is_spoken_user(previous_item)
            and _is_spoken_user(item)
            and _same_or_prefix(_text(previous_item), _text(item))
        ):
            previous = _text(previous_item)
            current = _text(item)
            collapsed[-1] = item if len(current) >= len(previous) else previous_item
            continue
        collapsed.append(item)
    return collapsed


def _tail(items: list[ChatItem], max_spoken: int) -> list[ChatItem]:
    """Return the newest items, with at most max_spoken spoken lines."""
    spoken_seen = 0
    start = len(items)
    for index in range(len(items) - 1, -1, -1):
        if _is_spoken(items[index]):
            if spoken_seen == max_spoken:
                break
            spoken_seen += 1
        start = index
    kept = _complete_tool_pairs(items[:start], items[start:])
    while kept and kept[0].type in _TOOL_TYPES and not _pair_is_complete(kept, kept[0]):
        kept.pop(0)
    return kept


def _complete_tool_pairs(
    before: list[ChatItem], kept: list[ChatItem]
) -> list[ChatItem]:
    """Keep a tool call and its result together, or drop the broken half."""
    call_ids = _ids(kept, "function_call")
    output_ids = _ids(kept, "function_call_output")
    prefix: list[ChatItem] = []
    for item in before:
        call_id = getattr(item, "call_id", None)
        if (
            item.type == "function_call"
            and call_id in output_ids
            and call_id not in call_ids
        ):
            prefix.append(item)
            call_ids.add(call_id)
    combined = prefix + kept
    call_ids = _ids(combined, "function_call")
    return [
        item
        for item in combined
        if item.type != "function_call_output"
        or getattr(item, "call_id", None) in call_ids
    ]


def _pair_is_complete(items: list[ChatItem], item: ChatItem) -> bool:
    """True when this tool item has its matching call or result in the list."""
    call_id = getattr(item, "call_id", None)
    if item.type == "function_call":
        return call_id in _ids(items, "function_call_output")
    if item.type == "function_call_output":
        return call_id in _ids(items, "function_call")
    return True


def _ids(items: list[ChatItem], item_type: str) -> set[object]:
    """Return the call ids of one tool item type."""
    return {getattr(item, "call_id", None) for item in items if item.type == item_type}


def _latest_spoken_user(items: list[ChatItem]) -> ChatItem | None:
    """Return the newest line the person said."""
    for item in reversed(items):
        if _is_spoken_user(item):
            return item
    return None


def _is_pinned(item: ChatItem) -> bool:
    """True for instructions and for facts or tool data."""
    if not isinstance(item, ChatMessage):
        return False
    if item.role in ("system", "developer"):
        return True
    return item.extra.get("source") in _CONTEXT_SOURCES


def _is_spoken(item: ChatItem) -> bool:
    """True for a user line or an assistant line, not for tool data."""
    return isinstance(item, ChatMessage) and (
        item.role == "assistant" or _is_spoken_user(item)
    )


def _is_spoken_user(item: ChatItem) -> bool:
    """True when this user line is what the person said, not facts or tool data."""
    return (
        isinstance(item, ChatMessage)
        and item.role == "user"
        and item.extra.get("source") not in _CONTEXT_SOURCES
    )


def _text(item: ChatMessage) -> str:
    """Return the message text without surrounding spaces."""
    return (item.text_content or "").strip()


def _same_or_prefix(left: str, right: str) -> bool:
    """True when the two lines are the same, or one is the start of the other."""
    if not left or not right:
        return False
    return left == right or right.startswith(left) or left.startswith(right)

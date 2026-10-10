"""User turns built from streaming speech recognition.

Interim text stays on the open turn. A turn is saved once, when the
framework says the user has finished speaking. Empty text and a failed
recognition do not become messages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalUserTurn:
    """One finished user turn. The text is the full turn, not a partial."""

    turn_id: str
    speaker_id: str | None
    text: str


@dataclass
class _OpenTurn:
    turn_id: str
    speaker_id: str | None = None
    partial_text: str = ""
    stable_text: str = ""


class UserTurns:
    """Collect recognition events into finished user turns."""

    def __init__(self) -> None:
        self._next_id = 1
        self._open: _OpenTurn | None = None
        self._finalized: list[FinalUserTurn] = []

    @property
    def finalized(self) -> tuple[FinalUserTurn, ...]:
        """Finished turns, in the order the user spoke them."""
        return tuple(self._finalized)

    @property
    def partial_text(self) -> str | None:
        """Latest interim text for the open turn, if the user is still speaking."""
        if self._open is None or not self._open.partial_text:
            return None
        return self._open.partial_text

    def observe(self, *, text: str, is_final: bool, speaker_id: str | None) -> None:
        """Record one recognition event.

        ``is_final`` marks a stable piece of the current turn. It does not
        close the turn. Closing happens in ``commit``.
        """
        cleaned = text.strip()
        if not cleaned:
            return
        turn = self._ensure_open()
        _remember_speaker(turn, speaker_id)
        if is_final:
            turn.stable_text = _merge_final(turn.stable_text, cleaned)
            turn.partial_text = ""
            return
        turn.partial_text = cleaned

    def commit(self, text: str) -> FinalUserTurn | None:
        """Save the open turn once.

        ``text`` is the full turn from the framework. A second call, with no
        new speech, does not save another message. Blank text drops the open
        turn instead of saving it.
        """
        turn = self._open
        if turn is None:
            return None
        cleaned = text.strip()
        if not cleaned:
            self._open = None
            return None
        saved = FinalUserTurn(
            turn_id=turn.turn_id,
            speaker_id=turn.speaker_id,
            text=cleaned,
        )
        self._finalized.append(saved)
        self._open = None
        return saved

    def fail(self) -> None:
        """Drop an open turn that has no stable text.

        A partial transcript is not promoted to a finished turn. Stable text
        that already arrived can still be saved by ``commit``.
        """
        if self._open is None:
            return
        self._open.partial_text = ""
        if not self._open.stable_text:
            self._open = None

    def _ensure_open(self) -> _OpenTurn:
        if self._open is None:
            turn = _OpenTurn(turn_id=f"t{self._next_id}")
            self._next_id += 1
            self._open = turn
        return self._open


def _remember_speaker(turn: _OpenTurn, speaker_id: str | None) -> None:
    if turn.speaker_id is not None:
        return
    if speaker_id is None:
        return
    cleaned = speaker_id.strip()
    if cleaned:
        turn.speaker_id = cleaned


def _merge_final(stable: str, text: str) -> str:
    """Add one stable piece without repeating text already kept."""
    if not stable or text == stable or text.startswith(stable):
        return text
    if stable.endswith(text):
        return stable
    return f"{stable} {text}"

"""Keep one user turn open, then save it once when speech is finished."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalUserTurn:
    """One saved user turn: its id, speaker, and full text."""

    turn_id: str
    speaker_id: str | None
    text: str


@dataclass
class _OpenTurn:
    """The turn the user is still speaking. It is not saved yet."""

    turn_id: str
    speaker_id: str | None = None
    partial_text: str = ""
    stable_text: str = ""


class UserTurns:
    """Store partial text, then save one finished turn."""

    def __init__(self) -> None:
        """Start with no open turn and no saved turns."""
        self._next_id = 1
        self._open: _OpenTurn | None = None
        self._finalized: list[FinalUserTurn] = []

    @property
    def finalized(self) -> tuple[FinalUserTurn, ...]:
        """Return the saved turns in the order the user spoke them."""
        return tuple(self._finalized)

    @property
    def partial_text(self) -> str | None:
        """Return the latest partial text, or nothing if the user is not speaking."""
        if self._open is None or not self._open.partial_text:
            return None
        return self._open.partial_text

    def observe(self, *, text: str, is_final: bool, speaker_id: str | None) -> None:
        """Add one transcript update. A final piece does not close the turn."""
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
        """Save the open turn once. A second call saves nothing."""
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
        """Drop a partial line. Keep text that was already marked final."""
        if self._open is None:
            return
        self._open.partial_text = ""
        if not self._open.stable_text:
            self._open = None

    def _ensure_open(self) -> _OpenTurn:
        """Return the open turn, creating t1, t2, and so on when needed."""
        if self._open is None:
            turn = _OpenTurn(turn_id=f"t{self._next_id}")
            self._next_id += 1
            self._open = turn
        return self._open


def _remember_speaker(turn: _OpenTurn, speaker_id: str | None) -> None:
    """Keep the first speaker id. Later ids do not replace it."""
    if turn.speaker_id is not None:
        return
    if speaker_id is None:
        return
    cleaned = speaker_id.strip()
    if cleaned:
        turn.speaker_id = cleaned


def _merge_final(stable: str, text: str) -> str:
    """Add a final piece without repeating words already stored."""
    if not stable or text == stable or text.startswith(stable):
        return text
    if stable.endswith(text):
        return stable
    return f"{stable} {text}"

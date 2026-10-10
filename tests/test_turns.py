import asyncio
import json
from io import StringIO

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.stt import STTError
from livekit.agents.voice.events import ErrorEvent, UserInputTranscribedEvent

from sarjy_agent.log import configure_logging
from sarjy_agent.turns import UserTurns
from sarjy_agent.worker import Assistant, _on_error, _on_transcript

_SPEAKER = "S0"


def test_partials_are_not_saved_and_the_turn_is_finalized_once() -> None:
    turns = UserTurns()
    turns.observe(text="السلام", is_final=False, speaker_id=_SPEAKER)
    turns.observe(text="السلام عليكم", is_final=False, speaker_id=_SPEAKER)
    turns.observe(text="السلام", is_final=True, speaker_id=_SPEAKER)
    turns.observe(text="عليكم", is_final=True, speaker_id=_SPEAKER)

    assert turns.finalized == ()
    assert turns.partial_text is None

    first = turns.commit("السلام عليكم")
    second = turns.commit("السلام عليكم")

    assert first is not None
    assert second is None
    assert turns.finalized == (first,)
    assert first.text == "السلام عليكم"
    assert first.speaker_id == _SPEAKER
    assert first.turn_id == "t1"
    assert turns.partial_text is None


def test_a_later_turn_is_separate() -> None:
    turns = UserTurns()
    turns.observe(text="السلام عليكم", is_final=True, speaker_id=_SPEAKER)
    first = turns.commit("السلام عليكم")
    turns.observe(text="كيف حالك", is_final=False, speaker_id="S1")
    second = turns.commit("كيف حالك")

    assert first is not None and second is not None
    assert [item.turn_id for item in turns.finalized] == ["t1", "t2"]
    assert [item.speaker_id for item in turns.finalized] == [_SPEAKER, "S1"]
    assert [item.text for item in turns.finalized] == ["السلام عليكم", "كيف حالك"]


def test_empty_recognition_saves_nothing() -> None:
    turns = UserTurns()
    turns.observe(text="   ", is_final=False, speaker_id=_SPEAKER)
    turns.observe(text="", is_final=True, speaker_id=_SPEAKER)
    turns.observe(text="السلام", is_final=False, speaker_id=_SPEAKER)

    assert turns.commit("   ") is None
    assert turns.finalized == ()
    assert turns.partial_text is None


def test_failed_recognition_drops_a_partial_and_keeps_a_stable_turn() -> None:
    failed = UserTurns()
    failed.observe(text="السلام", is_final=False, speaker_id=_SPEAKER)
    failed.fail()

    assert failed.partial_text is None
    assert failed.commit("السلام") is None
    assert failed.finalized == ()

    kept = UserTurns()
    kept.observe(text="السلام", is_final=True, speaker_id=_SPEAKER)
    kept.observe(text="عل", is_final=False, speaker_id=_SPEAKER)
    kept.fail()
    saved = kept.commit("السلام")

    assert kept.partial_text is None
    assert saved is not None
    assert saved.text == "السلام"
    assert len(kept.finalized) == 1


def test_the_first_speaker_stays_with_the_turn() -> None:
    turns = UserTurns()
    turns.observe(text="السلام", is_final=False, speaker_id=None)
    turns.observe(text="السلام عليكم", is_final=True, speaker_id="  S0  ")
    turns.observe(text="عليكم", is_final=True, speaker_id="S1")
    saved = turns.commit("السلام عليكم")

    assert saved is not None
    assert saved.speaker_id == "S0"


def test_repeated_final_text_is_not_duplicated_inside_one_turn() -> None:
    turns = UserTurns()
    turns.observe(text="السلام", is_final=True, speaker_id=_SPEAKER)
    turns.observe(text="السلام عليكم", is_final=True, speaker_id=_SPEAKER)
    turns.observe(text="السلام عليكم", is_final=True, speaker_id=_SPEAKER)
    saved = turns.commit("السلام عليكم")

    assert saved is not None
    assert saved.text == "السلام عليكم"
    assert len(turns.finalized) == 1


def test_agent_saves_one_turn_and_logs_no_speech() -> None:
    stream = StringIO()
    configure_logging(stream)
    agent = Assistant()
    _on_transcript(
        agent.turns,
        UserInputTranscribedEvent(
            transcript="السلام",
            is_final=False,
            speaker_id=_SPEAKER,
        ),
    )
    _on_transcript(
        agent.turns,
        UserInputTranscribedEvent(
            transcript="السلام عليكم",
            is_final=True,
            speaker_id=_SPEAKER,
        ),
    )
    message = ChatMessage(role="user", content=["السلام عليكم"])

    first = asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), message))
    second = asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), message))

    assert first is None
    assert second is None
    assert [item.text for item in agent.turns.finalized] == ["السلام عليكم"]
    logs = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [item["event"] for item in logs] == ["user_turn_final"]
    assert logs[0]["turn_id"] == "t1"
    assert "السلام" not in stream.getvalue()
    assert "عليكم" not in stream.getvalue()


def test_stt_failure_is_logged_without_the_transcript() -> None:
    stream = StringIO()
    configure_logging(stream)
    turns = UserTurns()
    turns.observe(text="السلام عليكم", is_final=False, speaker_id=_SPEAKER)
    secret = "السلام عليكم"

    _on_error(
        turns,
        ErrorEvent(
            error=STTError(
                timestamp=0,
                label="deepgram",
                error=TimeoutError(secret),
                recoverable=True,
            ),
            source=None,
        ),
    )

    assert turns.finalized == ()
    assert turns.partial_text is None
    text = stream.getvalue()
    assert secret not in text
    log = json.loads(text)
    assert log["event"] == "provider_failure"
    assert log["provider"] == "deepgram"
    assert log["operation"] == "transcribe"
    assert log["error_type"] == "TimeoutError"

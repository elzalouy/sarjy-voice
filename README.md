# Sarjy Voice Agent

Python package for the Sarjy voice agent.

## Runtime

| Tool | Version |
|------|---------|
| Python | 3.13.16 |
| uv | 0.12.24 |
| LiveKit Agents | 1.8.5 |
| LiveKit Deepgram plugin | 1.8.5 |
| LiveKit ElevenLabs plugin | 1.8.5 |

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in every empty value in `.env`. Do not commit that file.

## Environment

| Variable | Required | Default |
|----------|----------|---------|
| `LIVEKIT_URL` | Yes | |
| `LIVEKIT_API_KEY` | Yes | |
| `LIVEKIT_API_SECRET` | Yes | |
| `DEEPGRAM_API_KEY` | Yes | |
| `ELEVEN_API_KEY` | Yes | |
| `OPENAI_API_KEY` | Yes | |
| `NEXTJS_API_BASE_URL` | Yes | |
| `AGENT_SERVICE_TOKEN` | Yes | |
| `TTS_VOICE_ID` | Yes | |
| `STT_MODEL` | No | `nova-3` |
| `STT_LOCALE` | No | `ar-SA` |
| `TTS_MODEL` | No | `eleven_flash_v2_5` |

If a required value is missing, startup stops. The error lists the setting names. Secret values are not printed and are not written to logs.

## Run

```bash
uv run sarjy-agent
```

When settings are valid, the command prints `sarjy-agent ready` and starts the LiveKit worker. The worker uses the validated `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. Standard error receives JSON log lines.

`uv run sarjy-agent dev` starts the same worker in development mode.

The worker follows the official LiveKit Agents starter (`AgentServer` and `cli.run_app`). The dispatch name is `sarjy`. See Dispatch below. Speech recognition uses the Deepgram plugin. Speech playback uses the ElevenLabs plugin. Memory and tools are not connected.

## Dispatch

LiveKit does not add this worker to every new room. A room starts the worker only through explicit dispatch.

| Path | What the server sends | Room |
|------|----------------------|------|
| Browser session | Participant token with a room grant and a room configuration that dispatches `sarjy` | The room on the token |
| Server API | `CreateAgentDispatchRequest` for agent `sarjy` and one room | The room on the request |

`browser_dispatch()` is the agent entry for a participant token. `dispatch_request(room_name)` builds the API request for one room. Neither function adds metadata. The API secret stays on the server. The browser receives only the participant token.

When a job arrives, the worker checks the job itself:

| Check | Accept | Reject |
|-------|--------|--------|
| Agent name | `sarjy` | Any other name |
| Dispatch id | Short id on the job | Missing or free text |
| Job id | Short id on the job | Free text |
| Room | Short id on the job, and the connected room has that same name | Missing, free text, or a different room |

A short id is 1–128 characters: letters, digits, and `_` `.` `:` `-`.

The worker joins that job room only. It does not read a user id from job metadata, room metadata, or participant identity. Those values are not proof of who the caller is. The agent participant identity is `agent-` plus the job id.

The worker leaves the room on shutdown, cancellation, or a lost connection. A rejected dispatch is a `provider_failure` log with provider `livekit`, operation `dispatch`, and an `error_type` of `AgentMismatch`, `MissingDispatch`, `UnsafeJob`, `UnsafeRoom`, or `RoomMismatch`.

## Speech recognition

Room audio goes to Deepgram through `livekit-plugins-deepgram`. The plugin opens a streaming connection with the direct `DEEPGRAM_API_KEY`. The browser does not receive that key.

| Setting | Role | Default |
|---------|------|---------|
| `STT_MODEL` | Deepgram model | `nova-3` |
| `STT_LOCALE` | Recognition language | `ar-SA` |

Change `STT_LOCALE` to use another language. Saudi Arabic stays the default. A job loads these values when it starts.

Arabic speech in a room produces transcription events on the agent session.

| Recognition event | What the session keeps |
|-------------------|------------------------|
| Interim text | The current partial text only. It is not a saved message. |
| Stable piece of the same turn | Held until the user finishes speaking. |
| Finished turn | One message, with its turn id and speaker id. |
| Empty text | Nothing. |
| Recognition failure | No message. A `provider_failure` log is written. |

A second copy of the same finished turn is not saved. The spoken text stays in the process. It is not sent to the backend yet.

To check a real session on this Mac, with the key in `.env`:

```bash
SARJY_LIVE_STT=1 uv run pytest tests/test_stt_live.py
```

The check speaks `السلام عليكم` with the local Arabic voice and waits for a transcription event. It does not print the API key.

## Speech playback

Spoken text goes to ElevenLabs through `livekit-plugins-elevenlabs`. The plugin streams audio with the direct `ELEVEN_API_KEY`. The browser does not receive that key.

| Setting | Role | Default |
|---------|------|---------|
| `TTS_MODEL` | ElevenLabs model | `eleven_flash_v2_5` |
| `TTS_VOICE_ID` | ElevenLabs voice | Required. Use a voice that speaks Saudi Arabic. |
| `STT_LOCALE` | Language sent to the model | `ar-SA`, sent as `ar` |

Flash v2.5 is the primary model. A job loads `TTS_MODEL` and `TTS_VOICE_ID` when it starts. The voice id is not written in source code.

The same corpus was spoken with Flash v2.5 and Multilingual v2, using voice `pNGUEZc2WLum9bjP611g` for both. The key cannot list the voice library, so no other voice was tested. This is not a claim that the voice is the best Saudi voice. A person did not judge naturalness or accent.

| Measure | Flash v2.5 | Multilingual v2 |
|---------|------------|-----------------|
| Median time to first audio | 189 ms | 945 ms |
| P95 time to first audio | 338 ms | 1233 ms |
| Nova-3 accepted sentences | 4 / 37 | 3 / 37 |
| Mean word error rate | 0.462 | 0.455 |

Nova-3 is a hearing check, not a listener. The word error rates are close. Multilingual v2 was about five times slower to the first audio frame. Flash v2.5 stays the primary model.

```bash
uv run python -m sarjy_agent.eval.tts_bench
```

The report is `reports/tts/comparison.md`. Each row records the model and the voice id.

The agent session uses this speaker. When the session speaks, audio is streamed to the room.

Flash v2.5 does not spell numbers, dates, or prices. ElevenLabs documents that phone numbers, dates, and currencies are unclear, and that `apply_text_normalization` set to `on` is for Enterprise plans only. `apply_language_text_normalization` is for Japanese only. This account was not verified as Enterprise, so the speaker sends `apply_text_normalization=off` and does not send the Japanese option.

The process spells the evaluation numbers before synthesis:

| Digit text | Spoken text sent to the model |
|------------|--------------------------------|
| `0550000001` | One word per digit, including the leading zero |
| `+966500000002` | `زائد`, then one word per digit |
| `15.50 ريال` | `خمسة عشر ريال ونص` |
| `49.90 ريال` after `بـ` | `تسعة وأربعين ريال وتسعين هللة` |
| `9 أكتوبر 2026` | `تسعة أكتوبر ألفين وستة وعشرين` |
| `15 رمضان` | `خمسة عشر رمضان` |
| `3:30` | `ثلاثة ونص` |
| `8:00` | `ثمانية` |

Other text is unchanged. A bare number such as `iPhone 15` is not rewritten. The evaluation file still stores the digit text. The spoken form is applied only at synthesis.

To check a real stream on this Mac, with the key in `.env`:

```bash
SARJY_LIVE_TTS=1 uv run pytest tests/test_tts_live.py
```

The check speaks `السلام عليكم` and waits for streamed audio. It prints the model, voice id, duration, and time to the first audio chunk. It does not print the API key.

## Logs

Each log line is one JSON object. These fields are always present:

| Field | Meaning |
|-------|---------|
| `timestamp` | UTC time |
| `level` | `INFO` or `ERROR` |
| `event` | What happened |
| `room_id` | LiveKit room id, or `null` |
| `session_id` | Voice session id, or `null` |
| `turn_id` | Turn id, or `null` |

Startup writes `process_starting`, then `timing` for settings load, then `process_ready`. `process_ready` includes the speech model, locale, and voice id. If settings are missing, the event is `process_start_failed` and the line lists the setting names.

A room session writes `session_started` after the worker joins. `participant_joined` and `participant_left` record remote participants. `participant_id` is included only when it is a short id. `session_ended` means the worker left. Its `end_reason` is `shutdown`, `cancelled`, or `connection_lost`.

Provider failures use `provider_failure` with `provider`, `operation`, and `error_type`. A LiveKit connect, disconnect, unexpected room drop, or rejected dispatch uses provider `livekit`. A rejected dispatch uses operation `dispatch`. A Deepgram recognition failure uses provider `deepgram` and operation `transcribe`. The error text is not logged. Timing uses `timing` with `stage` and `duration_ms`.

A finished user turn writes `user_turn_final`. Its `turn_id` is set. The spoken text is not logged.

Logs do not include API keys, the service token, full conversations, or raw audio.

## Arabic evaluation set

Fixed Saudi Arabic text for later speech-provider comparisons.

| Field | Value |
|-------|-------|
| Version | `1.0.0` |
| Locale | `ar-SA` |
| File | `src/sarjy_agent/eval/data/ar_sa_v1.json` |

The set has 37 authored examples:

| Group | Count |
|-------|-------|
| Greetings | 5 |
| Questions and instructions | 5 |
| Names and places | 5 |
| Phone, price, date, and time | 2 each |
| Arabic mixed with English words | 5 |
| Ambiguous meaning or unclear speech | 9 |

Each example has an `expected_transcript` for speech-to-text and a `tts_text` for synthesis. Number examples record both the spoken words and the digit form. `acceptable_transcripts` lists other correct writings of the same utterance.

Use the same version, the same item ids, and the same order for every provider comparison. Load the set with `load_corpus()` from `sarjy_agent.eval`.

This version is text only. Add audio only in a new corpus version, and only when the speaker has given permission.

The speech-recognition report speaks each `tts_text` with ElevenLabs, then sends that audio to Deepgram. It does not save the audio. A hypothesis is accepted when it matches the expected transcript or an accepted writing. The report records accepted items, word error rate, and the wait from the end of the speech to the final transcript.

```bash
uv run python -m sarjy_agent.eval.stt_bench
```

The ElevenLabs result is `reports/stt/nova-3_ar-SA_eleven_flash_v2_5.md`. The earlier Mac voice result stays in `reports/stt/nova-3_ar-SA.md`. Both use the same corpus fingerprint and the same item ids. A new run names the items that changed.

If the text changes, bump `version` and update the fingerprint checked by the tests.

## Checks

`uv sync` installs these dev tools. Exact versions are in `uv.lock`.

| Check | Tool | Locked version |
|-------|------|----------------|
| Lint and format | Ruff | 0.16.10 |
| Unit tests | pytest | 9.1.1 |
| Static types | mypy | 2.4.0 |

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

`uv run ruff format` writes format fixes. mypy checks `src` in strict mode.

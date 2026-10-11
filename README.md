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
| `TTS_PROVIDER` | No | `elevenlabs` |
| `TTS_MODEL` | No | `eleven_flash_v2_5` |
| `TTS_LOCALE` | No | `ar-SA` |
| `TTS_TEXT_NORMALIZATION` | No | `off` |

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
| `TTS_PROVIDER` | Which speaker builder to use | `elevenlabs` |
| `TTS_MODEL` | Model name for that provider | `eleven_flash_v2_5` |
| `TTS_VOICE_ID` | Voice for that provider | Required. Use a voice that speaks Saudi Arabic. |
| `TTS_LOCALE` | Language sent to the speaker | `ar-SA`, sent as `ar` |
| `TTS_TEXT_NORMALIZATION` | Provider number normalization | `off` |

The room session only calls `create_tts(settings)`. It does not name the model or the provider. Change `TTS_MODEL` to switch models. The session startup code stays the same.

`TTS_LOCALE` is separate from `STT_LOCALE`. Recognition can stay on `ar-SA` while you try another speaker language.

To compare a future provider:

1. Add a builder next to `_elevenlabs_speaker` in `src/sarjy_agent/tts.py`. It should read `TtsConfig` and the server-side key only.
2. Set `TTS_PROVIDER` to that builder's name.
3. Set `TTS_MODEL`, `TTS_VOICE_ID`, and `TTS_LOCALE` for the new provider.
4. Speak the same evaluation sentences and record the model and voice id, as `reports/tts/comparison.md` does for Flash v2.5 and Multilingual v2.

```bash
uv run python -m sarjy_agent.eval.tts_bench
```

That command compares two ElevenLabs models with the configured voice. It does not change which model the room session uses.

Flash v2.5 is the primary model. A job loads these speaker settings when it starts. The voice id is not written in source code.

The same corpus was spoken with Flash v2.5 and Multilingual v2, using voice `pNGUEZc2WLum9bjP611g` for both. The key cannot list the voice library, so no other voice was tested. This is not a claim that the voice is the best Saudi voice. A person did not judge naturalness or accent.

| Measure | Flash v2.5 | Multilingual v2 |
|---------|------------|-----------------|
| Median time to first audio | 189 ms | 945 ms |
| P95 time to first audio | 338 ms | 1233 ms |
| Nova-3 accepted sentences | 4 / 37 | 3 / 37 |
| Mean word error rate | 0.462 | 0.455 |

Nova-3 is a hearing check, not a listener. The word error rates are close. Multilingual v2 was about five times slower to the first audio frame. Flash v2.5 stays the primary model.

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

Each clip is a pair in `src/sarjy_agent/eval/data/`.

| File | Role |
|------|------|
| `ar-sa-001.wav` | The recording sent to speech recognition. Mono 16-bit PCM. |
| `ar-sa-001.txt` | One line: the words spoken in that wav. |

The two files must share the same name. A wav with no txt, or a txt with no wav, stops the load. `load_corpus()` from `sarjy_agent.eval` reads the folder in filename order. The score checks Deepgram against the txt line.

The speech-recognition report sends each wav to Deepgram. It does not synthesize a new clip. A hypothesis is accepted when it matches the txt, or another accepted writing when one is stored. The report records accepted items, word error rate, and the wait from the end of the wav to the final transcript.

```bash
uv run python -m sarjy_agent.eval.stt_bench
```

The report is written under `reports/stt/`. A later run can be compared only when the wav bytes and the txt lines are the same. Changing a recording or a transcript changes the fingerprint.

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

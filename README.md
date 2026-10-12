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
| `LLM_MODEL` | No | `gpt-4.1-mini` |
| `LLM_TIMEOUT_SECONDS` | No | `30` |
| `BACKEND_TIMEOUT_SECONDS` | No | `2` |

If a required value is missing, startup stops. The error lists the setting names. Secret values are not printed and are not written to logs.

## Run

```bash
uv run sarjy-agent
```

When settings are valid, the command prints `sarjy-agent ready` and starts the LiveKit worker. The worker uses the validated `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. Standard error receives JSON log lines.

`uv run sarjy-agent dev` starts the same worker in development mode.

The worker follows the official LiveKit Agents starter (`AgentServer` and `cli.run_app`). The dispatch name is `sarjy`. See Dispatch below. Speech recognition uses the Deepgram plugin. Speech playback uses the ElevenLabs plugin. Text replies use the OpenAI Responses API. The OpenAI Realtime speech API is not used. Before the worker listens, it asks the Next.js backend to authorize the LiveKit room. Saved preferences are loaded as facts. The model can save a name, a city, or a language, and it can ask the backend for weather. Each finished user line and each finished assistant line is saved.

A finished user turn is answered out loud. The text model streams the reply. A phrase is spoken as soon as it is long enough. A single word is not spoken by itself. A long reply is not held until the last word. That audio is published in the room, so the participant hears it.

`LLM_MODEL` selects the text model. The default is `gpt-4.1-mini`. `LLM_TIMEOUT_SECONDS` is the wait for streamed text. The default is 30 seconds. `OPENAI_API_KEY` stays on the server.

The system prompt is fixed. It asks for one or two short sentences in everyday Saudi Arabic. If the assistant does not know, or the request is not supported, it says so. It does not invent prices, weather, or other outside facts. A tool result is a separate message, and the reply uses that result. Facts about the person are also a separate message. They are not part of the instructions, and they are not orders. No tool is connected yet.

```bash
SARJY_LIVE_LLM=1 uv run pytest tests/test_llm_live.py tests/test_instructions_live.py -s
```

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

The worker joins that job room only. It reads `conversationId` from the job metadata when that value is a UUID. It does not read a user id from job metadata, room metadata, or participant identity. Those values are not proof of who the caller is. The agent participant identity is `agent-` plus the job id.

The worker leaves the room on shutdown, cancellation, or a lost connection. A rejected dispatch is a `provider_failure` log with provider `livekit`, operation `dispatch`, and an `error_type` of `AgentMismatch`, `MissingDispatch`, `UnsafeJob`, `UnsafeRoom`, or `RoomMismatch`.

## Speech recognition

Room audio goes to Deepgram through `livekit-plugins-deepgram`. The plugin opens a streaming connection with the direct `DEEPGRAM_API_KEY`. The browser does not receive that key.

| Setting | Role | Default |
|---------|------|---------|
| `STT_MODEL` | Deepgram model | `nova-3` |
| `STT_LOCALE` | Recognition language | `ar-SA` |

Change `STT_LOCALE` to score another language on the Arabic recording set. Saudi Arabic stays the default for that check.

A live call does not use Nova-3 `ar-SA` for mixed speech. That model is Arabic only. Nova-3 `language=multi` code-switches among ten languages, and Arabic is not one of them. The call uses OpenAI `gpt-live-transcribe` with languages `ar` and `en` in the same utterance. English words, product names, and identifiers are expected to stay in Latin letters.

The fixed mixed lines are in `sarjy_agent.eval.code_switch`. They are scored with the same word split as the Arabic corpus. A heard line is accepted only when every required English word and identifier is still present. Latency is recorded on the same score.

Arabic speech in a room produces transcription events on the agent session.

| Recognition event | What the session keeps |
|-------------------|------------------------|
| Interim text | The current partial text only. It is not a saved message, and it does not start a reply. |
| Stable piece of the same turn | Held until the user finishes speaking. |
| Finished turn | One message, with its turn id and speaker id. |
| Empty text | Nothing. |
| Recognition failure | No message. A `provider_failure` log is written. |

A second copy of the same finished turn is not saved. The finished line is sent once to the Next.js backend. A partial line stays in the open turn and is not sent.

The session uses Silero to hear speech and silence, and the local turn detector `v1-mini`. That detector supports Arabic. A reply waits for silence. The wait is longer when the turn does not look finished.

| Wait | Seconds | Meaning |
|------|---------|---------|
| Shortest speech that counts | 0.05 | A short word such as نعم still counts. A click does not. |
| Silence before a possible end | 0.55 | A breath is not the end of speech. |
| Extra wait when the turn looks finished | 0.5 | The agent may reply after this. |
| Extra wait when the turn looks unfinished | 2.5 | A pause in the middle does not start a reply. |

A pause shorter than those waits does not start a reply. If the person speaks again, the wait starts over. One pause does not create a second reply.

If the person speaks over the agent, the current answer stops. Speech shorter than 0.5 seconds does not count, and no word minimum is required. The old answer does not keep playing. The new turn gets a new answer. The call stays open. LiveKit does this with adaptive interruption.

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

## Voice turn

One turn is hearing, a text reply, then speech.

| Step | What happens |
|------|----------------|
| 1 | The participant's microphone audio arrives in the room. The worker sends it to Deepgram. |
| 2 | A normal pause does not end the turn. When the person has stopped long enough, that text is one finished user turn. Interim text does not start a reply. |
| 3 | The text model streams a short Arabic reply. |
| 4 | ElevenLabs starts a phrase as soon as that phrase is ready. A one-word piece waits. A long reply starts before the last word. |
| 5 | The spoken audio is published in the room. The participant hears it. |
| 6 | If the person speaks over the answer for at least 0.5 seconds, that answer stops. The new turn is answered. The call stays open. |
| 7 | The finished user line is saved, then the finished assistant line is saved. An interrupted answer is saved as interrupted. |

```bash
SARJY_LIVE_VOICE=1 uv run pytest tests/test_voice_turn_live.py -s
```

The check speaks `السلام عليكم` with the local Arabic voice and waits for a spoken reply. It does not print API keys.

The call keeps messages in the order they happened. Instructions stay first. Facts and tool results stay in their own messages, and a tool call is not kept without its result. A partial line is not stored next to the finished line. Only the latest spoken lines are kept, and the newest user line stays. A long reply starts speaking before the last sentence is ready, in order. A normal answer is one or two sentences. A longer answer is allowed when the person asks for more detail.

The phrase cut is compared with the old sentence cut on the same text. Each side records when the first character arrives and when the first phrase is ready for speech. A comma phrase and a long reply are ready sooner. A one-word sentence is not sent on its own. A live reply also logs `llm_first_token` and `tts_first_audio`. Those two log lines are the model's first text token and the speaker's first audio frame. The comparison itself does not call the speaker, so the difference is only the cut.

## Next.js backend

The worker calls `NEXTJS_API_BASE_URL` directly. `AGENT_SERVICE_TOKEN` is sent as `Authorization: Bearer ...`. The browser does not receive that token. Redirects are not followed. `BACKEND_TIMEOUT_SECONDS` is the wait for one attempt. The default is 2 seconds. A timeout, a 408, a 429, or a 5xx response is tried once more. A 401, 403, 404, or 409 is not retried.

Each attempt sends a new `X-Request-Id`. A failed attempt logs that id.

The Next.js server puts `conversationId` in the agent dispatch metadata. The worker sends that id with the LiveKit room name. It does not send a user id. The authorize response names the owner. Later calls use that owner id.

### Authorize the room

`POST /api/internal/voice-sessions/authorize`

| Field | Source |
|-------|--------|
| `conversationId` | Job metadata |
| `roomName` | LiveKit room name |

A 200 response is `conversationId`, `userId`, and `roomName`. The room and conversation must match the request. A 409 means the conversation is not active, and the worker does not listen. A tool argument cannot replace `userId`.

### Save a finished line

`POST /api/internal/conversations/{conversationId}/transcripts`

| Field | Meaning |
|-------|---------|
| `userId` | The owner from authorize |
| `role` | `user` or `assistant` |
| `content` | The finished line. An empty line is not sent |
| `eventId` | The same key is stored once |
| `createdAt` | UTC time |

The first save is 201. The same `eventId` again is 200 and does not create a second line. Partial speech is not sent. An interrupted reply is saved when it has words.

### Memory

`GET`, `PUT`, `PATCH`, and `DELETE` `/api/internal/users/{userId}/memory`

The path user id is the authorized owner. Every fact is sent to Next.js. A password, token, email, or phone number is rejected before the call. Loaded facts are a separate message. They are not instructions.

The model is told a fact was saved only when the backend returns that row. If memory is down at the start of the call, the conversation continues with no facts.

### Weather

`POST /api/internal/weather` with `location` set to a city or `latitude,longitude`. A web address is rejected. The weather key stays on the Next.js server. The tool result contains the place, condition, temperature, and humidity. If weather is down, the model is told that weather is unavailable.

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

Provider failures use `provider_failure` with `provider`, `operation`, and `error_type`. A LiveKit connect, disconnect, unexpected room drop, or rejected dispatch uses provider `livekit`. A rejected dispatch uses operation `dispatch`. A Deepgram recognition failure uses provider `deepgram` and operation `transcribe`. A Next.js failure uses provider `nextjs` and operation `authorize` or `save_transcript`. That line includes `request_id`, the `X-Request-Id` of the failed attempt. The error text is not logged. Timing uses `timing` with `stage` and `duration_ms`.

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

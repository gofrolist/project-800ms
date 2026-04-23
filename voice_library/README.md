# voice_library/

Voice-clone profiles consumed by two engines that share the same directory shape:

- **Qwen3-TTS sidecar** — reads this directory when run with a `Base` variant (`QWEN3_VARIANT=0.6B-Base`, `1.7B-Base`). The sidecar mounts this directory read-only via `infra/docker-compose.yml` and looks up profiles by name from `profiles/<name>/meta.json`.
- **XTTS v2** (in-process in the agent) — reads the same directory shape from `XTTS_VOICE_LIBRARY_DIR` (default `/opt/voice_library` inside the agent container; see `services/agent/xtts_tts.py::_resolve_voice_profile`). A single profile serves both engines without duplicating reference audio.

Profile management is **out-of-band by design**. There is no API to upload, edit, or delete profiles; operators copy files in via `rsync`/`scp` and restart the containers. If your deployment needs interactive profile management, add dedicated endpoints — don't conflate session dispatch with asset management.

`CustomVoice` variants of Qwen3 ignore this directory — they use preset speakers via the OpenAI voice whitelist (`alloy`, `echo`, ...). XTTS v2 is voice-cloning only; every XTTS session needs a profile here. Both engines exercise this path when a request arrives with `voice="clone:<profile>"`.

## Layout

```
voice_library/
  README.md                 # this file
  profiles/
    <profile-name>/
      meta.json             # profile metadata
      ref.wav               # reference audio (any name; pointed at by meta)
```

## `meta.json` schema

```json
{
  "profile_id": "demo-ru",
  "name": "Demo Russian",
  "ref_audio_filename": "ref.wav",
  "ref_text": "Привет! Меня зовут Евгений, я тестирую систему.",
  "language": "Russian",
  "x_vector_only_mode": false
}
```

Fields:
- **`profile_id`** — stable identifier. Must be URL-safe.
- **`name`** — human-readable display name. Case-insensitive match against incoming `voice="clone:<name>"` requests.
- **`ref_audio_filename`** — relative to this profile directory.
- **`ref_text`** — exact transcript of the reference audio, required when `x_vector_only_mode=false`. Drives ICL (in-context learning) cloning: the model sees the audio↔text alignment as a conditioning prompt. Bad transcripts → bad output.
- **`language`** — for Qwen3: one of `"English"`, `"Russian"`, `"Chinese"`, `"Japanese"`, `"Korean"`, `"German"`, `"French"`, `"Spanish"`, `"Portuguese"`, `"Italian"` (see `infra/qwen3-tts-wrapper/api/routers/openai_compatible.py::LANGUAGE_CODE_MAPPING`). For XTTS: the 17 languages XTTS v2 supports (Arabic, Chinese, Czech, Dutch, English, French, German, Hindi, Hungarian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Turkish); either the English name (`"Russian"`) or XTTS code (`"ru"`) resolves via `_XTTS_LANGUAGE_NAME_TO_CODE` in `services/agent/xtts_tts.py`. Unknown values fall back to Russian.
- **`x_vector_only_mode`** — `true` skips ICL and uses only the reference's x-vector speaker embedding. Slightly lower quality but doesn't need `ref_text`. `false` (ICL) recommended when you have a clean transcript.

## Reference audio requirements

- Duration: 3-10 seconds
- Sample rate: >= 16 kHz (sidecar resamples as needed)
- Mono preferred; stereo is averaged on load
- Clean speech, minimal noise, no music/reverb
- Match the synthesis language — a Russian profile needs Russian reference speech

## Privacy

Reference audio files and transcripts are excluded from the repo via `.gitignore` — only this README and the directory scaffold are tracked. Keep profiles out of public repos unless the recorded speaker has consented to publication.

## Usage

### Qwen3 sidecar

```env
TTS_ENGINE=qwen3
QWEN3_VARIANT=0.6B-Base
QWEN3_TTS_VOICE=clone:<profile-id>
```

Restart the agent + qwen3-tts containers, then dispatch a session targeting the `qwen3` engine — the factory routes to the voice-library profile without going through the OpenAI voice whitelist.

### XTTS v2 (in-process)

```env
TTS_PRELOAD_ENGINES=piper,silero,qwen3,xtts
XTTS_TTS_VOICE=clone:<profile-id>
XTTS_VOICE_LIBRARY_DIR=/opt/voice_library
```

The agent bind-mounts `../voice_library:/opt/voice_library:ro` at compose level. Profile resolution happens at session-dispatch time (not at XTTS preload) — add or replace profiles without restarting the agent. The same `clone:<profile-id>` value works for both engines, so a single `demo-ru` profile serves both Qwen3 and XTTS sessions.

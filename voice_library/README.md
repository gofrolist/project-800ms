# voice_library/

Voice-clone profiles for the Qwen3-TTS sidecar, consumed when an operator runs a `Base` variant (`QWEN3_VARIANT=0.6B-Base`, `1.7B-Base`). The sidecar mounts this directory read-only via `infra/docker-compose.yml` and looks up profiles by name from `profiles/<name>/meta.json`.

`CustomVoice` variants ignore this directory — they use preset speakers via the OpenAI voice whitelist (`alloy`, `echo`, ...). This path is only exercised when a request arrives with `voice="clone:<profile>"`.

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
- **`language`** — one of `"English"`, `"Russian"`, `"Chinese"`, `"Japanese"`, `"Korean"`, `"German"`, `"French"`, `"Spanish"`, `"Portuguese"`, `"Italian"`. See `infra/qwen3-tts-wrapper/api/routers/openai_compatible.py::LANGUAGE_CODE_MAPPING`.
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

Agent side: set
```env
TTS_ENGINE=qwen3
QWEN3_VARIANT=0.6B-Base
QWEN3_TTS_VOICE=clone:<profile-id>
```
in `infra/.env`, restart the agent + qwen3-tts containers, and dispatch a session targeting the `qwen3` engine — the factory routes to the voice-library profile without going through the OpenAI voice whitelist.

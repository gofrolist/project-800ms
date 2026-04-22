---
title: "Silero v5 TTS — adapter-integration spike (R22)"
date: 2026-04-21
module: services/agent
component: tts
category: tts-selection
problem_type: best_practice
tags: [tts, silero, pipecat, russian-voice, adapter-spike]
related_components: [services/agent/pipeline.py, services/agent/gigaam_stt.py, services/agent/models.py]
resolution_type: research_memo
---

# Silero v5 TTS — adapter-integration spike (R22)

## Purpose

R22 from `docs/plans/2026-04-21-001-feat-tts-abstraction-piper-silero-qwen3-plan.md`: 4-hour timeboxed investigation to confirm the Silero v5 TTS Pipecat adapter can be built in the plan's 1-2 day estimate without hidden week-class issues. Produces go/no-go before committing to Unit 3.

## Go/no-go: **GO**

No week-class blockers found. Integration path is clean. Estimated adapter work: 1-2 days as planned.

## Silero v5 Python API (upstream — `snakers4/silero-models`)

Silero v5 has no PyPI package. Load via torch.hub at runtime:

```python
import torch
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v5_cis_base',
    trust_repo=True,
)
```

- Returns a callable model object + a discarded example-speaker-map.
- First load downloads the repo (~5 MB) + the model weights (~200 MB) to `$TORCH_HOME/hub/` and `$TORCH_HOME/hub/checkpoints/`. Plan sets `TORCH_HOME=/home/appuser/.cache/torch` mounted via new `torch_cache_agent` volume.
- Subsequent loads are local-only (no network) if the cache survives.

Synthesis API:

```python
audio = model.apply_tts(
    text="привет мир",
    speaker=None,               # v5_cis_base is a single-speaker model; speaker arg unused
    sample_rate=24000,          # 8000 | 24000 | 48000 supported
    put_accent=True,            # automatic Russian stress placement
    put_yo=True,                # automatic ё normalization
)
# Returns a torch.Tensor of shape [N] (mono int16-equivalent, float32 in [-1, 1]).
# For a ~5-word Russian phrase → ~1 second audio → ~24000-sample tensor.
```

- **Blocking call.** Runs on the model's bound device (GPU if `model.to('cuda')` was called after load; else CPU).
- Output needs conversion: `(audio.numpy() * 32767).astype(np.int16).tobytes()` to get the int16 PCM bytes Pipecat's helper expects.
- RTF on L4 GPU per upstream benchmarks: ~0.06 (1 s of audio produced in ~60 ms). File-at-a-time — entire output returns in one call.

Sample-rate trade-off:
- **24 kHz** — our default. Matches LiveKit transport expectation, zero resample overhead via Pipecat's `_stream_audio_frames_from_iterator`.
- **48 kHz** — marginally better perceptual quality but doubles the byte count and forces a Pipecat-side resample to 24 kHz for LiveKit. Not worth it for MVP.
- **8 kHz** — telephony quality. Skip.

## Pipecat TTSService contract (0.0.108)

From `services/agent/.venv/lib/python3.12/site-packages/pipecat/services/tts_service.py`:

- Base class: `pipecat.services.tts_service.TTSService(AIService)` — abstract, one required override.
- `__init__` accepts (subset relevant to us): `sample_rate: Optional[int]`, `push_start_frame: bool = False`, `push_stop_frames: bool = False`, `settings: Optional[TTSSettings]`, and `**kwargs` for the AIService base.
- **Recommended init kwargs (from PiperTTSService precedent at lines 94-95):** `push_start_frame=True, push_stop_frames=True`. Without these, downstream transport layers don't see the TTS lifecycle frames and barge-in/interruption gating doesn't work.
- Abstract method:
  ```python
  async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]
  ```
  Must be an async generator. Yields `TTSAudioRawFrame` chunks and optionally `ErrorFrame` on failure. `TTSStartedFrame` / `TTSStoppedFrame` are pushed automatically when `push_start_frame` / `push_stop_frames` are enabled.
- Helper: `self._stream_audio_frames_from_iterator(iterator, *, in_sample_rate, context_id)` at line 847. Takes an async iterator of `bytes`, optionally resamples from `in_sample_rate` to `self.sample_rate`, and wraps into properly-framed `TTSAudioRawFrame`s with `context_id` propagated. **This is the one-call helper that handles sample-rate conversion, chunk alignment, and frame wrapping** — our adapter should use it directly rather than constructing frames by hand.

## Compatibility: Silero + Pipecat, concrete shape

File-at-a-time Silero maps cleanly onto Pipecat's iterator helper:

```text
# Pseudo-code (directional, not implementation spec)
class SileroTTSService(TTSService):
    def __init__(self, *, silero_model, settings=None, **kwargs):
        kwargs.setdefault("push_start_frame", True)
        kwargs.setdefault("push_stop_frames", True)
        kwargs.setdefault("sample_rate", 24000)
        kwargs.setdefault("ttfs_p99_latency", 1.0)  # refined post-Unit 5 bench
        super().__init__(settings=TTSSettings(model="v5_cis_base", language="ru"), **kwargs)
        self._silero = silero_model

    async def run_tts(self, text, context_id):
        try:
            await self.start_tts_usage_metrics(text)
            # Blocking Silero call off-thread; synthesizes the full utterance.
            audio_tensor = await asyncio.to_thread(
                self._silero.apply_tts, text=text, sample_rate=24000,
                put_accent=True, put_yo=True,
            )
            pcm = (audio_tensor.numpy() * 32767).astype(np.int16).tobytes()

            async def _one_shot():
                yield pcm

            async for frame in self._stream_audio_frames_from_iterator(
                _one_shot(), in_sample_rate=24000, context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.exception("Silero TTS synth failed")
            yield ErrorFrame(error="TTS synth failed")  # redacted
```

This mirrors `PiperTTSService.run_tts` (which also uses `_stream_audio_frames_from_iterator`) and the `GigaAMSTTService` exception-redaction idiom.

## Barge-in / interruption semantics

**File-at-a-time has a real cost.** Piper emits chunks as it synthesizes — so Pipecat's `InterruptionFrame` can truncate output mid-sentence. Silero returns the full buffer first, then yields frames; once the first frame reaches `transport.output()`, barge-in CAN still interrupt the playback (LiveKit output transport honors cancellation), but the synth work is already done — the GPU time is already spent.

Failure mode that actually breaks user experience: **interruption DURING synth, before first frame yields.** The `await asyncio.to_thread(apply_tts)` is uncancellable from Python once it's running in a thread. If the user starts speaking during a long synth (10+ words, ~200-400 ms GPU time), the current pipeline's barge-in (`VADUserStartedSpeakingFrame` → `InterruptionFrame`) will fire but won't reach the blocked generator until `apply_tts` returns. Result: ~200-400 ms of "bot is about to say something but cancelled" silence instead of the faster Piper interrupt.

This is acceptable for R22's go/no-go gate. It's a known ceiling, not a bug. The plan's R11b current-Piper-TTFB baseline measurement will quantify the regression empirically in Unit 5.

## Risks flagged (none week-class)

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | `torch.hub.load` requires GitHub egress on first boot of any fresh container | Medium | New `torch_cache_agent` volume in `infra/docker-compose.yml`; healthcheck `start_period: 300s` tolerates first-run download. Pre-warm in CI if egress becomes an issue. |
| 2 | Silero's `apply_tts` blocks the event loop if not wrapped | Low | `asyncio.to_thread` wraps it (same pattern as GigaAM in `services/agent/gigaam_stt.py:155`). |
| 3 | Double-pin collision: `gigaam[torch]` already pulls torch; Silero needs the same torch | Low | Silero via `torch.hub` uses whatever torch is installed — inherits GigaAM's pin. No new torch dep in `pyproject.toml`. |
| 4 | Silero v5 weights licence | Low | Upstream repo is MIT (code) + CC0/public-domain (v5 Russian weights per Silero's release notes). Re-verify during Unit 3 implementation by reading the actual checkpoint license metadata. |
| 5 | Russian accent / stress placement may be inconsistent for game-domain vocabulary | Medium | `put_accent=True` + `put_yo=True` cover most cases. Domain-specific words (character names, place names) may need a custom stress dictionary — defer to post-experiment if it surfaces. |
| 6 | Barge-in latency regression (see previous section) | Medium | Accepted for MVP. Quantified via R11b baseline. If regression is >300ms vs Piper on long utterances, Unit 6's latency-budget gate should reject Silero despite quality wins. |

## Integration shape checklist (for Unit 3)

- [x] Confirm `torch.hub` repo + model name + speaker for Russian v5 — `snakers4/silero-models`, `silero_tts`, `v5_cis_base`.
- [x] Confirm `TTSService.run_tts(text, context_id)` abstract signature — verified in `tts_service.py:491`.
- [x] Confirm `_stream_audio_frames_from_iterator` helper exists and handles sample-rate conversion — verified at `tts_service.py:847`.
- [x] Confirm `push_start_frame=True, push_stop_frames=True` is the correct subclass pattern — verified in `PiperTTSService.__init__` at `piper/tts.py:94-95`.
- [x] Confirm exception-redaction pattern — mirror `GigaAMSTTService:171-177` (`ErrorFrame("TTS synth failed")`).
- [x] Confirm model preload + singleton pattern — mirror `services/agent/models.py` `load_gigaam` trio.
- [ ] **Deferred to Unit 3:** concrete `int16` normalization factor (32767 vs 32768 — Silero's docs vary). Verify with a test tone.
- [ ] **Deferred to Unit 3:** determine if `model.to('cuda')` needs to be called explicitly after `torch.hub.load` or if Silero returns a CUDA-bound model when GPU is available.

## Spike outcome

**GO.** Silero v5 integration is a standard Pipecat `TTSService` subclass following the `PiperTTSService` pattern. Main trade-off is file-at-a-time synth, which caps barge-in responsiveness but doesn't prevent the experiment. No hidden sample-rate, streaming-contract, or torch-compat issues.

Proceed to Unit 3 after R21 premise listen passes.

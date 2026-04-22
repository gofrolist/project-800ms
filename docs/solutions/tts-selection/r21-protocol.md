---
title: "R21 Phase-0 premise blind-listen protocol"
date: 2026-04-21
module: services/agent
component: tts
category: tts-selection
problem_type: best_practice
tags: [tts, phase-0, blind-listen, premise-validation]
resolution_type: protocol
---

# R21 Phase-0 premise blind-listen protocol

## Purpose

Validate (or falsify) the experiment's driving premise — *"Piper's Russian voice sounds robotic enough to justify 2-3 weeks of engine-replacement engineering"* — before any code is written downstream of Unit 1.

This is the first gate of `docs/plans/2026-04-21-001-feat-tts-abstraction-piper-silero-qwen3-plan.md`. If R21 fails, the whole plan aborts: the team files a *"Piper acceptable for MVP"* solution doc and stops. If R21 passes, Units 2-8 unlock.

Three team raters are committed to this protocol.

## What you need

- **Raters:** 3 members of the team, each listening independently. No cross-talk during rating.
- **Materials:**
  - 10 identical Russian utterances synthesized on each of 3 engines: Piper (`ru_RU-denis-medium`), Silero v5 (`v5_cis_base`), Qwen3-TTS (0.6 B via the vendored sidecar).
  - A shared blinded directory (e.g., `docs/solutions/tts-selection/r21/<YYYY-MM-DD>/blinded/<utterance-id>/<random-label>.wav`).
  - The un-blind mapping (`r21/<YYYY-MM-DD>/_unblind.json`) stored **outside** the rater-accessible path.
- **Time:** ~60 minutes total. ~10 min to prepare materials + ~15-20 min per rater listening and ranking + ~5 min aggregation.

## Setup (one-time, by whoever prepares the materials)

1. **Pick 10 Russian utterances** that reflect realistic assistant output. Mix the categories:
   - 3 short conversational (~5 words): greetings, confirmations, short replies.
   - 4 medium (~15 words): typical assistant response lines.
   - 2 long (~40+ words): multi-clause explanations, stories, multi-step instructions.
   - 1 domain-specific: include brand/persona/game-specific vocabulary if applicable — this is where Piper-specific "robotic" complaints are loudest.

   Write the utterances to a CSV: `docs/solutions/tts-selection/r21/<YYYY-MM-DD>/corpus.csv` with columns `utterance_id, text_ru, class`.

2. **Synthesize each utterance on each engine.** Use the **same synth path, same sample rate (24 kHz), same post-processing** for all three — this is critical. Do NOT mix a production-pipeline Piper capture (with LiveKit/WebRTC artifacts) against vendor cherry-picked demos (studio-polished). Local inference only:
   - Piper: `python -m pipecat.services.piper.tts` or direct Piper CLI, emit 24 kHz WAV.
   - Silero v5: `torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v5_cis_base')` + `model.apply_tts(text, sample_rate=24000)`; convert to int16 WAV.
   - Qwen3-TTS: local container via the compose sidecar (needs Unit 4's sidecar up — for R21, an ad-hoc one-shot run is fine; alternatively use the official Qwen3-TTS demo notebook with `language=ru` to bootstrap without the full sidecar). Save 24 kHz PCM WAV.

   Every output WAV must be:
   - 24 kHz mono int16 PCM
   - no added reverb, EQ, compression, or loudness normalization beyond what the engines do by default
   - trimmed silence at leading/trailing edges to keep durations comparable

3. **Build the blinded directory.** For each utterance, create a subfolder and place the 3 WAVs under random labels:

   ```
   docs/solutions/tts-selection/r21/2026-MM-DD/
     corpus.csv
     blinded/
       utt-01/
         a.wav    # random — might be Piper
         b.wav    # random — might be Silero
         c.wav    # random — might be Qwen3
       utt-02/
         a.wav    # different random — DO NOT reuse the same a=Piper mapping
         b.wav
         c.wav
       ...
     _unblind.json   # {"utt-01": {"a": "piper", "b": "silero", "c": "qwen3"}, "utt-02": {...}}
   ```

   **Randomize the `a`/`b`/`c` → engine mapping per utterance.** If you keep the same mapping across utterances, raters quickly pattern-match and the blind is broken. Use `random.shuffle(["piper", "silero", "qwen3"])` inside a script that generates the mapping and writes `_unblind.json` outside `blinded/`.

4. **Commit the blinded directory** (but NOT `_unblind.json` until after aggregation). Alternatively keep `_unblind.json` in an untracked location (e.g., `.git/tts-r21-unblind.json` or a text note held by whoever prepared the corpus); never share it with raters until the rating phase is complete.

## Rating (each rater, independently)

Rater flow — takes ~15-20 min:

1. Open `docs/solutions/tts-selection/r21/<date>/corpus.csv` to see the reference Russian text for each utterance.
2. Listen to the three WAVs (`a.wav`, `b.wav`, `c.wav`) for each utterance. Each utterance's labels are independently shuffled — no cross-utterance leakage. Listen to each WAV at normal volume on whatever monitoring you normally use (headphones preferred to reduce room-coloring).
3. Rank the three outputs 1 (best) / 2 / 3 (worst) per utterance on **perceived naturalness + Russian pronunciation correctness**. Ties are allowed if you genuinely can't tell: write `1`, `1`, `2` (two tied for best) — but prefer to force a ranking when you can.
4. Optional free-text note per utterance — "sounds mechanical", "misreads stress on 'слова'", "clearly better prosody", etc. These notes are useful for post-hoc debugging but don't drive the decision.
5. Commit your per-rater CSV at `docs/solutions/tts-selection/r21/<date>/ratings-<rater-id>.csv` with columns `utterance_id, a_rank, b_rank, c_rank, notes`.

**Do not discuss with other raters until all three CSVs are committed.** If there's a technical glitch in one of the clips (silence, noise, wrong language), mark that utterance as `skip` in the rating CSV rather than guessing — but the preparation step should catch these before rating starts.

## Aggregation (whoever prepares the materials, after all 3 rater CSVs land)

1. Unblind using `_unblind.json`. Convert each rater's `a/b/c` ranks to `piper/silero/qwen3` ranks.
2. For each utterance, compute per-engine ranks per rater. Find: how many times was Piper ranked last (rank 3)?
3. Compute the aggregate. **Pass criterion:** Piper ranked last on the majority of utterances by at least 2 of 3 raters. Concretely:
   - For each rater, count utterances where they ranked Piper last. Rater "votes Piper last" if their count is ≥ 6 (majority of 10).
   - Experiment proceeds if ≥ 2 of 3 raters vote Piper last.
4. **Fail criterion:** 0 or 1 raters voted Piper last → premise is unvalidated. Commit a `docs/solutions/tts-selection/<date>-r21-piper-acceptable.md` solution doc summarizing what was heard and the decision to stay on Piper. Abort Units 2-8.

## What to do with the result

- **Pass (≥ 2 raters vote Piper last):** commit `docs/solutions/tts-selection/r21-findings.md` recording the rater votes, representative free-text notes, and the "proceed" decision. Unlock Units 2-8.
- **Fail (< 2 raters vote Piper last):** commit `docs/solutions/tts-selection/<date>-r21-piper-acceptable.md` recording the same evidence and the "Piper stays for MVP" decision. The plan is formally closed. Silero + Qwen3 adapter work is not done. (R22 findings + the Qwen3 wrapper vendoring are kept as Phase-0 artifacts for future reference — they're cheap to leave in-tree and valuable if the premise is re-raised later.)
- **Inconclusive (raters split 1-1-1 on which engine is worst, or systematically rank something other than Piper last):** this is a partial pass. Proceed, but note in the findings doc that the premise is weaker than the experiment assumed — Unit 6's Decisive-winner branch needs to work harder to pick, and "Piper wins" becomes more likely as the outcome.

## Why this protocol exists

The ce-doc-review feedback flagged three P1 issues that this protocol addresses directly:

1. **Confounded comparison risk** — using Piper production-pipeline captures against vendor cherry-picked demos would structurally bias against Piper. Protocol fix: all three engines go through the same local synth path at 24 kHz PCM.
2. **Solo-eval risk** — a single rater (usually a team member invested in the outcome) validating their own hypothesis. Protocol fix: 3 raters, independent, no cross-talk until aggregation.
3. **Ambiguous threshold wording** — the original plan said *"Piper not ranked last by ≥ 50% of listeners"* which is a double negative. Protocol fix: concrete voting rule ("≥ 2 of 3 raters, each Piper-last on ≥ 6 of 10 utterances") with no negations.

## References

- Plan R21: `docs/plans/2026-04-21-001-feat-tts-abstraction-piper-silero-qwen3-plan.md` Unit 1
- Origin brainstorm R21: `docs/brainstorms/2026-04-21-tts-abstraction-and-silero-qwen3-ab-requirements.md`
- Silero Python API: `docs/solutions/tts-selection/silero-spike-findings.md` (R22 output)
- Qwen3 wrapper request shape: `infra/qwen3-tts-wrapper/README.md` (R23 output)

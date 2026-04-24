# Phase 0 — Research: Helper/Guide NPC KB-Grounded Answers

Status: All "NEEDS CLARIFICATION" resolved. Decisions below are binding for Phase 1.

---

## R1 — Embedding model

**Decision**: BGE-M3 (`BAAI/bge-m3`), 1024-dim dense vectors, CPU or GPU.

**Rationale**:
- No `"query:"` / `"passage:"` prefix discipline — eliminates a silent-recall-loss
  footgun that hits `multilingual-e5-large` users who forget the prefix on one
  side. Ingestion and query paths live in different codebases; the prefix
  contract is the kind of thing that rots at 2am.
- Strong Russian retrieval on ruMTEB; 8192-token context (we'll never use it all,
  but headroom is free).
- 1024-dim matches the schema sketch already in the user's input.

**Alternatives considered**:
- `intfloat/multilingual-e5-large`: similar quality, prefix footgun.
- `sberbank-ai/sbert_large_nlu_ru`: Russian-only; smaller corpus coverage; 1024-dim.
- GigaEmbeddings: SOTA on ruMTEB but GPU-only, not materially better for
  retrieval, and adding a second GPU-hungry model on the voice host fights the
  GigaAM STT for VRAM.

**Constraints from decision**:
- Image base must include `FlagEmbedding` and `sentence-transformers`. Both go
  in `services/retriever/Dockerfile.base` with an import-smoke layer.
- BGE-M3 weights ~2 GB; persisted via new named volume `hf_cache_retriever`.

---

## R2 — Hybrid-search fusion formula

**Decision**: Weighted sum, `final = 0.7·cosine + 0.3·ts_rank_cd`, in a single
SQL statement with two CTEs (semantic top-20, lexical top-20) joined and
ordered by `final_score DESC LIMIT 5`.

**Rationale**:
- Hand-tunable against a ≥50-question eval set; the user's plan is to iterate
  the weights during dev/test (spec §"Build order"). RRF is parameter-free but
  harder to steer when a single sibling category (e.g. prices) under-ranks
  semantically.
- Weighted sum is trivially explainable in a retrieval_trace row — the per-turn
  score breakdown is just two numbers.

**Alternatives considered**:
- Reciprocal Rank Fusion (`1/(k+rank)` summed): mathematically cleaner at
  small-n; parked as R2-F1, a deferred survivor we can A/B against the eval
  set after v1 lands.
- Pure semantic: fails on exact-command / exact-price queries.
- Pure lexical: fails on paraphrases and follow-up rewrites.

**Constraints from decision**:
- `hybrid_search.py` computes both CTEs server-side in one round trip (not two
  queries) to stay inside the 80 ms SQL budget.
- `ts_rank_cd(content_tsv, plainto_tsquery('russian', $2))` uses Postgres's
  built-in `russian` text-search config (see R3).

---

## R3 — Russian text-search stemmer

**Decision**: Postgres built-in `russian` snowball stemmer (`to_tsvector('russian',
...)`, `plainto_tsquery('russian', ...)`).

**Rationale**:
- Ships with `postgres:18-alpine` — no image rebuild, no extension to manage.
- Good enough for v1 against a curated KB; the eval harness will measure recall
  and we can swap if numbers fall short.
- No new attack surface.

**Alternatives considered**:
- `postgrespro/hunspell_dicts` Russian Hunspell: materially better morphology
  coverage, but requires a custom Postgres image / extension build. Parked as
  R3-F1; revisit when `ts_rank_cd` recall on the eval set is the dominant
  remaining error bucket.

---

## R4 — pgvector index parameters

**Decision**: HNSW with `m = 16`, `ef_construction = 200`, `ef_search = 40`
(Postgres session default). Index on `embedding vector_cosine_ops`.

**Rationale**:
- pgvector's recommended defaults for <100k vectors per tenant. We have
  thousands per tenant, not millions.
- HNSW insert cost is acceptable because ingestion is out-of-band (not on the
  voice path); `ef_construction = 200` gives good recall without making
  ingestion painful.
- Cosine is the natural similarity for normalized sentence-transformer outputs.

**Alternatives considered**:
- IVFFlat: cheaper to build, worse recall at small n; requires retraining when
  the KB grows. Rejected.
- `m = 32`, `ef_construction = 400`: marginal recall gains, 2× index memory.
  Reserved for tenants measured to need it.

---

## R5 — FrameProcessor insertion point

**Decision**: Insert `KBRetrievalProcessor` between `user_transcript` and
`user_agg` in `services/agent/pipeline.py:210`. Use `LLMMessagesAppendFrame`
to inject enriched context into the existing `LLMContextAggregatorPair`.

**Rationale**:
- `user_transcript` (`UserTranscriptForwarder`) fires on finalized
  `TranscriptionFrame`; that is the earliest moment we have a complete caller
  utterance. `user_agg` is Pipecat's aggregator that assembles the LLM context.
  Inserting between them lets us enrich the LLM input without reimplementing
  aggregation or touching the LLM service.
- `LLMMessagesAppendFrame` is Pipecat-idiomatic and preserves the aggregator's
  turn-boundary accounting. Pushing a raw `LLMMessagesFrame` would bypass the
  aggregator and break `AssistantTranscriptForwarder`'s turn correlation.

**Alternatives considered**:
- Embed retrieval inside a custom `LLMService` subclass: violates
  single-responsibility; every LLM service (vLLM, Groq, OpenAI) would need a
  parallel implementation.
- Context aggregator hook: Pipecat API doesn't currently expose a clean hook;
  would require forking or monkey-patching.

**Constraints from decision**:
- The processor tags every outbound frame with a monotonic `turn_id` so
  barge-in (`UserStartedSpeaking` during TTS) discards stale retrieval
  results. `asyncio.to_thread` work is uncancellable (constitution Principle I
  / Silero learning), so we can't cancel — we can only discard.

---

## R6 — Rewriter / in-scope classifier

**Decision**: One LLM call via the existing OpenAI-compatible client
(`OpenAILLMService` env contract: `LLM_BASE_URL`, `LLM_API_KEY`,
`REWRITER_MODEL`). The call returns strict JSON
`{"query": str, "in_scope": bool}` via the server's JSON-mode / structured-
outputs flag. Token budget ≤200 output tokens.

**Rationale**:
- One call does both jobs (rewrite + classify). Two calls double latency; two
  bespoke models would double ops surface.
- `REWRITER_MODEL` is configurable so operators can route this call to a
  cheaper / faster model than the main chat model (vLLM can host multiple;
  Groq has Llama 3.1 8B which is fast enough).
- JSON mode is broadly supported (vLLM ≥0.4, Groq, OpenAI). If malformed,
  `FR-008` kicks in: fail closed → route to refusal path.

**Alternatives considered**:
- Rule-based scope classifier (regex + keyword lists): brittle on Russian
  morphology; rejected.
- Small fine-tuned classifier (e.g. distilbert-ru): adds a second model to
  deploy and drift-manage; value unclear vs a warm-cache LLM call. Parked as
  R6-F1.
- Sequential (classify first, rewrite only if in-scope): saves rewriter tokens
  on out-of-scope turns but adds a second round trip. Rejected for latency.

**Constraints from decision**:
- Conversation history passed to the rewriter is bounded to **last 3 user
  turns + 3 assistant turns** to keep prompt size stable and prevent
  long-session prompt drift.
- Rewriter prompt is versioned in `services/retriever/rewriter.py`; changes
  are tracked in `retrieval_traces.rewriter_version` to reproduce any trace.

---

## R7 — In-scope / out-of-scope routing

**Decision**: In-scope → run retrieval, inject grounded Russian system prompt
(with clearly delimited `Контекст:\n---\n{chunks}\n---`) + rewritten query +
bounded history. Out-of-scope → inject a separate refusal system prompt
(Russian: "Я помогаю только с вопросами по игре. Чем я могу помочь с игрой?")
with no KB context; the LLM is still used to produce a natural-sounding
refusal rather than a canned fixed string.

**Rationale**:
- Using the LLM on the refusal path (rather than a fixed string) gives
  equal-ish work to the retrieval path — see R8 timing parity.
- Dynamic refusal sounds less robotic and lets the LLM personalize ("Я не
  знаю про погоду, но могу помочь с квестами") without being able to leak
  KB content (the refusal prompt contains no KB).

**Alternatives considered**:
- Fixed canned TTS string for refusal: fastest, but the timing asymmetry
  leaks scope (SC-008 violation). Could be used if we also add a calibrated
  delay — but adding a delay to a "fast" path is ugly; easier to just run
  the LLM on both branches.

---

## R8 — Timing-channel parity

**Decision**: Both in-scope and out-of-scope branches run through the LLM
(see R7), so their end-to-end timing is dominated by LLM latency. The
retrieval SQL round trip on the in-scope branch (p50 ~40 ms) is the only
systematic asymmetry. We pad the out-of-scope branch with a synthetic delay
calibrated to each tenant's rolling p50 retrieval SQL latency, applied via
`asyncio.sleep()` between rewriter return and refusal prompt dispatch. Delta
is observed via `retrieval_traces.stage_timings` and the SC-008 probe test.

**Rationale**:
- Constitution Principle III explicitly requires equal-work branches.
- Rolling per-tenant calibration (not a fixed 50 ms) handles tenants with
  hot vs cold retriever caches, and handles the retriever-service-down
  degradation (pad matches current retrieval latency ≈ timeout value).

**Alternatives considered**:
- Always-run-retrieval-even-on-out-of-scope: wastes embedding + SQL work on
  every injection probe; attacker-driven DoS amplification.
- Fixed padding (e.g. 50 ms): doesn't adapt to load. Measured asymmetry
  drifts.

**Constraints from decision**:
- `hybrid_search.py` exports a `record_p50(tenant_id, latency_ms)` sidecar
  used by the refusal path to look up the current pad. In-memory per-tenant
  rolling window (60 s, capped). Isolated namespace — not shared with
  `apps/api/rate_limit.py` TTLCaches (`xff-spoof` learning).

---

## R9 — Tenant isolation

**Decision**: `tenant_id UUID NOT NULL` on `kb_entries`, `kb_chunks`,
`retrieval_traces`, and `synthetic_questions` (if adopted). Every retrieval
SQL statement has `WHERE tenant_id = $1` unconditionally. The retriever
service receives the tenant id on every `/retrieve` call (from the agent,
which has it on the session); the retriever never trusts an inferred tenant
from the query text.

**Rationale**:
- Constitution Principle IV: tenant isolation is a data invariant. A
  nullable tenant_id eventually gets defaulted to `NULL` by a buggy client
  and leaks cross-tenant. Non-null from day one.
- Same pattern as `sessions.tenant_id` in existing schema.

**Alternatives considered**:
- Row-level security (Postgres RLS) with per-tenant Postgres roles: more
  defense-in-depth, but adds connection-pool-per-tenant complexity that we
  don't need at this scale. Revisit if we ever run untrusted SQL on this DB.

---

## R10 — Ingestion triggering

**Decision**: CLI / scheduled job, not live. Operators run
`python -m retriever.ingest --tenant <slug> --source <path>` (or equivalent
cron entry). The command is idempotent: upsert by `(tenant_id, kb_entry_id,
section)`; content-hash skip on unchanged chunks to avoid re-embedding.

**Rationale**:
- Keeps ingestion out of the live voice path (FR-012).
- Matches operator mental model ("I edited the KB; I run ingest").
- Webhook-driven ingestion is a straightforward v1.1 on top of this (call the
  CLI from a webhook handler); not needed for v1 dev/test.

**Alternatives considered**:
- Live webhook ingestion from the source-of-truth KB: extra auth surface,
  extra failure mode (webhook delivery); deferred to v1.1.
- Continuous streaming ingestion: not justified at thousands-of-chunks scale.

---

## R11 — Synthetic-question seeding

**Decision**: Adopt the user's sketch — generate 3–5 synthetic questions per
chunk at ingestion time via a single LLM pass, embed each, store as
additional rows in `kb_chunks` with a flag `is_synthetic_question = true`
pointing back to the source chunk via `parent_chunk_id`. Only adopt if the
eval harness measures ≥5 percentage-point recall improvement over content-
only ingestion.

**Rationale**:
- Known technique for narrowing the semantic gap between how users phrase
  questions and how source KB phrases answers.
- Cheap at ingestion time (one LLM call per chunk); free at retrieval time
  (just more rows in the index).
- Gating on measured improvement prevents cargo-culting.

**Alternatives considered**:
- Storing synthetic questions in a separate `synthetic_questions` table: adds
  a join at retrieval time for 0 benefit. One table with a discriminator is
  simpler.
- No synthetic questions: fallback if eval says the feature doesn't help on
  this KB.

---

## R12 — Testcontainers Postgres + pgvector for tests

**Decision**: Use the `pgvector/pgvector:pg16` container image in
`testcontainers` for all integration tests that touch the schema. Run Alembic
migrations against the fresh container per-test-module. No mocking of
pgvector.

**Rationale**:
- Constitution Principle II: integration tests must hit a real database, not
  mocks. The `delete-sessions` incident is exactly the class of bug mocked
  tests miss.
- Containerized tests are slow (~5 s startup) but scoped to the retriever
  package; CI keeps them in a separate workflow step from the agent's
  per-commit lint.

**Alternatives considered**:
- Shared CI Postgres: state leakage across test runs; rejected.
- Ephemeral DB on localhost: same problem if developer runs tests twice.

---

## R13 — Pipecat version compatibility

**Decision**: Target Pipecat 0.0.108 (currently pinned in
`services/agent/pyproject.toml`). `FrameProcessor`, `TranscriptionFrame`, and
`LLMMessagesAppendFrame` are stable on this version.

**Rationale**: Upgrading Pipecat is out of scope for this feature. If a later
Pipecat version materially changes the `FrameProcessor` contract, it's a
separate refactor task.

**Alternatives considered**: Upgrade to latest Pipecat: rejected — separate
concern.

---

## R14 — Embedder deployment topology

**Decision**: BGE-M3 runs inside the `retriever` service process as a
preloaded singleton (mirror of `services/agent/models.py::load_gigaam`).
CPU by default; GPU opt-in via env var `EMBEDDER_DEVICE=cuda` when the host
has spare VRAM.

**Rationale**:
- One round trip to the retriever ( agent→retriever HTTP) is simpler than
  two (agent→retriever→embedder).
- BGE-M3 on CPU does a single query embedding in ~20–50 ms on modern x86,
  within budget.
- GPU upgrade path exists without architectural change.

**Alternatives considered**:
- Separate embedder microservice: extra hop, extra failure mode. Rejected.
- Shared GPU with GigaAM on the same host: the `hf_cache_retriever` named
  volume is already scoped per-service UID (`xff-spoof`-style isolation);
  the GPU is multi-tenant friendly if both processes opt in. Rejected for
  v1 because GigaAM's GPU VRAM headroom is the bottleneck on L4 / A10G; we
  won't share until measured.

---

## R15 — Observability format for `retrieval_traces`

**Decision**: JSONB `stage_timings` column + typed top-level columns for the
fields that are queried most often (tenant_id, session_id, turn_id,
in_scope, rewriter_version). Append-only; no UPDATE, no DELETE except via
tenant-wide retention job.

**Rationale**:
- Typed top-level columns for the 80% query path ("give me all out-of-scope
  turns for tenant X in the last hour").
- JSONB for the long tail (per-stage ms, per-chunk score, LLM token counts).
- Append-only matches the user's institutional learning from the
  `retrieval_traces` deferred survivor in the ideation doc.

**Alternatives considered**:
- Pure JSONB blob: no typed indexes; slower queries.
- Full normalization (one row per chunk-hit): row count explodes.

---

**Summary of deferred survivors** (preserved; revisit post-v1):

- R2-F1: RRF fusion A/B vs weighted sum on eval set.
- R3-F1: Hunspell Russian stemmer.
- R6-F1: Fine-tuned small scope classifier.
- R10-F1: Webhook-driven ingestion.
- R14-F1: GPU-shared embedder with GigaAM.
- Two-phase speculative retrieval (pre-EOS + between-turn): deferred. Measured
  prior art (Stream RAG, VoiceAgentRAG) suggests 20 % latency reduction;
  revisit once v1 baseline is measured.

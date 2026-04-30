---
date: 2026-04-23
topic: helper-guide-npc-rag
focus: KB-grounded retrieval for Helper/Guide NPCs — Russian-only, conversations-only MVP, retriever service :8002 + pgvector + Russian tsvector hybrid + query rewriter + Pipecat FrameProcessor between STT and LLM.
mode: repo-grounded
outcome: user elected to proceed directly to implementation planning with the originally-sketched architecture (see "Architecture to Implement" below). Survivors below are preserved for reference and for future iteration once the MVP lands.
---

# Ideation: KB-grounded Helper/Guide NPC RAG (Russian MVP)

## Grounding Context

### Codebase Context

- Pipecat 0.0.108 agent (Python 3.12, CUDA), FastAPI (3.14, uv), React SPA, PG18. Pipeline: Silero VAD → GigaAM-v3 STT → `LLMContextAggregatorPair` → LLM (OpenAILLMService → vLLM/Groq/OpenAI) → Piper TTS.
- FrameProcessor precedent: `services/agent/transcript.py` (UserTranscriptForwarder, AssistantTranscriptForwarder, ErrorFrameForwarder). Insertion point for KB retrieval: between GigaAM STT output and `user_agg` — inject `LLMMessagesAppendFrame` before the LLM aggregator finalizes.
- PG schema: `tenants`, `api_keys`, `sessions`, `session_transcripts`. `sessions.npc_id` (nullable TEXT), `persona` (JSONB), `context` (JSONB, unused) already present. `PerSessionOverrides` already plumbs npc/persona/context per session.
- No pgvector extension today. Alembic configured (`apps/api/alembic.ini`), migrations 0001-0003. No NPCs table.
- Russian routing: `services/agent/lang.py` rejects CJK → Russian. GigaAM hallucination filter drops <300ms / <2 tokens.
- Latency marker: `~700ms after end-of-speech on RTX 5080 / L4`. `_USER_TRANSCRIPT_DEBOUNCE_SECS=1.0`. GPU saturated on L4.

### Past Learnings (docs/solutions/)

- **xff-spoof**: isolate per-namespace TTLCaches (tenant / ip / query_hash); `ipaddress.ip_network` with `trusted_proxy_cidrs`; `Retry-After: 60` centrally in `apps/api/errors.py`; equal-work branches to avoid timing-channel leaks of KB membership.
- **DELETE /v1/sessions missing audio_seconds**: any terminal-row writer must cover DELETE + LiveKit `room_finished` webhook; SQL-layer idempotent UPDATE beats Python `if` guards; happy-path tests assert every writable column.
- **Silero spike / Pipecat FrameProcessor**: wrap blocking calls in `asyncio.to_thread` (uncancellable — sets barge-in floor = retrieval RTT); `ErrorFrame` redaction on failures; preload heavy models via singleton (mirror `services/agent/models.py::load_gigaam`); HF cache named volume (`hf_cache_agent` precedent); add `python -c "import X"` import smoke to `Dockerfile.base`; healthcheck `start_period: 300s`.

### External Context

- **VoiceAgentRAG (Salesforce, Mar 2026)**: dual-agent speculative prefetch during previous TTS playback; index by document embeddings, not prediction queries; 75% hit rate, 316× speedup, 0.40 similarity threshold, 300s TTL. [arxiv 2603.02206, open-sourced]
- **Stream RAG (Oct 2025)**: parallel retrieval during ASR before EOS → 20% additional latency reduction. [arxiv 2510.02044]
- **Russian embedders**: BGE-M3 (no prefix discipline, 8192-token, ColBERT available) is the strongest CPU retrieval choice for Russian. GigaEmbeddings SOTA on ruMTEB but GPU-only, not materially better on retrieval. multilingual-E5-large requires `"query:"/"passage:"` prefixes and lags BGE-M3 on retrieval.
- **Postgres Russian FTS**: default snowball stemmer weak on morphology; `postgrespro/hunspell_dicts` Russian Hunspell materially improves recall.
- **pgvector <10k docs**: HNSW safer than IVFFlat (`m=16, ef_construction=200`). RRF > weighted sum for hybrid at small scale.
- **Voice safety**: prefer async post-gen filter over blocking pre-LLM gate for latency; NeMo Guardrails / Llama Guard 3 viable for input classification.
- **OWASP LLM #1**: prompt injection / roleplay framing is the dominant jailbreak — mitigate with hard in-scope classifier + system-prompt persona lock.
- **Alternatives to pgvector**: no benefit at this scale; stick with pgvector in existing Postgres.

## Architecture to Implement (Chosen)

Per the user's original focus, the MVP implements the following architecture exactly as described, without further variants or simplifications:

1. **Retriever service on `:8002`** — standalone FastAPI app. Postgres + pgvector (HNSW) + Russian tsvector hybrid search.
2. **Schema**: one `kb_chunks` table with `(id, kb_entry_id, title, section, content, content_tsv, embedding vector(1024), metadata JSONB, version, updated_at)`; indexes: `hnsw (embedding vector_cosine_ops)`, `gin (content_tsv)`, `btree (kb_entry_id)`.
3. **Hybrid retrieval query**: parallel semantic (cosine) + lexical (`ts_rank_cd` with `plainto_tsquery('russian', ...)`); combine as weighted sum (0.7 semantic + 0.3 lexical) for MVP tunability.
4. **Query rewriter + in-scope classifier** — one LLM call, structured JSON output `{query: str, in_scope: bool}`. In-scope=false → canned out-of-scope response or refusal-prompt path; in-scope=true → retrieval.
5. **Pipecat FrameProcessor (`KBRetrievalProcessor`)** — intercepts `TranscriptionFrame` between STT and LLM, calls `POST :8002/retrieve`, injects enriched system prompt with retrieved chunks + rewritten query into `LLMMessagesFrame`.
6. **System prompt** — grounded Russian helper-guide template ("Отвечай на вопрос игрока, используя ТОЛЬКО информацию из контекста ниже…").
7. **Ingestion pipeline** — split KB entries by H2/H3; embed `title + "\n" + content` with `multilingual-e5-large` or `bge-m3`; one-shot metadata extraction (prices/commands/level reqs → JSONB); generate 3-5 synthetic questions per chunk; upsert by `(kb_entry_id, section)`. Trigger: webhook on edit, or scheduled full-sync.
8. **Scope constraints**: Helper/Guide NPCs only. Russian only. Conversations only — no map geotagging, no callbacks to the game client.

Build order (per user's original sketch): (1) Ingestion script; (2) Retriever service + hybrid endpoint + 50 hand-written Russian eval queries; (3) Query rewriter; (4) Pipecat FrameProcessor; (5) System prompt tuning; (6) End-to-end voice test.

Latency target: <800ms first-audio-out; ~200-500ms retrieval overhead acceptable for MVP.

## Survivor Ideas (Reference Only — Not Scoped Into MVP)

These ideas emerged during ideation. Per user direction they are **not** part of the MVP scope, but are preserved as a working backlog to reconsider once the MVP lands.

### 1. Inline-KB MVP in the system prompt
**Description:** For NPCs whose KB fits in ~4k tokens, skip the retriever service entirely and inline the full KB into the Pipecat system prompt at dispatch; rely on vLLM `--enable-prefix-caching`.
**Rationale:** Simplest possible MVP; graduates per-NPC on measured miss.
**Downsides:** Requires prefix caching; doesn't generalize to large KBs.
**Confidence:** 80% · **Complexity:** Low · **Status:** Rejected by user for MVP (user chose the retriever path)

### 2. Two-phase speculative retrieval (pre-EOS + between-turn)
**Description:** Phase A — fire async retrieval on `InterimTranscriptionFrame` partials, await on EOS; Phase B — speculatively prefetch 3-5 likely-next chunks during previous TTS playback, cache per `session_id`. Tag all work with `turn_id` so barge-in discards stale results.
**Rationale:** Directly attacks the <800ms budget with measured 2025-2026 prior art (Stream RAG 20%, VoiceAgentRAG 316× / 75% hit rate).
**Downsides:** Two caches, two consistency problems; key Phase B on document embeddings, not predictions.
**Confidence:** 75% · **Complexity:** Medium · **Status:** Unexplored — strong candidate for v2 once MVP baseline is measured

### 3. `retrieval_traces` + `rag_configs` as compounding log substrate
**Description:** Append-only trace row per turn (raw transcript → rewritten query → classification → retrieved ids → scores → chunks shown → final answer → latency breakdown → `config_id`). `rag_configs` table with `prompt_hash`, `retrieval_params`, `active_weight` for weighted A/B traffic.
**Rationale:** One primitive pays six ways — forensics UI, eval replay, SFT dataset, A/B analytics, cache-warm source, regression fixtures.
**Downsides:** JSONB drift temptation; ship log first, UI later.
**Confidence:** 90% · **Complexity:** Low · **Status:** Unexplored — highly recommended to add during or immediately after MVP

### 4. Russian-first retrieval stack — Hunspell tsvector + BGE-M3 + RRF
**Description:** Swap default Postgres snowball stemmer for `postgrespro/hunspell_dicts` Russian Hunspell; use BGE-M3 as embedder (no prefix discipline, ColBERT available); fuse with Reciprocal Rank Fusion instead of weighted sum at small scale.
**Rationale:** Hunspell materially outperforms snowball on Russian morphology; BGE-M3 is 2026 best Russian CPU retriever; RRF is parameter-free.
**Downsides:** Hunspell Postgres image step; BGE-M3 is 2GB (HF cache volume, never bake).
**Confidence:** 85% · **Complexity:** Medium · **Status:** Unexplored — variant of chosen architecture; worth considering as specific model/config decision within the chosen path

### 5. Voice-first UX trio — FAQ-pair unit + scope ladder + silence-gate refusal
**Description:** Unit of retrieval is `(question, answer_ru, variants[])`; scope classifier emits 5-rung ladder (`DOMAIN_HIT | TOPIC_HIT | FUZZY | CLARIFY | REFUSE`); refusal is a silence-gate (empty chunks + short system nudge), not a canned TTS interruption.
**Rationale:** Question-to-question embedding is an easier retrieval job; ladder converts off-topic failure to guided on-topic; silence-gate eliminates jarring-interruption bugs.
**Confidence:** 75% · **Complexity:** Medium · **Status:** Unexplored

### 6. Unified `kb_units` schema with `unit_type` + structured hits + shared scope tuple
**Description:** Generalize the `kb_chunks` schema to `kb_units(unit_type, payload JSONB, tenant_id, npc_scope[], visibility, provenance, content_sha256, valid_from/to, ...)`. Retriever returns typed records; scope tuple reused for future `memory_atoms`.
**Rationale:** Every future content source becomes an adapter; typed payload enables deterministic answers for structured queries; shared scope primitive unifies with user-memory ideation.
**Confidence:** 80% · **Complexity:** Medium · **Status:** Unexplored — schema-shape decision worth revisiting before the migration lands if generalization is cheap

### 7. Versioned ingestion contract + diff-aware re-embedding + atomic webhook cutover
**Description:** `KbIngestionRecord v1` JSON format; `embedding_cache(content_sha256, model_id)` → skip embed on hit; shadow rows with atomic `UPDATE kb_pointer SET version=$new` cutover; synthetic-question generator seeds cache + eval fixtures in the same pass.
**Rationale:** Swapping embedders is free on unchanged rows; atomic cutover kills stale-KB class; one LLM pass pays five ways.
**Confidence:** 80% · **Complexity:** Medium · **Status:** Unexplored — valuable refinement to the user's sketched ingestion pipeline; consider adopting the content-hash cache + shadow-row cutover during implementation

## Rejection Summary

The full rejection table of 38 ideas (museum game-state proximity, chess opening-book WAV cache, ATC readback, stenographer chords, self-curating KB, etc.) is preserved at `/var/folders/xv/.../compound-engineering/ce-ideate/b7c3d4e2/survivors.md` — scratch only, not copied in here to keep the ideation doc focused.

## Next Step

Hand off to `/ce-plan` using **"Architecture to Implement"** above as the source of truth, **not** the survivor variants. Per user direction (2026-04-23), implement exactly the architecture described in the original focus hint — no additional details, no simplifications, no variants.

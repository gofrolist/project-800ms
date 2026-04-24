# Implementation Plan: Helper/Guide NPC — KB-Grounded Answers

**Branch**: `002-helper-guide-npc` | **Date**: 2026-04-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-helper-guide-npc/spec.md`

## Summary

Add a KB-grounded preprocessing step on the LLM input path of the existing Pipecat
voice agent so the Helper/Guide NPC answers Russian game-help questions from a
curated knowledge base and refuses off-topic / roleplay / prompt-injection probes
without leaking scope via timing. The change spans four units:

1. A new internal service `services/retriever/` (FastAPI on `:8002`) exposing
   `POST /retrieve`: hybrid semantic + lexical search over `kb_chunks` in Postgres
   (pgvector HNSW + Russian `tsvector` GIN), plus a single LLM call that both
   rewrites the raw transcript into a standalone Russian query and classifies the
   turn as `in_scope` / `out_of_scope`.
2. A new Postgres migration adding `vector` + `pg_trgm` extensions and the
   `kb_chunks` table (tenant-scoped) plus a `retrieval_traces` table (append-only
   per-turn forensics substrate).
3. A new Pipecat `FrameProcessor` (`KBRetrievalProcessor`) inserted between
   `user_transcript` and `user_agg` at `services/agent/pipeline.py:210`. On every
   final `TranscriptionFrame` it calls the retriever, enriches the LLM context
   with retrieved chunks (in-scope) or routes to the refusal system prompt
   (out-of-scope), and discards work when the caller barges in.
4. An out-of-band ingestion CLI / job (`services/retriever/ingest.py`) that reads
   the tenant's KB source, splits by heading, embeds `title + "\n" + content` with
   BGE-M3, extracts structured metadata, optionally seeds synthetic questions, and
   upserts idempotently by `(tenant_id, kb_entry_id, section)`.

End-to-end latency stays inside the constitution's 800 ms budget: retrieval p95
≤500 ms (rewrite 150–300, embed 20–50, hybrid SQL 20–80, margin for HTTP round
trip). No speculative prefetch, no between-turn caching in v1 — those are deferred.

## Technical Context

**Language/Version**:
- `services/retriever/` — Python 3.12 (match the agent's CUDA base; BGE-M3 runs
  on CPU or GPU).
- `services/agent/` (existing) — Python 3.12, adds one new `FrameProcessor`.
- `apps/api/` (existing) — Python 3.14, adds one Alembic migration.

**Primary Dependencies**:
- FastAPI, Pydantic v2, Uvicorn (retriever HTTP surface).
- asyncpg + pgvector-python (async Postgres with typed `vector` columns).
- SQLAlchemy 2.x async (parity with `apps/api`) for migrations and query
  construction.
- `sentence-transformers` + `FlagEmbedding` (BGE-M3 embedder).
- `httpx` for internal agent→retriever calls and retriever→LLM rewriter calls.
- Pipecat 0.0.108 (already pinned in `services/agent/pyproject.toml`) —
  `FrameProcessor`, `TranscriptionFrame`, `LLMMessagesAppendFrame`.
- OpenAI-compatible client (reuses existing vLLM / Groq / OpenAI endpoint from
  `services/agent`).

**Storage**:
- PostgreSQL 18 via `pgvector/pgvector:pg18` in `infra/docker-compose.yml`
  (swapped from stock `postgres:18-alpine` so the `vector` extension is
  available for migration 0004_kb_chunks).
- New extensions: `vector` (pgvector) and `pg_trgm`.
- New tables: `kb_entries`, `kb_chunks`, `retrieval_traces`,
  `synthetic_questions` (optional, gated on eval-set measurement).
- HF model cache on a new named volume `hf_cache_retriever` — model weights
  never baked into the image (constitution Principle V).

**Testing**:
- `pytest` + `pytest-asyncio` + `httpx.AsyncClient` (parity with `apps/api`).
- `testcontainers[postgres]` spinning up a real Postgres 18 + pgvector instance
  for migrations, upsert idempotency, hybrid retrieval, and tenant-isolation
  tests. The constitution forbids mocked integration tests on write paths.
- A `services/retriever/evals/` directory with a curated Russian eval set
  (`≥50` in-scope questions, `≥20` out-of-scope / injection probes) checked in
  as YAML + expected top-3 chunk ids for regression gating.

**Target Platform**: Linux server (Docker Compose on constitution reference
hardware — RTX 5080 / L4 / A10G / L40S, 16 GB+ VRAM, NVIDIA Container Toolkit).

**Project Type**: Multi-service web-backend (voice platform + new internal
microservice). Additive — spec 001's session/token-minting surface is unchanged.

**Performance Goals**:
- End-to-end first-audio-out p95 ≤ 800 ms (inherited from constitution).
- Retrieval-stage overhead p95 ≤ 500 ms (FR-019, SC-004).
- Rewrite+classify p95 ≤ 300 ms; embed p95 ≤ 50 ms; hybrid SQL p95 ≤ 80 ms.
- In-scope vs out-of-scope p95 end-to-end latency diverges by ≤ 50 ms (SC-008).

**Constraints**:
- Retrieval must be *discardable* on barge-in (FR-018, SC-010). `asyncio.to_thread`
  work is uncancellable, so the retrieval coroutine is tagged with `turn_id`; its
  result is discarded if the turn id no longer matches the current turn by the
  time it resolves.
- Equal-work branches (FR-011, SC-008): the refusal path performs a synthetic
  delay calibrated to p50 retrieval latency on the same tenant so scope does not
  leak through timing.
- Tenant-scoped KB: `kb_chunks.tenant_id NOT NULL` on day one (constitution
  Principle IV). Retrieval SQL always carries `WHERE tenant_id = $session.tenant`.
- Model weights never baked into images; `hf_cache_retriever` named volume
  persists BGE-M3 across restarts.

**Scale/Scope**:
- Small-to-medium KB per tenant: thousands of `kb_chunks`, not millions. HNSW
  `m=16, ef_construction=200` is appropriate; IVFFlat is rejected.
- `≤100` concurrent retrieval calls in early-stage dev/test; single retriever
  replica sufficient. Horizontal scaling deferred.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Principles from `.specify/memory/constitution.md` (v1.0.0). Status per principle:

| # | Principle | Status | How this plan satisfies it |
|---|-----------|--------|---------------------------|
| I | Latency Budget Is Law | ✅ Pass | Retrieval sub-budget explicit (p95 ≤500 ms) and counts against the 800 ms total; rewrite+embed can run partly in parallel (Phase 0 R6); barge-in floor raised by at most the HTTP RTT to the retriever, measured in `quickstart.md`. Before/after latency measurement is mandatory on the PR (constitution, Principle I) — included in test plan below. |
| II | Test-First Discipline | ✅ Pass | Every new write path (ingestion upsert, retrieval_traces insert) has an integration test against a real Postgres+pgvector container that asserts *every* writable column (`xff-spoof` and `delete-sessions` learnings). Eval harness is its own test suite; retrieval recall and groundedness are measured CI-gated. 80 % coverage floor on new modules. |
| III | Grounded & Bounded NPC Behavior | ✅ Pass | This feature *is* Principle III. In-scope classifier mandatory (FR-006); persona-lock system prompt (FR-010); timing-parity on in-scope vs out-of-scope (FR-011, SC-008); KB-only answers with "I don't know" fallback (FR-003). Scope boundaries explicit in spec: Helper/Guide + Russian + conversations-only. |
| IV | Security & Multi-Tenant Boundaries | ✅ Pass | `kb_chunks.tenant_id NOT NULL` from first migration; retriever SQL always filters by tenant; isolated cache namespaces (`_kb_query_cache` not shared with `_tenant_buckets` / `_ip_buckets` from `apps/api/rate_limit.py`); retriever bound to private internal network (never exposed publicly); rewriter prompts scrubbed of persona content on egress to external LLMs. No new secrets — reuses existing `LLM_BASE_URL` / `LLM_API_KEY`. |
| V | Observable, Reproducible Pipelines | ✅ Pass | `retrieval_traces` is the observability substrate (FR-020, SC-007); structured loguru logs with lazy placeholders on every stage; errors via `ErrorFrame` (FR-022, per `transcript.py` precedent); BGE-M3 preloaded via singleton at retriever boot (mirrors `services/agent/models.py::load_gigaam`); `Dockerfile.base` + thin app image with `python -c "import pgvector; import sentence_transformers; import FlagEmbedding"` smoke layer; `hf_cache_retriever` named volume; `healthcheck start_period: 300s`. |

**Gate result**: All pass. No Complexity Tracking entries required.

## Project Structure

### Documentation (this feature)

```text
specs/002-helper-guide-npc/
├── plan.md              # This file
├── research.md          # Phase 0 output — resolved technical decisions
├── data-model.md        # Phase 1 output — entities, schema, state transitions
├── quickstart.md        # Phase 1 output — ingest sample KB, run eval, observe traces
├── contracts/
│   ├── retrieve.openapi.yaml   # POST /retrieve contract
│   └── ingest.cli.md           # Ingestion CLI contract
├── checklists/
│   └── requirements.md  # Already produced by /speckit-specify
└── tasks.md             # Produced by /speckit-tasks (NOT by this command)
```

### Source Code (repository root)

```text
services/
├── agent/                              # existing — modified
│   ├── pipeline.py                     # insert KBRetrievalProcessor between
│   │                                   # user_transcript and user_agg (line 210)
│   ├── kb_retrieval.py                 # NEW — FrameProcessor impl
│   ├── overrides.py                    # unchanged (persona/npc_id already plumbed)
│   └── tests/
│       ├── test_kb_retrieval.py        # NEW — unit: frame handling, turn-id tagging
│       └── test_pipeline_insertion.py  # NEW — assert processor ordering
│
└── retriever/                          # NEW — standalone FastAPI on :8002
    ├── Dockerfile.base                 # CUDA/CPU + Python deps + BGE-M3 import smoke
    ├── Dockerfile                      # thin FROM ${BASE_IMAGE}; COPY .
    ├── pyproject.toml                  # uv-managed; asyncpg, pgvector, sentence-transformers
    ├── main.py                         # FastAPI app; /retrieve; /healthz; /ready
    ├── config.py                       # Pydantic Settings (LLM_BASE_URL, DB_URL, ...)
    ├── embedder.py                     # BGE-M3 singleton; preloaded at boot
    ├── rewriter.py                     # single-LLM-call rewrite + classify, JSON mode
    ├── hybrid_search.py                # weighted-sum fusion over pgvector + tsvector
    ├── traces.py                       # retrieval_traces writer
    ├── ingest.py                       # ingestion CLI entry; upsert idempotent
    ├── ingest_pipeline.py              # chunking, metadata extraction, synth questions
    ├── evals/
    │   ├── inscope_ru.yaml             # ≥50 curated questions + expected chunk ids
    │   └── probes_ru.yaml              # ≥20 off-topic / injection probes
    └── tests/
        ├── conftest.py                 # testcontainers Postgres + pgvector fixture
        ├── test_hybrid_search.py
        ├── test_rewriter.py
        ├── test_ingest_idempotency.py
        ├── test_tenant_isolation.py
        ├── test_retrieve_endpoint.py
        └── test_eval_harness.py        # gates recall + groundedness numeric SCs

apps/
└── api/                                # existing — modified
    └── migrations/versions/
        └── 0004_kb_chunks.py           # NEW — vector/pg_trgm ext + kb_entries +
                                        #        kb_chunks + retrieval_traces +
                                        #        (optional) synthetic_questions

infra/
├── docker-compose.yml                  # existing — modified: add `retriever` service
│                                        # + hf_cache_retriever volume
└── .env.example                        # existing — modified: RETRIEVER_URL, REWRITER_MODEL
```

**Structure Decision**: Reuse existing multi-service layout (`apps/` + `services/` +
`infra/`). Adding one new service under `services/retriever/` mirrors the existing
`services/agent/` pattern — CUDA-capable base image split, thin app layer, named
HF cache volume. No changes to `apps/api/` beyond a single Alembic migration. No
changes to `apps/web/`. This is additive; spec 001's contracts are unchanged
(FR-017).

## Complexity Tracking

No constitution violations. This section is intentionally empty.

---

**Phase 0** (research) and **Phase 1** (design) outputs live in sibling files:
- [research.md](./research.md) — resolved technical decisions + alternatives
- [data-model.md](./data-model.md) — schema, state transitions
- [contracts/](./contracts/) — retrieve API + ingest CLI
- [quickstart.md](./quickstart.md) — end-to-end bring-up for dev

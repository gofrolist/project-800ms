# Phase 1 — Data Model: Helper/Guide NPC KB-Grounded Answers

Source of truth: `apps/api/migrations/versions/0004_kb_chunks.py` (new migration,
to be authored during `/speckit-implement`). This document describes **what** the
schema must be and the rules that govern it; the migration is the **how**.

All entities are tenant-scoped. Every query MUST carry `WHERE tenant_id = $1`
(constitution Principle IV; research R9).

---

## Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- BM25-ish fallback
-- pgcrypto already enabled by migration 0001_init
```

---

## Entities

### `kb_entries`

The source unit of curated knowledge for a tenant. A KB entry corresponds to one
article / page / chapter in the tenant's source material and decomposes into one
or more `kb_chunks` at ingestion time.

| Field            | Type         | Null | Notes                                          |
|------------------|--------------|------|------------------------------------------------|
| id               | UUID         | no   | `gen_random_uuid()` default; primary key       |
| tenant_id        | UUID         | no   | FK → `tenants.id` ON DELETE CASCADE            |
| kb_entry_key     | TEXT         | no   | Stable external id from the tenant's source KB |
| title            | TEXT         | no   | Human-readable, used in chunk title composition |
| source_uri       | TEXT         | yes  | URL or path to the origin document             |
| content_sha256   | TEXT         | no   | Hash of the full source content; drives re-embed skip |
| ingested_at      | TIMESTAMPTZ  | no   | `now()` default                                |
| updated_at       | TIMESTAMPTZ  | no   | `now()` default; updated on re-ingest          |

**Uniqueness**: `UNIQUE(tenant_id, kb_entry_key)` — the tenant's own id for the
entry is unique within that tenant only.

**Index**: `(tenant_id)` btree.

---

### `kb_chunks`

The retrievable unit. A chunk is a (section-sized) piece of a `kb_entry` with an
embedding and a `tsvector`. Chunks are the rows the retriever scans.

| Field                | Type          | Null | Notes                                                |
|----------------------|---------------|------|------------------------------------------------------|
| id                   | BIGSERIAL     | no   | primary key (sequential, smaller than UUID for HNSW) |
| tenant_id            | UUID          | no   | FK → `tenants.id` ON DELETE CASCADE                  |
| kb_entry_id          | UUID          | no   | FK → `kb_entries.id` ON DELETE CASCADE               |
| section              | TEXT          | yes  | e.g. "Стоимость", "Как получить"; NULL for unsectioned entries |
| title                | TEXT          | no   | Composed: `entry.title` + " / " + `section` (or just entry.title) |
| content              | TEXT          | no   | Russian chunk text                                   |
| content_tsv          | TSVECTOR      | no   | `GENERATED ALWAYS AS (to_tsvector('russian', title \|\| ' ' \|\| content)) STORED` |
| content_sha256       | TEXT          | no   | Hash of `content`; drives re-embed skip               |
| embedding            | VECTOR(1024)  | no   | BGE-M3 output (research R1)                          |
| metadata             | JSONB         | no   | `'{}'` default; prices, commands, level reqs extracted by ingestion LLM pass |
| is_synthetic_question| BOOLEAN       | no   | `false` default; `true` for generated questions (R11) |
| parent_chunk_id      | BIGINT        | yes  | FK → `kb_chunks.id` ON DELETE CASCADE; set only when `is_synthetic_question = true` |
| version              | INT           | no   | `1` default; bumped on content change                |
| ingested_at          | TIMESTAMPTZ   | no   | `now()` default                                      |
| updated_at           | TIMESTAMPTZ   | no   | `now()` default; updated on re-ingest                |

**Uniqueness**: `UNIQUE(tenant_id, kb_entry_id, section, is_synthetic_question, parent_chunk_id)`.
Composite uniqueness gives us idempotent upsert by
`(tenant_id, kb_entry_id, section)` for content chunks, and allows multiple
synthetic questions to attach to the same parent without collision.

**Indexes**:
- `hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)` —
  required for the semantic CTE.
- `gin (content_tsv)` — required for the lexical CTE.
- `(tenant_id, kb_entry_id)` btree — entry-scoped lookups and bulk re-ingest.
- `(tenant_id)` btree — tenant-scoped scans if HNSW is bypassed.

**Validation rules** (enforced in ingestion code, documented here):
- If `is_synthetic_question = true`, then `parent_chunk_id` MUST be non-null and
  point to a chunk with `is_synthetic_question = false`. Only one level of
  nesting.
- `embedding` dimensionality MUST be exactly 1024 (BGE-M3). Schema enforces via
  `vector(1024)`.
- `content` MUST NOT be empty.
- `metadata` values MUST be JSON-serializable scalars, arrays, or simple maps —
  no deeply nested structures.

**State transitions** (per content chunk identity = `(tenant_id, kb_entry_id, section)`):
- **ingested**: row exists with current `version`.
- **updated**: re-ingest detected content_sha256 change → same row,
  `content_sha256` / `content` / `embedding` updated, `version += 1`,
  `updated_at = now()`. Associated synthetic questions are also re-generated
  (DELETE + INSERT their rows in the same transaction).
- **orphaned**: the source `kb_entry_key` was removed from ingestion input. The
  ingestion tool marks these for deletion (see R10). Hard DELETE cascades to
  synthetic questions via `parent_chunk_id` FK.

---

### `retrieval_traces`

Per-turn immutable forensics record (research R15). Drives FR-020 / FR-021 /
SC-007.

| Field              | Type                      | Null | Notes                                        |
|--------------------|---------------------------|------|----------------------------------------------|
| id                 | UUID                      | no   | `gen_random_uuid()` default; primary key     |
| tenant_id          | UUID                      | no   | FK → `tenants.id` ON DELETE CASCADE          |
| session_id         | UUID                      | no   | FK → `sessions.id` ON DELETE CASCADE         |
| turn_id            | TEXT                      | no   | Monotonic within a session (agent assigns)    |
| created_at         | TIMESTAMPTZ               | no   | `now()` default                              |
| raw_transcript     | TEXT                      | no   | Whatever `TranscriptionFrame` carried         |
| rewritten_query    | TEXT                      | yes  | NULL when rewriter failed / malformed          |
| in_scope           | BOOLEAN                   | yes  | NULL when rewriter failed (fail-closed → refusal) |
| rewriter_version   | TEXT                      | no   | e.g. `"rewriter-v3-2026-04-23"`              |
| retrieved_chunks   | JSONB                     | no   | `'[]'` default; array of `{chunk_id, score, fusion_components: {semantic, lexical}}` |
| stage_timings_ms   | JSONB                     | no   | `'{}'` default; `{rewrite: int, embed: int, sql: int, llm: int, tts: int, total: int}` |
| final_reply_text   | TEXT                      | yes  | NULL when the turn errored before LLM        |
| error_class        | TEXT                      | yes  | e.g. `"retriever_timeout"`, `"malformed_rewriter_json"`; NULL on success |

**Uniqueness**: `UNIQUE(session_id, turn_id)`.

**Indexes**:
- `(tenant_id, created_at DESC)` btree — "what did tenant X do in the last hour".
- `(session_id, turn_id)` btree — trace lookup from a session.
- `(in_scope) WHERE in_scope = false` partial btree — refusal forensics.

**State transitions**: None. Rows are immutable once written. Retention is managed
by a separate tenant-wide retention job (out of scope for v1; default: keep all).

**Validation rules**:
- `retrieved_chunks` is `[]` when `in_scope = false` or when the turn errored.
- `stage_timings_ms.total` MUST equal the sum of the other stage timings within
  ±1 ms (test assertion; prevents silent timing drift).

---

## Relationships

```
tenants (existing, spec 001)
  ├── 1:N → kb_entries
  │           └── 1:N → kb_chunks (content)
  │                     └── 1:N → kb_chunks (synthetic questions; parent_chunk_id FK)
  ├── 1:N → sessions (existing, spec 001)
  │           └── 1:N → retrieval_traces
  └── 1:N → retrieval_traces
            (denormalized tenant_id for direct "recent traces by tenant" query)
```

All FK chains lead to `tenants`. `ON DELETE CASCADE` everywhere that a tenant
might leave the system — removing a tenant is total.

---

## Migration order

A single migration file: `apps/api/migrations/versions/0004_kb_chunks.py`
(revision `0004_kb_chunks`, `down_revision = "0003_transcripts"`).

```text
upgrade():
  1. CREATE EXTENSION IF NOT EXISTS vector;
  2. CREATE EXTENSION IF NOT EXISTS pg_trgm;
  3. CREATE TABLE kb_entries (...);
  4. CREATE TABLE kb_chunks (...);            -- includes tsvector GENERATED
  5. CREATE INDEX kb_chunks_embedding_idx ...;
  6. CREATE INDEX kb_chunks_tsv_idx ...;
  7. CREATE INDEX kb_chunks_tenant_entry_idx ...;
  8. CREATE INDEX kb_chunks_tenant_idx ...;
  9. CREATE TABLE retrieval_traces (...);
 10. CREATE INDEX retrieval_traces_tenant_created_idx ...;
 11. CREATE INDEX retrieval_traces_session_turn_idx ...;
 12. CREATE INDEX retrieval_traces_refusals_idx ...;  -- partial

downgrade():
  reverse order; DROP TABLEs; DROP EXTENSIONs only if no other migration depends.
```

**Safety notes** (for the migration reviewer):
- `vector` and `pg_trgm` are standard, non-destructive. Safe on a running system.
- No existing table is altered. No backfill needed.
- Creating the HNSW index is fast on an empty table; on a large table it's
  O(N log N). We're creating before any data exists, so the cost is zero.

---

## Data ownership & lifecycle

- **KB entries and chunks**: owned by the tenant operator. Created and updated
  via the ingestion CLI (`services/retriever/ingest.py`). Never mutated by the
  live voice path.
- **Retrieval traces**: owned by the platform. Written by the retriever service
  on every `/retrieve` call. Never mutated after write. Retention policy is
  platform-level (TBD — out of v1 scope; v1 keeps everything).
- **Synthetic questions**: owned by the ingestion pipeline. Regenerated whenever
  their parent chunk's `content_sha256` changes. Deleted with their parent.

---

## What is NOT in this data model

The following were considered and deferred (research R-deferred list):

- `rag_configs` table for A/B retrieval params: parked; v1 uses one global
  config.
- `embedding_cache(content_sha256, model_id)` table for cross-re-embed sharing:
  parked; v1 skips re-embed via `content_sha256` match in-place.
- User memory / per-player long-term context: out of spec 002; a future spec.

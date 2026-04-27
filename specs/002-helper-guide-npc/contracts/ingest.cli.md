# Ingestion CLI Contract

Command: `python -m retriever.ingest` (inside `services/retriever/`).
Alternatively `uv run python -m retriever.ingest` from the service root.

Not exposed over HTTP in v1 (research R10). Run manually or from a scheduled
job on the retriever host.

---

## Synopsis

```
python -m retriever.ingest \
  --tenant <tenant-slug> \
  --source <path-or-url> \
  [--mode {full,incremental}] \
  [--dry-run] \
  [--skip-synthetic-questions] \
  [--verbose]
```

---

## Arguments

| Arg                         | Required | Default        | Notes |
|-----------------------------|----------|----------------|-------|
| `--tenant <slug>`           | yes      | —              | Must match `tenants.slug`. The CLI rejects unknown slugs. |
| `--source <path-or-url>`    | yes      | —              | Path to a directory of Markdown/JSON/YAML files, or a URL returning the same. Tenant-specific; format convention documented per tenant. |
| `--mode {full,incremental}` | no       | `incremental`  | `incremental`: upsert by content hash (skip re-embed on unchanged chunks). `full`: force re-embed every chunk. |
| `--dry-run`                 | no       | false          | Print the plan (chunks to add / update / delete) without writing to the DB or calling the embedder. |
| `--skip-synthetic-questions`| no       | false          | Skip the synthetic-question-generation LLM pass. Used when the eval harness is measuring content-only recall (R11 gate). |
| `--verbose`                 | no       | false          | Emit per-chunk decisions (added / updated / skipped / deleted) as structured loguru output. |

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Ingestion completed. Summary line on stdout. |
| 64   | Bad CLI args (unknown tenant slug, bad path, etc.). |
| 65   | Source parse error (bad Markdown / JSON). |
| 74   | Upstream dependency failure (DB, embedder, rewriter LLM). No partial writes committed. |
| 75   | Partial success — some chunks failed; successful chunks committed; failures logged with kb_entry_key. Operator re-runs. |

---

## Output (on stdout, structured)

On success, one JSON line to stdout:

```json
{
  "tenant": "<slug>",
  "mode": "incremental",
  "elapsed_ms": 12834,
  "entries_seen": 42,
  "chunks_added": 7,
  "chunks_updated": 3,
  "chunks_unchanged": 117,
  "chunks_deleted": 2,
  "synthetic_questions_added": 28,
  "synthetic_questions_deleted": 8,
  "embed_calls": 10,
  "rewriter_calls": 7
}
```

On `--dry-run`, same shape but no writes occurred. On exit codes 65/74/75,
additionally write a second JSON line naming affected entries.

---

## Idempotency guarantees

- Running with unchanged source → `chunks_added = 0`, `chunks_updated = 0`,
  `embed_calls = 0`. Enforced by content-hash skip (SC-009).
- Content-hash computation: `SHA-256(normalize(content))` where `normalize`
  strips trailing whitespace and normalizes newlines. Any richer normalization
  (punctuation, diacritics) is NOT applied — it would make "trivial" edits
  invisible and cause staleness.
- Upsert keys: `(tenant_id, kb_entry_id, section, is_synthetic_question=false)`
  for content chunks. Synthetic questions are DELETE + INSERT together with
  their parent on parent-content change; their key is
  `(tenant_id, parent_chunk_id, sequence)`.

---

## Concurrency

The CLI takes a Postgres advisory lock keyed on `tenant_id` for the duration
of the run. Two concurrent ingests for the same tenant serialize; two
concurrent ingests for different tenants proceed in parallel.

Advisory lock key: `hash_int8('kb_ingest:' || tenant_id::text)`.

---

## Failure modes

- **Embedder OOM / missing CUDA**: exit 74, nothing committed.
- **Postgres connection lost mid-run**: exit 74, the current transaction
  rolls back; previous transactions (if any) remain. Operator re-runs
  `--mode incremental` — idempotent.
- **Rewriter LLM rate-limited**: the metadata-extraction pass retries with
  exponential backoff up to 3 attempts per entry; after that, the entry's
  metadata stays empty but the chunk is still committed (retrieval still
  works, just without structured metadata). Exit code 75 if any entry hit
  this.
- **Source parse error on one entry**: skip the entry, log the key, continue;
  exit code 75.

---

## Security / tenancy

- The CLI reads `tenant_id` from `tenants.slug` at startup. It never accepts a
  tenant_id directly from CLI args to avoid operator typos writing to the
  wrong tenant.
- Source content is treated as untrusted. The ingestion pipeline does NOT
  evaluate, execute, or shell-out to any part of it.
- The CLI requires DB credentials via env vars (`DB_URL`) and LLM credentials
  via env vars (`LLM_BASE_URL` / `LLM_API_KEY`). Never accepted on the
  command line — would leak via shell history / `ps`.

# Ingestion CLI Contract

Command: `python -m retriever.ingest` (inside `services/retriever/`).
Alternatively `uv run python -m retriever.ingest` from the service root.

Not exposed over HTTP in v1 (research R10). Run manually or from a scheduled
job on the retriever host.

---

## Synopsis

```
python -m ingest \
  --tenant <tenant-slug> \
  --source <path> \
  [--mode {full,incremental}] \
  [--dry-run] \
  [--allow-mass-deletion] \
  [--skip-synthetic-questions] \
  [--verbose]
```

Run from `services/retriever/` (locally) or `docker compose exec retriever uv run python -m ingest ...` (production). The flat-layout retriever package means the canonical invocation is `python -m ingest`, NOT `python -m retriever.ingest` — the latter would require a parent package this service does not have.

---

## Arguments

| Arg                         | Required | Default        | Notes |
|-----------------------------|----------|----------------|-------|
| `--tenant <slug>`           | yes      | —              | Must match `tenants.slug`. The CLI rejects unknown or non-active slugs with exit 64. |
| `--source <path>`           | yes      | —              | Path to a directory of `*.json` files in the canonical KB-entry shape (see below). Tenant-source mapping is operator policy; the CLI is source-agnostic. |
| `--mode {full,incremental}` | no       | `incremental`  | `incremental`: upsert by content hash (skip re-embed on unchanged chunks). `full`: force re-embed every chunk. |
| `--dry-run`                 | no       | false          | Print the plan (entries to add / update / delete) without writing to the DB or calling the embedder. |
| `--allow-mass-deletion`     | no       | false          | Override the >25% delete-rate safeguard. Only set when the operator deliberately intends to prune ≥25% of existing entries (e.g., bulk Chatwoot cleanup). The safeguard fires above `MASS_DELETION_FLOOR=4` deletions; a 5-entry KB with 4 deletes triggers it. |
| `--skip-synthetic-questions`| no       | false          | Skip the synthetic-question-generation LLM pass. Used when the eval harness is measuring content-only recall (R11 gate). Default behavior runs synthesis as a separate phase after the content commit. |
| `--verbose`                 | no       | false          | Emit per-entry decisions (added / updated / unchanged / deleted) as structured loguru output on stderr. Stdout still carries only the JSON summary. |

---

## Source format

Each `*.json` file under `--source` MUST conform to:

```json
{
  "kb_entry_key": "<namespace>:<id>",
  "title": "...",
  "content": "...markdown...",
  "source_uri": "https://...",       // optional
  "metadata": { "...": "..." }       // optional, free-form, lands in kb_chunks.metadata
}
```

The `<namespace>` segment of `kb_entry_key` MUST match `[A-Za-z0-9.-]+` (no SQL LIKE wildcards or whitespace). A single source directory MUST contain entries from exactly one namespace; mixed namespaces exit with code 65. The fetcher (`tools/fetch_chatwoot_kb.py`) writes entries with `kb_entry_key="chatwoot:<id>"`.

`_manifest.json` in the source directory is reserved for fetcher metadata and skipped by ingest.

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

On success, exactly one JSON line is written to stdout. The shape groups
related counts so consumers can iterate over per-phase counters cleanly:

```json
{
  "tenant": "<slug>",
  "source": "/abs/path/to/source-dir",
  "namespace": "chatwoot",
  "mode": "incremental",
  "dry_run": false,
  "entries": {
    "seen": 71,
    "added": 2,
    "updated": 3,
    "unchanged": 65,
    "deleted": 1
  },
  "chunks": {
    "added": 17,
    "deleted": 8,
    "unchanged": 462
  },
  "embed_calls": 17,
  "synthetic_questions": {
    "added": 68,
    "deleted": 0
  },
  "rewriter_calls": 17,
  "parse_errors": [],
  "elapsed_ms": 42100
}
```

`parse_errors` is an array of human-readable strings naming each source
file that failed validation, in the form `"<filename>: <reason>"`. An
empty array means every `*.json` parsed cleanly; non-empty + exit code
75 (partial success) means the run committed valid entries while
recording the failures. Consumers that want to alert on parse drift
should branch on `len(parse_errors) > 0` rather than expecting a
separate JSON document.

`synthetic_questions.deleted` is always `0` in the current
implementation — synthetic-question rows cascade-delete with their
parent on content replacement (the content phase handles it before the
synth phase runs). The field is reserved for a future explicit
"force-regenerate" mode.

On `--dry-run`, the same shape is emitted; `entries` reflects the
diff plan and `chunks.added` / `chunks.deleted` are zero (no writes
happened). `chunks.unchanged` is populated from the DB so the operator
can see the baseline.

---

## Idempotency guarantees

- Running with unchanged source → `entries.added = entries.updated = 0`,
  `entries.unchanged = entries.seen`, `chunks.added = embed_calls = 0`.
  Enforced by content-hash skip (SC-009).
- Content-hash computation: `SHA-256(normalize(content))` where `normalize`
  strips NUL bytes (Postgres TEXT rejects them), normalizes Windows line
  endings to Unix, and strips trailing whitespace per line. Any richer
  normalization (punctuation, diacritics, image stripping) is NOT applied —
  it would make "trivial" edits invisible and cause staleness.
- Upsert keys: `(tenant_id, kb_entry_id, section)` with `NULLS NOT DISTINCT`
  for content chunks (see migration 0004). Synthetic questions are DELETE
  + INSERT together with their parent on parent-content change via
  `parent_chunk_id ON DELETE CASCADE`; their key is the BIGSERIAL `id`.

---

## Concurrency

The CLI takes a Postgres advisory lock keyed on `tenant_id` for the duration
of the run. Two concurrent ingests for the same tenant serialize; two
concurrent ingests for different tenants proceed in parallel.

Advisory lock key: `hash_int8('kb_ingest:' || tenant_id::text)`.

---

## Failure modes

- **Embedder OOM / missing CUDA / model-load error**: exit 74, nothing
  committed. Caught at `main()` boundary.
- **Postgres connection lost mid-run / SQLAlchemy error**: exit 74, the
  current transaction rolls back. Operator re-runs `--mode incremental`
  — idempotent.
- **Rewriter LLM rate-limited (synth phase only)**: per-chunk retries
  with exponential backoff up to 3 attempts. Beyond that, the chunk is
  skipped and remains naked for the next run (the NOT EXISTS filter
  picks it up). After 5 consecutive chunk failures the synth phase
  aborts to avoid burning quota; content is unaffected.
- **Synth-phase per-chunk DB error**: each chunk's question batch runs
  in its own transaction so a FK violation (concurrent ingest deleted
  the parent) or transient asyncpg error rolls back only that chunk's
  questions; the chunk stays naked for the next run.
- **Source parse error on one entry**: skip the entry, append to
  `parse_errors`, continue. Exit code 75 if any entry hit this.
- **Empty source dir or all entries failed parse**: short-circuit with
  exit 75 and `parse_errors` populated. The CLI explicitly does NOT
  fall through to the diff in this case — empty source would otherwise
  match every existing entry for the (now-empty) namespace and queue
  them all for deletion.
- **Mass-deletion safeguard**: when `len(plan.delete) >= 4` AND
  `len(plan.delete) / len(db_entries) > 0.25`, exit 75 without writes
  unless `--allow-mass-deletion` is passed.
- **NUL bytes in source content**: stripped during normalization with a
  loguru warning (`ingest.nul_bytes_stripped`). Postgres TEXT rejects
  NUL; silently failing would wedge the entire run on rollback.

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

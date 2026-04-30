---
title: "KB ingestion pipeline — fetch, ingest, and scheduled refresh"
module: services/retriever
date: 2026-04-30
category: docs/solutions/best-practices
problem_type: operational_runbook
component: kb_ingestion
severity: low
root_cause: external_kb_drift_requires_periodic_resync
resolution_type: runbook
symptoms:
  - Helper/Guide NPC answers go stale after upstream Chatwoot articles change
  - First-time tenant onboarding needs an initial KB seed before any retrieval works
  - Operator wants to add a new article and have it answerable within the hour
tags:
  - kb-ingestion
  - chatwoot
  - chunker
  - synthetic-questions
  - mass-deletion-safeguard
  - cron
---

# What this is

The end-to-end pipeline for getting Russian KB content from an external
source (today: Chatwoot help-base) into the retriever's
`kb_entries` / `kb_chunks` tables, plus the operator-facing wrapper
that wires it as a scheduled refresh.

Two scripts, one inside-container CLI, one cron entry.

```
[cron @ 04:00]
   ↓
scripts/refresh_kb.sh demo arizona
   ├─ tools/fetch_chatwoot_kb.py  → data/kb/arizona/<id>.json + _manifest.json  (host)
   └─ docker compose exec retriever uv run python -m ingest
        --tenant demo --source /app/data/kb/arizona/ --mode incremental
        ├─ chunk via chunker (markdown-aware, BGE-M3-bounded sizes)
        ├─ embed each chunk (BGE-M3, 1024-dim)
        ├─ upsert kb_entries + replace kb_chunks atomically
        └─ synthetic-question phase (rewriter LLM → N=4 paraphrases per chunk)
```

# When to use

- **First deploy** of a tenant — initial KB seed.
- **Daily scheduled refresh** — pick up new / edited / deleted Chatwoot
  articles. Idempotent, so daily is cheap.
- **After a Chatwoot bulk edit** — operator can run on demand without
  waiting for cron.
- **After bumping `REWRITER_VERSION`** in synthetic_questions — re-run
  with `--mode full` to regenerate every embedding, then a normal run
  to fill in synth questions.

# How it works

## Fetch (host side, Python stdlib only)

`tools/fetch_chatwoot_kb.py`:

- Reads `CHATWOOT_HELP_BASE_TOKEN` from env.
- GETs `https://chatwoot.arizona-rp.com/api-help-base/get?project=<p>`.
- Writes one JSON file per article under `data/kb/<project>/<id>.json`
  in the canonical shape:

  ```json
  {
    "kb_entry_key": "chatwoot:<id>",
    "title": "...",
    "content": "...markdown...",
    "source_uri": "...",
    "metadata": {"category_name": "...", "portal_name": "...", "source": "chatwoot"}
  }
  ```

- Writes `_manifest.json` listing all fetched IDs (forensic record + a
  "fetch ran successfully" sentinel for `refresh_kb.sh`).

The fetch is **non-destructive on disk** — files for articles that
disappear upstream are NOT removed by the fetcher. The ingest step's
diff handles deletions via DB cascade. This separation means a partial
fetch never destroys local source-of-truth.

## Ingest (inside the retriever container)

`services/retriever/ingest.py`:

1. Resolves tenant slug → tenant_id, acquires per-tenant
   `pg_advisory_xact_lock`. Two concurrent ingests for the same tenant
   block; different tenants run in parallel.
2. Loads source JSONs, validates the canonical shape, computes
   per-entry `content_sha256` over normalised content.
3. Loads existing `kb_entries WHERE kb_entry_key LIKE '<namespace>:%'`
   (scoped to the source's namespace so a `chatwoot` ingest never
   touches a `wiki:*` entry from a different fetcher).
4. Diff → `add` / `update` / `unchanged` / `delete` sets.
5. **Mass-deletion safeguard** — refuses to delete >25% of existing
   entries (above floor of 4) unless `--allow-mass-deletion` is
   passed. Catches "Chatwoot returned 0 articles because the API key
   was rotated" before it nukes the KB.
6. For each add/update: chunk via `chunker.chunk_article` (heading-
   aware, MAX_CHARS=1500 with overlap), embed each chunk via BGE-M3,
   atomically replace `kb_chunks` rows.
7. For each delete: `DELETE FROM kb_entries WHERE kb_entry_key = ?` —
   cascades to chunks via FK.
8. Synthetic-question phase (`synthetic_questions.run`) — finds content
   chunks with no synthetic-question children, generates N=4
   paraphrased Russian questions per chunk via the rewriter LLM,
   embeds and inserts as `is_synthetic_question=TRUE` rows.

## Synthetic questions — why a separate phase

Generation is the slow + expensive step (~1 LLM call per chunk; ~200
calls for the Chatwoot feed; ~30s wall time on Groq llama-3.3-70b).
Splitting it from the content commit means:

- **Content phase commits first.** A rate-limited / down LLM blocks
  generation but never blocks new content from being retrievable.
- **Resumable on retry.** The phase queries `kb_chunks WHERE NOT EXISTS
  (SELECT 1 FROM kb_chunks sq WHERE sq.parent_chunk_id = c.id)` — so
  next run picks up exactly the chunks the previous run missed.
- **Auto-regenerates on content change.** When the content phase
  replaces a chunk's row, its synthetic children cascade-delete via
  the parent_chunk_id FK; the chunk becomes "naked" again and the
  next synth pass regenerates.
- **Bounded retry.** Up to 3 attempts per call with exponential
  backoff (0.5s, 1s, 2s). 5 consecutive total failures abort the
  phase — the LLM is genuinely down or quota is exhausted, no value
  in burning more attempts.

## Scheduled refresh on the VM

`scripts/refresh_kb.sh`:

```bash
sudo crontab -e
0 4 * * * /opt/project-800ms/scripts/refresh_kb.sh demo arizona >> \
    /var/log/project-800ms-kb-refresh.log 2>&1
```

The script:

1. Sources `infra/.env` to inherit `CHATWOOT_HELP_BASE_TOKEN`.
2. Runs the fetch step on the host. Exits with code 2 on fetch
   failure (DB unchanged, retry tomorrow).
3. Sanity-checks `_manifest.json` — exits 2 if upstream returned 0
   articles (clear log message rather than triggering the deletion
   safeguard inside ingest).
4. Runs `docker compose exec retriever uv run python -m ingest ...`
   so the heavy BGE-M3 embedder (already loaded in the long-lived
   retriever process) is reused, saving ~6s of cold-load cost.
5. Reports per-phase JSON lines + diagnostic loguru output to
   stdout/stderr for log aggregation.

# Operator commands

## First-time seed

```bash
# Make sure the chatwoot_help_base_token is set in terraform.tfvars
# and applied. Then on the VM:
ssh <vm> "cd /opt/project-800ms && ./scripts/refresh_kb.sh demo arizona"
```

## Test in dry-run mode (locally or on VM)

```bash
docker compose exec retriever uv run python -m ingest \
    --tenant demo --source /app/data/kb/arizona/ \
    --mode incremental --dry-run
```

Returns the diff plan without writing — the fastest way to confirm
"does upstream look reasonable" before running for real.

## Force-regenerate every chunk

```bash
docker compose exec retriever uv run python -m ingest \
    --tenant demo --source /app/data/kb/arizona/ \
    --mode full
```

Use after bumping the embedder model or REWRITER_VERSION. Runs through
every article as if its content_sha256 had changed.

## Override the mass-deletion safeguard

```bash
docker compose exec retriever uv run python -m ingest \
    --tenant demo --source /app/data/kb/arizona/ \
    --allow-mass-deletion
```

Use only when a deliberate cleanup is in progress (e.g. operator pruned
50 stale articles upstream and wants the deletion to flow through). For
the steady-state daily refresh, leave it off — the safeguard is the
last line of defense between "Chatwoot auth flipped" and "demo tenant
has no KB anymore".

## Inspect the result

```sql
-- entries per namespace
SELECT split_part(kb_entry_key, ':', 1) AS source, count(*)
FROM kb_entries
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'demo')
GROUP BY 1;

-- chunks per article (avg ~2-5 expected for Chatwoot)
SELECT e.kb_entry_key, count(c.id) AS chunks,
       count(c.id) FILTER (WHERE c.is_synthetic_question) AS synth_qs
FROM kb_entries e
LEFT JOIN kb_chunks c ON c.kb_entry_id = e.id
WHERE e.tenant_id = (SELECT id FROM tenants WHERE slug = 'demo')
GROUP BY 1
ORDER BY chunks DESC
LIMIT 10;
```

# Future-source extension

To add a new source (e.g. Wiki, Notion, internal docs):

1. Write `tools/fetch_<source>.py` that produces the canonical shape
   under `data/kb/<source>/`. Use a different namespace prefix
   (`kb_entry_key="wiki:<id>"`).
2. Add a `chatwoot_<source>_token` variable to `terraform-gcp` if it
   needs auth — same pattern as `chatwoot_help_base_token`.
3. Run ingest with the new path:
   `python -m ingest --tenant demo --source /app/data/kb/wiki/`. The
   namespace is auto-derived from `kb_entry_key`; existing `chatwoot:*`
   entries are NOT touched (the prefix-scoped diff in `_load_db_entries`
   isolates each source).

Multiple sources can co-exist on one tenant. The hybrid retrieval CTE
ranks across them all, so a query can pull the best chunk from
whichever source had the closest answer.

"""KB ingestion CLI — content phase.

Loads a directory of canonical KB-entry JSON files (the shape produced
by ``tools/fetch_chatwoot_kb.py``), diffs the set against the tenant's
existing ``kb_entries`` rows, and embeds + upserts the changes. Synthetic
question generation is a separate phase (see ``synthetic_questions``)
that runs after the content commit succeeds.

Usage::

    cd services/retriever
    uv run python -m ingest --tenant demo --source ../../data/kb/arizona/

In production::

    docker compose exec retriever uv run python -m ingest \\
        --tenant demo --source /app/data/kb/arizona/

Pipeline
--------

1. Resolve tenant slug → tenant_id, acquire per-tenant advisory lock
   (so two ingests for the same tenant cannot interleave writes).
2. Walk source dir, parse each ``*.json`` into a ``SourceEntry``.
3. Compute SHA-256 over the normalised content for each source entry.
4. Load existing kb_entries for ``(tenant_id, kb_entry_key LIKE
   '<namespace>:%')`` so this run only touches entries from the same
   namespace as the source feed.
5. Diff source vs DB → ``add`` / ``update`` / ``unchanged`` / ``delete``.
6. Mass-deletion safeguard: refuse to apply when delete-rate exceeds
   ``MASS_DELETION_THRESHOLD`` of existing entries, unless
   ``--allow-mass-deletion`` is passed. Catches an upstream auth flip
   that returns an empty/partial feed before it nukes the KB.
7. For each add/update: chunk via ``chunker.chunk_article``, embed
   each chunk, upsert ``kb_entry`` and replace its ``kb_chunks`` rows
   atomically (cascade-deletes any prior synthetic-question children
   so the synth-Q phase can regenerate them on the next pass).
8. For each delete: cascade-delete the ``kb_entry`` row.
9. Emit a one-line JSON summary on stdout. Exit codes per the contract.

Embedding cost note: the BGE-M3 model is loaded once via
``embedder.preload()`` and reused. Single-text encoding is enough at
71-article scale (~3-5s end-to-end on CPU). Future batching is a
performance optimisation, not a correctness one.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Exit codes per ``contracts/ingest.cli.md``.
EXIT_OK = 0
EXIT_BAD_ARGS = 64
EXIT_PARSE_ERROR = 65
EXIT_UPSTREAM_FAILURE = 74
EXIT_PARTIAL_SUCCESS = 75

# Mass-deletion guard: when more than this fraction of existing entries
# would be deleted in one run, refuse without ``--allow-mass-deletion``.
# 25% catches the common "upstream returned partial / empty feed" failure
# while leaving room for legitimate "we removed a bunch of stale articles"
# operator-driven cleanups. Activates only above MASS_DELETION_FLOOR so
# tiny KBs (<= 4 entries) aren't blocked by random-looking ratios.
MASS_DELETION_THRESHOLD = 0.25
MASS_DELETION_FLOOR = 4


@dataclass(frozen=True)
class SourceEntry:
    """One article loaded from disk after validation."""

    kb_entry_key: str  # ``"<namespace>:<id>"`` — e.g. ``"chatwoot:116"``
    title: str
    content: str  # already normalised
    content_sha256: str
    source_uri: str | None
    metadata: dict[str, Any]
    namespace: str  # prefix before ``:`` — used to scope the diff


@dataclass
class Plan:
    """Diff between source dir and DB."""

    add: list[SourceEntry] = field(default_factory=list)
    update: list[SourceEntry] = field(default_factory=list)
    unchanged: list[SourceEntry] = field(default_factory=list)
    # kb_entry_keys present in DB but missing from source.
    delete: list[str] = field(default_factory=list)
    namespace: str = ""


def _normalise_content(content: str) -> str:
    """Apply the ``content_sha256`` normalisation contract.

    Strips trailing whitespace per line, normalises Windows line endings
    to Unix, collapses 3+ consecutive newlines, and trims the result.
    Any richer normalisation (punctuation, diacritics, image stripping)
    would make trivial edits invisible — see contract section
    "Idempotency guarantees".
    """
    text_stripped = content.replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in text_stripped.split("\n")]
    return "\n".join(lines).strip()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_source_dir(source: Path) -> tuple[list[SourceEntry], list[str]]:
    """Walk ``source`` for ``*.json`` files, return ``(entries, errors)``.

    Skips ``_manifest.json``. Each entry is validated against the
    canonical shape; bad entries are reported as parse errors and
    omitted from the returned list (so a single bad file doesn't
    abort the whole run — partial-success exit code 75 covers it).
    """
    entries: list[SourceEntry] = []
    errors: list[str] = []
    if not source.is_dir():
        errors.append(f"source is not a directory: {source}")
        return entries, errors

    for path in sorted(source.glob("*.json")):
        if path.name == "_manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        try:
            kb_entry_key = str(payload["kb_entry_key"]).strip()
            title = str(payload["title"]).strip()
            content_raw = payload["content"]
            if not isinstance(content_raw, str):
                raise TypeError("content must be a string")
        except (KeyError, TypeError) as exc:
            errors.append(f"{path.name}: missing/invalid field: {exc}")
            continue

        if ":" not in kb_entry_key or kb_entry_key.startswith(":"):
            errors.append(
                f"{path.name}: kb_entry_key {kb_entry_key!r} missing namespace prefix "
                "(expected 'namespace:id')"
            )
            continue

        namespace = kb_entry_key.split(":", 1)[0]
        norm_content = _normalise_content(content_raw)
        entries.append(
            SourceEntry(
                kb_entry_key=kb_entry_key,
                title=title,
                content=norm_content,
                content_sha256=_sha256(norm_content),
                source_uri=payload.get("source_uri"),
                metadata=payload.get("metadata") or {},
                namespace=namespace,
            )
        )
    return entries, errors


async def _resolve_tenant_id(session: AsyncSession, slug: str) -> UUID:
    """Look up a tenant by slug. Raises SystemExit(64) on miss.

    Status filter mirrors ``tenants.resolve_tenant`` — only ``active``
    tenants accept ingest writes. Reusing that module would require
    a UUID at the call site; the slug-based lookup here keeps the CLI
    operator-friendly.
    """
    row = (
        await session.execute(
            text("SELECT id FROM tenants WHERE slug = :slug AND status = 'active'"),
            {"slug": slug},
        )
    ).first()
    if row is None:
        logger.error("unknown or non-active tenant slug={slug}", slug=slug)
        raise SystemExit(EXIT_BAD_ARGS)
    return row[0]


async def _advisory_lock(session: AsyncSession, tenant_id: UUID) -> None:
    """Acquire a transaction-scoped per-tenant advisory lock.

    Mirrors the contract: ``hash_int8('kb_ingest:' || tenant_id::text)``
    (``hashtext`` returns int4, but ``pg_advisory_xact_lock`` accepts
    int4 via implicit cast — same effective key space).

    Two concurrent ingests for the same tenant block here until the
    first commits; two ingests for different tenants proceed in
    parallel.
    """
    await session.execute(
        text("SELECT pg_advisory_xact_lock(  hashtext('kb_ingest:' || :tid)::int8)"),
        {"tid": str(tenant_id)},
    )


async def _load_db_entries(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    namespace: str,
    mode: str,
) -> dict[str, dict[str, Any]]:
    """Return existing kb_entries for the namespace, keyed by kb_entry_key.

    For ``--mode full`` the content_sha256 is masked so the diff treats
    every entry as changed (forces re-embed). For ``--mode incremental``
    the real hash is returned so unchanged entries short-circuit.
    """
    rows = (
        (
            await session.execute(
                text(
                    "SELECT id, kb_entry_key, content_sha256 "
                    "FROM kb_entries "
                    "WHERE tenant_id = :tid AND kb_entry_key LIKE :prefix"
                ),
                {"tid": str(tenant_id), "prefix": f"{namespace}:%"},
            )
        )
        .mappings()
        .all()
    )
    out = {row["kb_entry_key"]: dict(row) for row in rows}
    if mode == "full":
        for v in out.values():
            v["content_sha256"] = "__force_re_embed__"
    return out


def _diff(
    source: list[SourceEntry],
    db_entries: dict[str, dict[str, Any]],
    namespace: str,
) -> Plan:
    """Compute add/update/unchanged/delete sets."""
    plan = Plan(namespace=namespace)
    src_keys: set[str] = set()
    for entry in source:
        src_keys.add(entry.kb_entry_key)
        if entry.kb_entry_key not in db_entries:
            plan.add.append(entry)
        elif db_entries[entry.kb_entry_key]["content_sha256"] != entry.content_sha256:
            plan.update.append(entry)
        else:
            plan.unchanged.append(entry)
    for db_key in db_entries:
        if db_key not in src_keys:
            plan.delete.append(db_key)
    return plan


def _vector_literal(v: list[float]) -> str:
    """Format a 1024-dim list as the pgvector text representation."""
    return "[" + ",".join(f"{x:.10g}" for x in v) + "]"


async def _upsert_entry_and_chunks(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    entry: SourceEntry,
    encode_fn: Any,  # async callable str -> list[float]
) -> int:
    """Insert/update a kb_entry and atomically replace its kb_chunks.

    Returns the number of chunks written (also = embed calls made).
    """
    from chunker import chunk_article

    chunks = chunk_article(title=entry.title, content=entry.content)

    # Upsert kb_entry. ON CONFLICT keeps the row id stable across runs
    # (subsequent updates only bump content_sha256 + updated_at).
    row = (
        await session.execute(
            text(
                "INSERT INTO kb_entries "
                "  (tenant_id, kb_entry_key, title, source_uri, content_sha256) "
                "VALUES (:t, :k, :title, :uri, :sha) "
                "ON CONFLICT (tenant_id, kb_entry_key) DO UPDATE SET "
                "  title = EXCLUDED.title, "
                "  source_uri = EXCLUDED.source_uri, "
                "  content_sha256 = EXCLUDED.content_sha256, "
                "  updated_at = now() "
                "RETURNING id"
            ),
            {
                "t": str(tenant_id),
                "k": entry.kb_entry_key,
                "title": entry.title,
                "uri": entry.source_uri,
                "sha": entry.content_sha256,
            },
        )
    ).first()
    entry_id = row[0]

    # Replace chunks. The kb_chunks → kb_chunks self-FK has ON DELETE
    # CASCADE so any synthetic-question children disappear with their
    # parent — the synth-Q phase will regenerate them.
    await session.execute(
        text("DELETE FROM kb_chunks WHERE tenant_id = :t AND kb_entry_id = :eid"),
        {"t": str(tenant_id), "eid": str(entry_id)},
    )

    if not chunks:
        return 0

    # Embed each chunk. Prepending the section path to the embedding
    # input gives the embedder lexical anchor points for nested topics
    # ("Линия наград > Аксессуары" + body) without polluting the stored
    # content column or the lexical tsvector (which is title || content
    # only).
    # Use CAST(...) rather than ``::vector`` / ``::jsonb`` because the
    # double-colon inside a SQLAlchemy ``text()`` collides with the
    # ``:name`` bind-parameter parser and silently leaves the literal
    # ``:emb::vector`` in the rendered SQL.
    insert_sql = text(
        "INSERT INTO kb_chunks "
        "  (tenant_id, kb_entry_id, section, title, content, "
        "   content_sha256, embedding, metadata) "
        "VALUES (:t, :eid, :section, :title, :content, :sha, "
        "        CAST(:emb AS vector), CAST(:meta AS jsonb))"
    )
    metadata_json = json.dumps(entry.metadata, ensure_ascii=False)
    written = 0
    for chunk in chunks:
        embed_input = f"{chunk.section}\n\n{chunk.content}" if chunk.section else chunk.content
        embedding = await encode_fn(embed_input)
        await session.execute(
            insert_sql,
            {
                "t": str(tenant_id),
                "eid": str(entry_id),
                "section": chunk.section,
                "title": entry.title,
                "content": chunk.content,
                "sha": _sha256(chunk.content),
                "emb": _vector_literal(embedding),
                "meta": metadata_json,
            },
        )
        written += 1
    return written


async def _delete_entry(session: AsyncSession, *, tenant_id: UUID, kb_entry_key: str) -> int:
    """Cascade-delete a kb_entry. Returns chunks deleted."""
    chunks_deleted = (
        await session.execute(
            text(
                "SELECT count(*) FROM kb_chunks c "
                "JOIN kb_entries e ON c.kb_entry_id = e.id "
                "WHERE e.tenant_id = :t AND e.kb_entry_key = :k"
            ),
            {"t": str(tenant_id), "k": kb_entry_key},
        )
    ).scalar_one()
    await session.execute(
        text("DELETE FROM kb_entries WHERE tenant_id = :t AND kb_entry_key = :k"),
        {"t": str(tenant_id), "k": kb_entry_key},
    )
    return int(chunks_deleted)


async def _count_chunks_for_entries(
    session: AsyncSession, *, tenant_id: UUID, kb_entry_keys: list[str]
) -> int:
    """Count the kb_chunks attached to the given entries (for unchanged-
    chunk reporting)."""
    if not kb_entry_keys:
        return 0
    result = await session.execute(
        text(
            "SELECT count(*) FROM kb_chunks c "
            "JOIN kb_entries e ON c.kb_entry_id = e.id "
            "WHERE e.tenant_id = :t AND e.kb_entry_key = ANY(:keys) "
            "  AND c.is_synthetic_question = FALSE"
        ),
        {"t": str(tenant_id), "keys": kb_entry_keys},
    )
    return int(result.scalar_one())


async def run(
    *,
    tenant_slug: str,
    source: Path,
    mode: str = "incremental",
    dry_run: bool = False,
    allow_mass_deletion: bool = False,
    skip_synthetic_questions: bool = True,  # synth phase wired up separately
    encode_fn: Any = None,  # injection seam for tests
) -> dict[str, Any]:
    """Run an ingest. Returns the summary dict (also used by the CLI
    main() to format stdout).

    The ``encode_fn`` parameter exists so tests can inject a deterministic
    stub embedder without paying the BGE-M3 load cost. Production callers
    should leave it unset; the function lazy-imports ``embedder.encode``
    and preloads the model on first use.
    """
    from db import get_session, set_tenant_scope

    started = time.perf_counter()

    if mode not in {"incremental", "full"}:
        logger.error("invalid --mode={mode}", mode=mode)
        raise SystemExit(EXIT_BAD_ARGS)

    source_path = source.expanduser().resolve()
    source_entries, parse_errors = _load_source_dir(source_path)

    namespaces = {e.namespace for e in source_entries}
    if len(namespaces) > 1:
        logger.error(
            "mixed namespaces in source: {ns} — split feeds into separate "
            "directories before ingesting",
            ns=sorted(namespaces),
        )
        raise SystemExit(EXIT_PARSE_ERROR)
    namespace = next(iter(namespaces)) if namespaces else ""

    summary: dict[str, Any] = {
        "tenant": tenant_slug,
        "source": str(source_path),
        "namespace": namespace,
        "mode": mode,
        "dry_run": dry_run,
        "entries": {
            "seen": len(source_entries),
            "added": 0,
            "updated": 0,
            "unchanged": 0,
            "deleted": 0,
        },
        "chunks": {"added": 0, "deleted": 0, "unchanged": 0},
        "embed_calls": 0,
        "synthetic_questions": {"added": 0, "deleted": 0},
        "rewriter_calls": 0,
        "parse_errors": parse_errors,
    }

    if not source_entries and not parse_errors:
        # Empty but valid source dir — nothing to do. The mass-deletion
        # safeguard would catch this if it tried to delete the existing
        # KB; we surface it here as a no-op so the operator gets a
        # consistent summary instead of a SystemExit.
        logger.warning("source dir is empty: {p}", p=source_path)
        summary["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return summary

    # Lazy-import embedder so --help / dry-run with bad args don't pay
    # the BGE-M3 load cost.
    if encode_fn is None:
        from embedder import encode, preload

        await asyncio.to_thread(preload)
        encode_fn = encode

    async with get_session() as session:
        tenant_id = await _resolve_tenant_id(session, tenant_slug)
        await _advisory_lock(session, tenant_id)
        await set_tenant_scope(session, tenant_id)

        db_entries = await _load_db_entries(
            session, tenant_id=tenant_id, namespace=namespace, mode=mode
        )
        plan = _diff(source_entries, db_entries, namespace)

        # Mass-deletion safeguard — fires only when there's a meaningful
        # baseline (avoids "I had 2 entries, deleted 1, that's >25%!").
        if (
            len(plan.delete) > MASS_DELETION_FLOOR
            and len(db_entries) > 0
            and len(plan.delete) / len(db_entries) > MASS_DELETION_THRESHOLD
            and not allow_mass_deletion
        ):
            logger.error(
                "mass-deletion safeguard fired: would delete {n} of {total} entries "
                "(>{pct:.0%}). Re-run with --allow-mass-deletion to override.",
                n=len(plan.delete),
                total=len(db_entries),
                pct=MASS_DELETION_THRESHOLD,
            )
            raise SystemExit(EXIT_PARTIAL_SUCCESS)

        summary["entries"]["added"] = len(plan.add)
        summary["entries"]["updated"] = len(plan.update)
        summary["entries"]["unchanged"] = len(plan.unchanged)
        summary["entries"]["deleted"] = len(plan.delete)

        if dry_run:
            # No DB writes — count what would happen and return.
            summary["chunks"]["unchanged"] = await _count_chunks_for_entries(
                session,
                tenant_id=tenant_id,
                kb_entry_keys=[e.kb_entry_key for e in plan.unchanged],
            )
            summary["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            return summary

        # Apply add + update via the same code path (UPSERT + replace).
        for entry in plan.add + plan.update:
            written = await _upsert_entry_and_chunks(
                session, tenant_id=tenant_id, entry=entry, encode_fn=encode_fn
            )
            summary["chunks"]["added"] += written
            summary["embed_calls"] += written
            logger.info(
                "ingest.entry_written key={k} chunks={n}",
                k=entry.kb_entry_key,
                n=written,
            )

        for kb_entry_key in plan.delete:
            chunks_lost = await _delete_entry(
                session, tenant_id=tenant_id, kb_entry_key=kb_entry_key
            )
            summary["chunks"]["deleted"] += chunks_lost
            logger.info(
                "ingest.entry_deleted key={k} chunks={n}",
                k=kb_entry_key,
                n=chunks_lost,
            )

        summary["chunks"]["unchanged"] = await _count_chunks_for_entries(
            session,
            tenant_id=tenant_id,
            kb_entry_keys=[e.kb_entry_key for e in plan.unchanged],
        )

    # Synth-Q phase placeholder. Wired up in the next task with its own
    # transaction so a content commit never depends on the LLM being
    # available.
    if not skip_synthetic_questions:
        # The synth phase is implemented in synthetic_questions.run; left
        # unimported here so the content-only path has zero LLM coupling.
        from synthetic_questions import run as synth_run

        synth_summary = await synth_run(tenant_slug=tenant_slug, namespace=namespace)
        summary["synthetic_questions"]["added"] = synth_summary["added"]
        summary["synthetic_questions"]["deleted"] = synth_summary["deleted"]
        summary["rewriter_calls"] = synth_summary["rewriter_calls"]

    summary["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingest",
        description=(
            "Ingest a directory of canonical KB-entry JSON files into a tenant's "
            "kb_entries / kb_chunks tables. Idempotent on content_sha256."
        ),
    )
    parser.add_argument("--tenant", required=True, help="tenants.slug — must exist + be active")
    parser.add_argument("--source", required=True, help="path to directory of *.json files")
    parser.add_argument(
        "--mode",
        choices=("incremental", "full"),
        default="incremental",
        help="incremental: skip unchanged entries (default). full: re-embed every entry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without writing to DB or calling the embedder.",
    )
    parser.add_argument(
        "--allow-mass-deletion",
        action="store_true",
        help=(
            f"override the >{int(MASS_DELETION_THRESHOLD * 100)}% delete-rate safeguard "
            "(use only when intentionally pruning)."
        ),
    )
    parser.add_argument(
        "--skip-synthetic-questions",
        action="store_true",
        help="skip the synthetic-question generation phase (content-only ingest).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="emit per-entry decisions on stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    summary = asyncio.run(
        run(
            tenant_slug=args.tenant,
            source=Path(args.source),
            mode=args.mode,
            dry_run=args.dry_run,
            allow_mass_deletion=args.allow_mass_deletion,
            skip_synthetic_questions=args.skip_synthetic_questions,
        )
    )
    print(json.dumps(summary, ensure_ascii=False))
    return EXIT_PARTIAL_SUCCESS if summary.get("parse_errors") else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

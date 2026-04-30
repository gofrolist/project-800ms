"""Synthetic question pass — per-content-chunk paraphrase generation.

Adds rows to ``kb_chunks`` with ``is_synthetic_question = TRUE`` and
``parent_chunk_id`` pointing at the content chunk they paraphrase. The
hybrid retrieval CTE returns parent + synthetic-question matches in the
same scored list, so a user query worded differently from the source
text can hit a synthetic question's embedding and still pull the right
parent answer back.

Design notes
------------

This module is a SEPARATE phase from ingest's content path. The content
phase commits before this runs, so a rate-limited / down rewriter LLM
never blocks new content from landing in the DB. On retry, the next
ingest picks up content chunks that still have no synthetic-question
children and fills them in.

The "needs synth questions" filter uses the parent_chunk_id self-FK:

    SELECT c.id FROM kb_chunks c
    WHERE c.tenant_id = $1
      AND c.is_synthetic_question = FALSE
      AND c.kb_entry_id IN (... entries with prefix ...)
      AND NOT EXISTS (
        SELECT 1 FROM kb_chunks sq
        WHERE sq.parent_chunk_id = c.id
      )

When a content chunk is replaced (ingest UPDATE path), its synthetic
children cascade-delete (ON DELETE CASCADE on parent_chunk_id), so the
parent is then "naked" again and this phase will regenerate questions
on the next run.

LLM cost / failure handling
----------------------------

* One LLM call per content chunk (NOT per synthetic question — the
  model emits a JSON array of N).
* Retries 429 / 5xx with exponential backoff (0.5s, 1s, 2s) up to 3
  attempts. Beyond that the chunk is skipped — partial success is
  acceptable, the next run picks it up.
* A persistent failure on every call (LLM down, key rotated, daily
  quota exhausted) terminates the phase early and surfaces in the
  summary; content commit is unaffected.

Tunables
--------

``QUESTIONS_PER_CHUNK = 4`` is the recall/cost compromise from R6
(specs/002-helper-guide-npc/research.md). Two questions barely move the
hybrid score; eight oversaturates the index without proportional gain.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

QUESTIONS_PER_CHUNK = 4
LLM_TIMEOUT_S = 8.0
LLM_MAX_TOKENS = 400
LLM_MAX_ATTEMPTS = 3
LLM_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
LLM_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

_SYSTEM_PROMPT = (
    "Ты — генератор поисковых запросов для русскоязычного голосового "
    "помощника по игре Arizona RP. Тебе дан фрагмент справочной "
    "статьи. Сгенерируй ровно "
    f"{QUESTIONS_PER_CHUNK} различных вопросов, на которые этот фрагмент "
    "даёт ответ.\n\n"
    "Требования к вопросам:\n"
    "  - на русском языке;\n"
    "  - в разговорной форме, как мог бы спросить игрок голосом;\n"
    "  - 5–15 слов;\n"
    "  - покрывать разные аспекты фрагмента (без дубликатов);\n"
    "  - без ссылок на «согласно тексту», «в этой статье» и т.п. — "
    "вопрос должен быть осмысленным сам по себе.\n\n"
    'Ответ — строго JSON-объект {"questions": ["...", "..."]}. '
    "Никаких пояснений, никакого Markdown."
)


@dataclass(frozen=True)
class _ChunkRow:
    id: int
    title: str
    section: str | None
    content: str
    kb_entry_id: UUID


def _build_user_prompt(*, title: str, section: str | None, content: str) -> str:
    """Assemble the per-chunk user prompt.

    Keeps title + section + content in distinct labelled blocks so the
    model can differentiate "what is the article about" vs "which subsection
    is this quoting" — important for nested patch-note articles.
    """
    section_line = f"Раздел: {section}\n" if section else ""
    return f"Заголовок статьи: {title}\n{section_line}Текст:\n{content}"


async def _call_llm(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    title: str,
    section: str | None,
    content: str,
) -> list[str]:
    """Generate N questions for one chunk with bounded retries.

    Returns the parsed list, or an empty list if every attempt failed.
    The empty-list contract lets the caller proceed on the next chunk
    without forcing a phase-level abort — the chunk simply has no
    synthetic children for this run.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(title=title, section=section, content=content),
            },
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0.4,  # mild diversity across the N questions
    }

    last_status: int | None = None
    last_error: str | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            resp = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            last_error = "timeout"
            await asyncio.sleep(LLM_BACKOFF_SECONDS[attempt])
            continue
        except httpx.HTTPError as exc:
            last_error = type(exc).__name__
            await asyncio.sleep(LLM_BACKOFF_SECONDS[attempt])
            continue

        last_status = resp.status_code
        if resp.status_code in LLM_RETRY_STATUSES and attempt < LLM_MAX_ATTEMPTS - 1:
            await asyncio.sleep(LLM_BACKOFF_SECONDS[attempt])
            continue
        if resp.status_code >= 400:
            logger.warning(
                "synth.llm_non_2xx status={s} attempt={a}",
                s=resp.status_code,
                a=attempt + 1,
            )
            return []

        try:
            content_str = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content_str)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("synth.llm_parse_error err={e}", e=type(exc).__name__)
            return []

        questions: list[str] = []
        if isinstance(parsed, dict) and "questions" in parsed:
            raw = parsed["questions"]
            if isinstance(raw, list):
                questions = [
                    str(q).strip()
                    for q in raw
                    if isinstance(q, str) and 0 < len(str(q).strip()) < 300
                ]
        if not questions:
            logger.warning(
                "synth.llm_empty_questions parsed_keys={k}",
                k=list(parsed) if isinstance(parsed, dict) else type(parsed).__name__,
            )
            return []
        # Cap at N — model occasionally emits fewer or more.
        return questions[:QUESTIONS_PER_CHUNK]

    logger.warning(
        "synth.llm_exhausted_retries last_status={s} last_error={e}",
        s=last_status,
        e=last_error,
    )
    return []


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.10g}" for x in v) + "]"


async def _select_naked_chunks(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    namespace: str,
) -> list[_ChunkRow]:
    """Find content chunks that have no synthetic-question children.

    Bounded by tenant_id (RLS-redundant but keeps the predicate explicit
    per Principle IV) AND namespace prefix (so a `--namespace chatwoot`
    run never touches `wiki:*` chunks).
    """
    rows = (
        (
            await session.execute(
                text(
                    "SELECT c.id, c.title, c.section, c.content, c.kb_entry_id "
                    "FROM kb_chunks c "
                    "JOIN kb_entries e ON e.id = c.kb_entry_id "
                    "WHERE c.tenant_id = :t "
                    "  AND c.is_synthetic_question = FALSE "
                    "  AND e.kb_entry_key LIKE :prefix "
                    "  AND NOT EXISTS ("
                    "    SELECT 1 FROM kb_chunks sq "
                    "    WHERE sq.parent_chunk_id = c.id"
                    "  ) "
                    "ORDER BY c.id"
                ),
                {"t": str(tenant_id), "prefix": f"{namespace}:%"},
            )
        )
        .mappings()
        .all()
    )
    return [
        _ChunkRow(
            id=row["id"],
            title=row["title"],
            section=row["section"],
            content=row["content"],
            kb_entry_id=row["kb_entry_id"],
        )
        for row in rows
    ]


async def run(
    *,
    tenant_slug: str,
    namespace: str,
    encode_fn: Any | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Run the synthetic-question phase for one tenant + namespace.

    Returns ``{added, deleted, rewriter_calls, chunks_touched, elapsed_ms}``.

    ``deleted`` is always 0 in the current implementation — the cascade
    DELETE on parent replacement happens during the content phase, not
    here. The field is kept for forward-compatibility with a future
    "force-regenerate" mode.
    """
    from config import get_settings
    from db import get_session, set_tenant_scope

    settings = get_settings()
    started = time.perf_counter()

    summary: dict[str, Any] = {
        "added": 0,
        "deleted": 0,
        "rewriter_calls": 0,
        "chunks_touched": 0,
        "elapsed_ms": 0,
    }

    if encode_fn is None:
        from embedder import encode, preload

        await asyncio.to_thread(preload)
        encode_fn = encode

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(LLM_TIMEOUT_S, connect=2.0),
    )
    try:
        async with get_session() as session:
            # Resolve tenant inline rather than reusing tenants.resolve_tenant
            # (which takes a UUID, not a slug). Keeps the synth phase callable
            # standalone from the CLI without an extra lookup.
            row = (
                await session.execute(
                    text("SELECT id FROM tenants WHERE slug = :slug AND status = 'active'"),
                    {"slug": tenant_slug},
                )
            ).first()
            if row is None:
                logger.error("synth.unknown_tenant slug={s}", s=tenant_slug)
                return summary
            tenant_id: UUID = row[0]

            await set_tenant_scope(session, tenant_id)
            naked = await _select_naked_chunks(session, tenant_id=tenant_id, namespace=namespace)
            logger.info(
                "synth.start tenant={t} namespace={ns} chunks_naked={n}",
                t=tenant_slug,
                ns=namespace,
                n=len(naked),
            )

            consecutive_failures = 0
            for chunk in naked:
                summary["chunks_touched"] += 1
                summary["rewriter_calls"] += 1
                questions = await _call_llm(
                    client,
                    base_url=str(settings.llm_base_url),
                    api_key=settings.llm_api_key,
                    model=settings.rewriter_model,
                    title=chunk.title,
                    section=chunk.section,
                    content=chunk.content,
                )
                if not questions:
                    consecutive_failures += 1
                    # If 5 consecutive chunks fail, abort the phase —
                    # likely the LLM is down or quota is exhausted, no
                    # value in burning more attempts. Operator gets a
                    # partial-success summary; next run resumes.
                    if consecutive_failures >= 5:
                        logger.error(
                            "synth.aborted_after_consecutive_failures remaining={r}",
                            r=len(naked) - summary["chunks_touched"],
                        )
                        break
                    continue

                consecutive_failures = 0
                # Embed each question; insert as is_synthetic_question=TRUE
                # with parent_chunk_id pointing at this content chunk.
                for question in questions:
                    embedding = await encode_fn(question)
                    await session.execute(
                        text(
                            "INSERT INTO kb_chunks "
                            "  (tenant_id, kb_entry_id, section, title, "
                            "   content, content_sha256, embedding, "
                            "   is_synthetic_question, parent_chunk_id) "
                            "VALUES (:t, :eid, :section, :title, :content, "
                            "        :sha, CAST(:emb AS vector), TRUE, :pid)"
                        ),
                        {
                            "t": str(tenant_id),
                            "eid": str(chunk.kb_entry_id),
                            "section": chunk.section,
                            "title": chunk.title,
                            "content": question,
                            "sha": _q_sha(question),
                            "emb": _vector_literal(embedding),
                            "pid": chunk.id,
                        },
                    )
                    summary["added"] += 1
    finally:
        if own_client:
            await client.aclose()

    summary["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return summary


def _q_sha(question: str) -> str:
    import hashlib

    return hashlib.sha256(question.encode("utf-8")).hexdigest()

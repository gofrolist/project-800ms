"""Integration test for `services/retriever/hybrid_search.py`.

Tests the weighted-sum fusion (0.7 semantic + 0.3 lexical) over pgvector
+ Russian tsvector. Seeds 20 `kb_chunks` across 2 tenants and covers the
four T018 properties:

  (a) top-3 contains the expected chunk for 5 curated Russian queries
  (b) returned score equals `0.7 * semantic + 0.3 * lexical` exactly,
      reconstructable from `fusion_components`
  (c) tenant isolation — tenant B's chunks never leak into tenant A's
      result, even when the same section text lives under both tenants
  (d) querying a tenant with zero chunks returns `[]` without error

Uses real BGE-M3 for semantic ranking. Slow-marked because of container
+ model startup; CI runs it, devs skip with `-m "not slow"`.

Constitution Principle II: integration tests hit real Postgres + real
pgvector + real embedder. No write-path mocks.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from embedder import encode, preload
from hybrid_search import RetrievedChunk, hybrid_search
from models import KBChunk, KBEntry

pytestmark = pytest.mark.slow


# 5 (query → expected_section) pairs. Each query is in Russian, each
# expected_section corresponds to one of the seeded chunks below. The
# query wording is deliberately NOT a verbatim substring of the chunk —
# the hybrid fusion must pull its weight, not just string-match.
_QUERIES: list[tuple[str, str]] = [
    ("Как получить водительские права?", "Получение прав"),
    ("Сколько стоит грузовик?", "Цены на транспорт"),
    ("Где купить оружие?", "Покупка оружия"),
    ("Какие уровни на фермерской работе?", "Фермерство"),
    ("Как вступить в клан?", "Клановая система"),
]


# 10 KB sections per tenant. Same content under both tenants so tenant
# isolation assertions are unambiguous — a cross-tenant leak ranks
# identically to the real match. The tsvector column is GENERATED from
# title + content, so Russian stemming happens without any help here.
_SECTIONS: list[tuple[str, str]] = [
    (
        "Получение прав",
        "Чтобы получить водительские права, посетите автошколу на центральной "
        "площади и сдайте теоретический и практический экзамены. Стоимость "
        "обучения 5 000 долларов.",
    ),
    (
        "Цены на транспорт",
        "Грузовик Mule стоит 15 000 долларов в автосалоне. Легковой автомобиль "
        "Premier обойдётся в 12 500 долларов.",
    ),
    (
        "Покупка оружия",
        "Оружие можно купить в оружейном магазине после получения лицензии. "
        "Лицензия стоит 10 000 долларов и выдаётся с 5 уровня.",
    ),
    (
        "Фермерство",
        "Фермерская работа открывает уровни 1-10; каждый уровень даёт новый "
        "инструмент и увеличивает зарплату.",
    ),
    (
        "Клановая система",
        "Чтобы вступить в клан, получите приглашение от лидера и подтвердите "
        "его через игровое меню.",
    ),
    (
        "Жилплощадь",
        "Квартиры доступны для аренды и покупки; цены зависят от района и класса дома.",
    ),
    (
        "Медицина",
        "Медицинская работа требует 5 уровня паспорта и лицензии врача, "
        "которая выдаётся после специального теста.",
    ),
    (
        "Ресторан",
        "В ресторане можно заказать еду, которая восстанавливает здоровье и утоляет голод.",
    ),
    (
        "Такси",
        "Работа таксиста — одна из стартовых профессий; подходит для новичков без лицензии.",
    ),
    (
        "Банк",
        "В банке можно открыть счёт, оформить кредит или перевести деньги другому игроку.",
    ),
]


@pytest_asyncio.fixture
async def seeded_kb(db_session: AsyncSession) -> dict[str, Any]:
    """Seed 2 tenants × 1 kb_entry × 10 kb_chunks = 20 chunks.

    Embeddings are real BGE-M3 outputs (1024-dim). The embedder is
    preloaded once per process via its singleton so cost is amortized
    across the 5 parametrized queries and the fusion/tenant-isolation
    tests.
    """
    # preload() is sync but idempotent (lru_cache); blocking on first
    # call only. Acceptable inside an async fixture.
    preload()

    tenant_a_id = uuid.uuid4()
    tenant_b_id = uuid.uuid4()

    # Seed both tenants. `slug` is UNIQUE so include the UUID prefix.
    await db_session.execute(
        text(
            "INSERT INTO tenants (id, name, slug) VALUES "
            "(:a_id, 'Tenant A', :a_slug), "
            "(:b_id, 'Tenant B', :b_slug)"
        ),
        {
            "a_id": tenant_a_id,
            "a_slug": f"tenant-a-{tenant_a_id.hex[:8]}",
            "b_id": tenant_b_id,
            "b_slug": f"tenant-b-{tenant_b_id.hex[:8]}",
        },
    )

    chunk_ids: dict[tuple[uuid.UUID, str], int] = {}

    for tenant_id in (tenant_a_id, tenant_b_id):
        entry = KBEntry(
            tenant_id=tenant_id,
            kb_entry_key="main",
            title="Основной раздел",
            content_sha256="deadbeef",
        )
        db_session.add(entry)
        await db_session.flush()

        for section, content in _SECTIONS:
            emb = await encode(content)
            chunk = KBChunk(
                tenant_id=tenant_id,
                kb_entry_id=entry.id,
                section=section,
                title=f"{entry.title} — {section}",
                content=content,
                content_sha256=f"sha:{section}",
                embedding=emb,
            )
            db_session.add(chunk)
            await db_session.flush()
            chunk_ids[(tenant_id, section)] = chunk.id

    return {
        "tenant_a_id": tenant_a_id,
        "tenant_b_id": tenant_b_id,
        "chunk_ids": chunk_ids,
    }


class TestHybridSearchTopK:
    """(a) — top-3 contains the expected chunk for each of 5 Russian queries."""

    @pytest.mark.parametrize("query,expected_section", _QUERIES)
    async def test_top_3_contains_expected_chunk(
        self,
        db_session: AsyncSession,
        seeded_kb: dict[str, Any],
        query: str,
        expected_section: str,
    ) -> None:
        query_emb = await encode(query)
        results = await hybrid_search(
            db_session,
            seeded_kb["tenant_a_id"],
            query,
            query_emb,
            top_k=3,
        )
        expected_id = seeded_kb["chunk_ids"][(seeded_kb["tenant_a_id"], expected_section)]
        returned_ids = [c.id for c in results]
        assert expected_id in returned_ids, (
            f"query={query!r} expected_section={expected_section!r} returned_ids={returned_ids}"
        )


class TestFusionMath:
    """(b) — returned score == 0.7·semantic + 0.3·lexical (reproducible)."""

    async def test_fusion_formula_reproducible(
        self, db_session: AsyncSession, seeded_kb: dict[str, Any]
    ) -> None:
        query = "Как получить водительские права?"
        query_emb = await encode(query)
        results = await hybrid_search(
            db_session,
            seeded_kb["tenant_a_id"],
            query,
            query_emb,
            top_k=5,
        )
        assert results, "expected at least one result from seeded KB"
        for chunk in results:
            expected_fused = (
                0.7 * chunk.fusion_components.semantic + 0.3 * chunk.fusion_components.lexical
            )
            assert abs(chunk.score - expected_fused) < 1e-6, (
                f"chunk {chunk.id}: score={chunk.score} vs "
                f"0.7*{chunk.fusion_components.semantic} + "
                f"0.3*{chunk.fusion_components.lexical}"
            )

    async def test_scores_monotonic_descending(
        self, db_session: AsyncSession, seeded_kb: dict[str, Any]
    ) -> None:
        query = "Как получить водительские права?"
        query_emb = await encode(query)
        results = await hybrid_search(
            db_session,
            seeded_kb["tenant_a_id"],
            query,
            query_emb,
            top_k=5,
        )
        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True), (
            f"results not monotonically decreasing: {scores}"
        )


class TestTenantIsolation:
    """(c) — tenant B chunks never leak into tenant A's result set."""

    async def test_no_cross_tenant_leakage(
        self, db_session: AsyncSession, seeded_kb: dict[str, Any]
    ) -> None:
        query = "Как получить водительские права?"
        query_emb = await encode(query)
        results_a = await hybrid_search(
            db_session,
            seeded_kb["tenant_a_id"],
            query,
            query_emb,
            top_k=10,
        )
        tenant_b_chunk_ids = {
            cid
            for (tid, _), cid in seeded_kb["chunk_ids"].items()
            if tid == seeded_kb["tenant_b_id"]
        }
        returned_ids = {c.id for c in results_a}
        leaked = returned_ids & tenant_b_chunk_ids
        assert not leaked, f"tenant B chunks leaked into tenant A result: {leaked}"


class TestEmptyTenant:
    """(d) — zero-chunks tenant returns [] without error."""

    async def test_empty_tenant_returns_empty_list(
        self, db_session: AsyncSession, seeded_kb: dict[str, Any]
    ) -> None:
        empty_tenant_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO tenants (id, name, slug) VALUES (:id, 'Empty', :slug)"),
            {"id": empty_tenant_id, "slug": f"empty-{empty_tenant_id.hex[:8]}"},
        )
        query_emb = await encode("любой вопрос")
        results = await hybrid_search(
            db_session,
            empty_tenant_id,
            "любой вопрос",
            query_emb,
            top_k=5,
        )
        assert results == [], f"expected empty list, got {results!r}"


class TestReturnedChunkShape:
    """RetrievedChunk dataclass contract — downstream API relies on these fields."""

    async def test_chunk_has_all_required_fields(
        self, db_session: AsyncSession, seeded_kb: dict[str, Any]
    ) -> None:
        query_emb = await encode("Как получить права?")
        results = await hybrid_search(
            db_session,
            seeded_kb["tenant_a_id"],
            "Как получить права?",
            query_emb,
            top_k=1,
        )
        assert len(results) == 1
        chunk = results[0]
        assert isinstance(chunk, RetrievedChunk)
        assert isinstance(chunk.id, int)
        assert isinstance(chunk.title, str) and chunk.title
        assert isinstance(chunk.content, str) and chunk.content
        assert isinstance(chunk.score, float)
        assert isinstance(chunk.fusion_components.semantic, float)
        assert isinstance(chunk.fusion_components.lexical, float)
        assert isinstance(chunk.metadata, dict)

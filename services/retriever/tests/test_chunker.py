"""Tests for the markdown-aware KB chunker.

Pure-Python tests — no Postgres, no embedder — so they run fast and
gate every commit. The chunker is deterministic given input; assertions
target the structural contract (one chunk per H2/H3, overflow handling,
image stripping, headingless fallback) rather than exact byte counts.
"""

from __future__ import annotations

import pytest

from chunker import (
    MAX_CHARS,
    MIN_CHARS,
    Chunk,
    chunk_article,
)


def _content(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


def test_no_headings_returns_single_chunk_with_none_section() -> None:
    body = _content(
        "Это короткая статья без подзаголовков.",
        "Второй абзац с дополнительной информацией.",
    )
    chunks = chunk_article(title="Тест", content=body)
    assert len(chunks) == 1
    assert chunks[0].section is None
    assert "короткая статья" in chunks[0].content


def test_h2_split_emits_one_chunk_per_section() -> None:
    body = (
        "## Раздел один\n\n"
        "Первый раздел про получение прав.\n\n"
        "## Раздел два\n\n"
        "Второй раздел про цены.\n"
    )
    chunks = chunk_article(title="Тест", content=body)
    sections = [c.section for c in chunks]
    assert sections == ["Раздел один", "Раздел два"]
    assert "получение прав" in chunks[0].content
    assert "цены" in chunks[1].content


def test_h3_section_path_includes_parent_h2() -> None:
    body = (
        "## Линия наград\n\n"
        "Введение в линию наград.\n\n"
        "### Аксессуары\n\n"
        "Подробности про аксессуары.\n"
    )
    chunks = chunk_article(title="Тест", content=body)
    sections = [c.section for c in chunks]
    assert sections == ["Линия наград", "Линия наград > Аксессуары"]


def test_orphan_h3_without_parent_h2_uses_h3_alone() -> None:
    body = "### Первый подраздел\n\nТекст подраздела.\n"
    chunks = chunk_article(title="Тест", content=body)
    assert chunks[0].section == "Первый подраздел"


def test_preamble_before_first_heading_is_emitted_with_none_section() -> None:
    body = (
        "Введение перед первым подзаголовком статьи.\n\n"
        "## Раздел\n\n"
        "Содержимое самого раздела статьи.\n"
    )
    chunks = chunk_article(title="Тест", content=body)
    sections = [c.section for c in chunks]
    # Preamble with section=None first, then the H2 section. has_headings
    # is true so the H2 section keeps its name.
    assert sections == [None, "Раздел"]


def test_long_section_splits_with_part_suffix() -> None:
    long_para = "Очень длинный абзац. " * (MAX_CHARS // 20 + 50)
    body = f"## Большой раздел\n\n{long_para}\n"
    chunks = chunk_article(title="Тест", content=body)
    assert len(chunks) >= 2, "expected overflow split into multiple chunks"
    # First chunk keeps the bare section name; subsequent chunks carry a
    # part suffix to satisfy the (tenant, entry, section) unique key.
    assert chunks[0].section == "Большой раздел"
    for c in chunks[1:]:
        assert c.section is not None and c.section.startswith("Большой раздел #")
    for c in chunks:
        assert len(c.content) <= MAX_CHARS + 1, "chunks must respect MAX_CHARS cap"


def test_very_long_paragraph_falls_back_to_char_slicing() -> None:
    # No paragraph breaks at all — chunker must character-slice.
    huge = "слово " * (MAX_CHARS * 2 // 6)
    body = f"## Стена текста\n\n{huge}\n"
    chunks = chunk_article(title="Тест", content=body)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.content) <= MAX_CHARS + 1


def test_image_with_alt_text_replaced_with_placeholder() -> None:
    body = (
        "## Скриншот\n\n"
        "![Линия наград](https://chatwoot.example/blob/abc123)\n\n"
        "Описание скриншота.\n"
    )
    chunks = chunk_article(title="Тест", content=body)
    text = "\n".join(c.content for c in chunks)
    assert "[image: Линия наград]" in text
    assert "blob/abc123" not in text, "raw blob URL must be stripped"


def test_image_with_empty_alt_is_dropped() -> None:
    body = "## Картинка\n\n![](https://chatwoot.example/blob/empty)\n\nТекст после картинки.\n"
    chunks = chunk_article(title="Тест", content=body)
    text = "\n".join(c.content for c in chunks)
    assert "blob/empty" not in text
    assert "Текст после" in text


def test_empty_content_returns_no_chunks() -> None:
    assert chunk_article(title="Пустая", content="") == []
    assert chunk_article(title="Пустая", content="   \n  \n") == []


def test_headingless_overflow_uses_synthetic_section_keys() -> None:
    long_text = ("Длинный абзац без подзаголовков. " * (MAX_CHARS // 30 + 40)).strip()
    chunks = chunk_article(title="Без заголовков", content=long_text)
    assert len(chunks) >= 2
    # First chunk has section=None; subsequent overflow chunks get
    # synthetic "#N" labels so the (tenant, entry, section) unique key
    # still discriminates them.
    assert chunks[0].section is None
    for c in chunks[1:]:
        assert c.section is not None and c.section.startswith("#")


def test_empty_section_body_is_dropped() -> None:
    body = "## Только заголовок\n\n## Реальный раздел\n\nРеальное содержимое.\n"
    chunks = chunk_article(title="Тест", content=body)
    sections = [c.section for c in chunks]
    assert sections == ["Реальный раздел"]


def test_short_section_below_min_chars_is_kept_when_only_chunk() -> None:
    # Tiny content — chunker must still emit at least one chunk so the
    # operator dashboard sees the article rather than silently skipping
    # it. The "drop below MIN_CHARS" filter is best-effort and yields
    # to the no-empty-output invariant.
    tiny = "## H\n\nкр.\n"
    chunks = chunk_article(title="Тест", content=tiny)
    assert len(chunks) == 1
    assert chunks[0].content


def test_returned_objects_are_chunk_instances() -> None:
    body = "## A\n\nБ.\n\n## B\n\nГ.\n"
    chunks = chunk_article(title="T", content=body)
    assert all(isinstance(c, Chunk) for c in chunks)
    # Frozen dataclass — mutation must fail.
    with pytest.raises(Exception):  # noqa: BLE001 — FrozenInstanceError
        chunks[0].section = "x"  # type: ignore[misc]


def test_min_chars_constant_is_below_max() -> None:
    # Sanity check on the tunables — guards against an accidental edit
    # that would make MIN >= MAX and produce zero-chunk output forever.
    assert 0 < MIN_CHARS < MAX_CHARS


def test_paragraph_overlap_does_not_exceed_max_chars() -> None:
    """Regression for an overflow path in `_split_long`. Earlier code
    seeded the next chunk with `tail (≤OVERLAP_CHARS) + "\\n\\n" + para`,
    where `para` could be up to MAX_CHARS-1. The combined size could
    reach OVERLAP_CHARS + 2 + (MAX_CHARS - 1) chars, busting the cap.

    Reproduces the bound violation by sequencing paragraphs of size
    ~800/1400/100 chars: first flush emits 800-char chunk, leaves an
    overlap tail; second paragraph (~1400) joined with the tail would
    produce ~1550 chars, exceeding MAX_CHARS=1500. After the fix, the
    overlap is dropped when it would push past the cap.
    """
    para_a = "а" * 800  # short of MAX_CHARS — will pack into chunk 1
    para_b = "б" * 1400  # large enough that overlap+para overflows
    para_c = "в" * 100  # tail content
    body = f"## Раздел\n\n{para_a}\n\n{para_b}\n\n{para_c}\n"
    chunks = chunk_article(title="Тест", content=body)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.content) <= MAX_CHARS, (
            f"chunk exceeds MAX_CHARS={MAX_CHARS}: {len(c.content)} chars"
        )


def test_repeated_section_names_get_unique_suffixes() -> None:
    """Articles with repeated H2 (boilerplate footer-style headings,
    common in Chatwoot help-base — e.g. ``"🆘 Не нашли решения?"``
    appearing at the bottom of multiple sections) used to violate the
    ``(tenant_id, kb_entry_id, section) NULLS NOT DISTINCT`` unique
    index. Repeated sections now get ``" #2"``, ``" #3"`` suffixes
    so each chunk has a distinct DB key.

    Concrete failure: article 17 in the live Chatwoot feed has
    ``"🆘 Не нашли решения?"`` three times; the ingest UPSERT raised
    ``UniqueViolationError`` and rolled back the entire run.
    """
    body = (
        "## Раздел A\n\nСодержимое первого раздела статьи.\n\n"
        "## 🆘 Не нашли решения?\n\nКонтакт первый, обратитесь к нам.\n\n"
        "## Раздел B\n\nСодержимое второго раздела статьи.\n\n"
        "## 🆘 Не нашли решения?\n\nКонтакт второй, повторим ссылку.\n\n"
        "## Раздел C\n\nСодержимое третьего раздела статьи.\n\n"
        "## 🆘 Не нашли решения?\n\nКонтакт третий, последний.\n"
    )
    chunks = chunk_article(title="Тест", content=body)
    sections = [c.section for c in chunks]
    # Hard DB invariant — all section keys within an article must be
    # distinct.
    assert len(set(sections)) == len(sections), f"duplicate sections produced: {sections}"
    # The first repeat keeps the bare name; subsequent ones get suffixes.
    repeats = [s for s in sections if s and "Не нашли" in s]
    assert repeats[0] == "🆘 Не нашли решения?"
    assert repeats[1] == "🆘 Не нашли решения? #2"
    assert repeats[2] == "🆘 Не нашли решения? #3"

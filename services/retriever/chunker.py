"""Markdown-aware chunker for KB ingestion.

Splits a Russian Markdown article into retrievable chunks. The unique key
on ``kb_chunks`` is ``(tenant_id, kb_entry_id, section)``, so the chunker
must produce a stable, unique ``section`` per chunk of the same entry. We
use the heading path joined with " > " (e.g. ``"Линия наград > Аксессуары"``)
and disambiguate within-section overflow with a numeric suffix.

The Chatwoot help-base feed is the immediate consumer. Articles look like::

    # Title
    [intro paragraphs]

    ## H2 section
    [paragraphs, lists, images]

    ### H3 subsection
    [...]

The chunker:

1. Strips image references and bare placeholder URLs that would otherwise
   smear the embedding with low-information tokens.
2. Splits on H2 / H3 headings — H1 is treated as the article title and
   not a section break (only one H1 expected).
3. Within a section, if content exceeds ``MAX_CHARS``, splits further on
   paragraph boundaries with overlap.
4. Falls back to a single ``section=None`` chunk when the article has
   no H2 / H3 structure.

Empty sections (just headings, no body) are dropped — embedding a heading
on its own dilutes the index without adding recall.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Tunable size constants. Picked for BGE-M3 (8192-token cap, ~5K Russian
# chars) with comfortable headroom and enough context per chunk to support
# multi-fact answers. Overlap stitches paragraphs together so a question
# whose answer straddles a chunk boundary still hits at least one chunk.
MAX_CHARS = 1500
OVERLAP_CHARS = 150
# Below this, a chunk almost certainly carries no useful retrieval signal
# (a stray "Note:" leftover, a single short sentence after image-stripping).
# Set to a length that still admits a meaningful Russian fact ("Цена 5000$.")
# but rejects fragments. The "always emit at least one chunk" fallback in
# `chunk_article` overrides this for tiny-but-non-empty articles.
MIN_CHARS = 20

# H2 / H3 heading detection. H1 is reserved for the article title and not
# treated as a section break. We deliberately do NOT match H4+ — deeper
# nesting in Chatwoot articles is rare and treating it as a leaf section
# blows up chunk count without recall benefit.
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)

# Image / blob noise filter. Chatwoot exports rails active_storage redirect
# URLs (200+ chars of opaque base64) which contribute no semantic signal
# to retrieval. Replace ``![alt](url)`` with ``[image: alt]`` when alt-text
# carries information; drop entirely when alt is empty.
_IMG_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
# Bare ``<https://...>`` autolinks — useful as anchor text when the URL is
# meaningful to the user, but Chatwoot uses this for raw video links that
# add no recall signal. Keep as-is; the hybrid lexical index can still
# match the URL token if a user actually asks "what's the YouTube link".


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit, before embedding.

    ``section`` is ``None`` only when the article has no H2/H3 headings
    (a flat-prose article). The DB unique key ``(tenant_id, kb_entry_id,
    section) NULLS NOT DISTINCT`` guarantees at most one such row per
    entry — the chunker enforces this by emitting a single chunk for
    headingless articles.
    """

    section: str | None
    content: str


def _strip_images(text: str) -> str:
    """Replace ``![alt](url)`` with ``[image: alt]`` (or drop empty-alt)."""

    def _repl(match: re.Match[str]) -> str:
        alt = match.group(1).strip()
        return f"[image: {alt}]" if alt else ""

    return _IMG_RE.sub(_repl, text)


def _normalise(text: str) -> str:
    """Collapse runs of blank lines, trim trailing whitespace per line.

    Also strips H1 lines: the article title is carried separately on
    ``kb_entries.title``, and including the H1 in chunk content would
    double-weight the title in retrieval (lexical tsvector AND embedding
    AND title column all touch it).
    """
    text = _strip_images(text)
    # Drop H1 lines entirely — Chatwoot articles usually have one H1
    # paraphrasing the article title, so it's pure duplication for our
    # purposes. Multiple H1s are rare; strip them all defensively.
    text = re.sub(r"^#\s+.*$", "", text, flags=re.MULTILINE)
    # Trim trailing whitespace on every line so a Chatwoot edit that only
    # adds/removes spaces at line ends doesn't trigger a re-embed.
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines down to 2 — preserves paragraph
    # breaks while killing the formatting noise rendered Markdown emits.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(content: str, *, section: str | None) -> Iterable[Chunk]:
    """Split a long section into MAX_CHARS-sized pieces with overlap.

    Splits on paragraph boundaries (blank lines) when possible to keep
    each chunk semantically intact. If a single paragraph already
    exceeds MAX_CHARS, it gets character-sliced with overlap — rare in
    practice (Chatwoot articles use plenty of structure) but the safety
    net is required.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if not paragraphs:
        return

    # Greedy pack paragraphs into chunks until adding the next paragraph
    # would exceed MAX_CHARS. Overlap is a tail of the previous chunk
    # (last OVERLAP_CHARS chars), prepended to the next.
    current = ""
    part = 1
    for para in paragraphs:
        # Single paragraph exceeds the cap — character-slice it.
        if len(para) > MAX_CHARS:
            if current:
                yield Chunk(_section_with_part(section, part), current.strip())
                part += 1
                current = ""
            for slice_text in _char_slices(para):
                yield Chunk(_section_with_part(section, part), slice_text)
                part += 1
            continue

        # Adding this paragraph fits — extend current.
        if len(current) + len(para) + 2 <= MAX_CHARS:
            current = (current + "\n\n" + para) if current else para
            continue

        # Otherwise, flush current and start a new chunk seeded with overlap.
        if current:
            yield Chunk(_section_with_part(section, part), current.strip())
            part += 1
            tail = current[-OVERLAP_CHARS:].lstrip()
            # Avoid splitting mid-word in the overlap by trimming back to
            # the next whitespace if possible.
            ws = tail.find(" ")
            if 0 < ws < OVERLAP_CHARS // 2:
                tail = tail[ws + 1 :]
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = para

    if current and len(current.strip()) >= MIN_CHARS:
        yield Chunk(_section_with_part(section, part), current.strip())


def _char_slices(text: str) -> Iterable[str]:
    """Last-resort slicer for paragraphs that exceed MAX_CHARS on their own."""
    step = MAX_CHARS - OVERLAP_CHARS
    pos = 0
    while pos < len(text):
        end = min(pos + MAX_CHARS, len(text))
        yield text[pos:end].strip()
        if end == len(text):
            break
        pos += step


def _section_with_part(section: str | None, part: int) -> str | None:
    """Append ``" #N"`` to disambiguate overflow chunks within one section.

    The first chunk keeps the bare section (or ``None``) so the unique
    key ``(tenant_id, kb_entry_id, section) NULLS NOT DISTINCT`` admits
    exactly one row per article-section. Subsequent overflow parts get a
    synthetic ``"#N"`` (or ``"<section> #N"``) suffix so they don't
    collide on the unique key.
    """
    if part == 1:
        return section
    if section is None:
        return f"#{part}"
    return f"{section} #{part}"


def _split_by_headings(text: str) -> list[tuple[str | None, str]]:
    """Split content on H2 / H3 boundaries.

    Returns a list of ``(section_path, body)`` pairs in document order.
    Content before the first heading is emitted with ``section=None`` —
    callers treat that as the article preamble.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    out: list[tuple[str | None, str]] = []
    # Preamble (before the first heading).
    first = matches[0]
    preamble = text[: first.start()].strip()
    if preamble:
        out.append((None, preamble))

    # Track the current H2 so an H3 can be reported as "H2 > H3".
    current_h2: str | None = None
    for i, match in enumerate(matches):
        level = len(match.group(1))  # 2 or 3
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if level == 2:
            current_h2 = title
            section = title
        else:  # level == 3
            section = f"{current_h2} > {title}" if current_h2 else title

        if body:
            out.append((section, body))
    return out


def chunk_article(*, title: str, content: str) -> list[Chunk]:
    """Chunk a single article into retrievable pieces.

    Args:
        title: The article title (H1). Reused for every chunk's
            ``kb_chunks.title`` column at INSERT time, but NOT prepended
            to the chunk content here — keeping it out lets the lexical
            index reflect only the chunk's actual content while the
            generated tsvector concatenates ``title`` + ``content`` at
            the SQL layer (see migration 0004).
        content: The full article body, Markdown.

    Returns:
        A list of ``Chunk`` instances. Always non-empty for non-trivial
        input; an article whose normalised body is shorter than
        ``MIN_CHARS`` returns a single chunk anyway (rather than
        skipping the entry entirely) so the operator sees something
        for every kb_entry — silent zero-chunk articles would surprise
        on the dashboard.
    """
    del title  # see docstring — caller stamps title on the kb_chunks row.
    body = _normalise(content)
    if not body:
        return []

    chunks: list[Chunk] = []
    sections = _split_by_headings(body)
    has_headings = any(section is not None for section, _ in sections)

    for section, section_body in sections:
        if len(section_body) <= MAX_CHARS:
            chunks.append(Chunk(section if has_headings else None, section_body))
        else:
            chunks.extend(_split_long(section_body, section=section if has_headings else None))

    # Drop chunks too small to carry signal, but never return an empty
    # list for an article that had any content — keep at least the
    # longest chunk as the article's representation.
    survivors = [c for c in chunks if len(c.content) >= MIN_CHARS]
    if survivors:
        return survivors
    return [max(chunks, key=lambda c: len(c.content))] if chunks else []

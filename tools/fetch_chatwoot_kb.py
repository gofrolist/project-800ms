"""Fetch the Chatwoot help-base feed and write per-article JSON files.

Usage::

    CHATWOOT_HELP_BASE_TOKEN=<bearer> \\
        uv run python tools/fetch_chatwoot_kb.py --project arizona

Output layout::

    data/kb/<project>/
        <id>.json          # one per article (normalised shape)
        _manifest.json     # {fetched_at, count, ids, project, base_url}

Each article file contains the canonical shape consumed by the (still
to-be-implemented) ``services/retriever/ingest.py`` CLI:

    {
      "kb_entry_key": "chatwoot:<id>",
      "title": "...",
      "content": "...markdown...",
      "source_uri": "https://chatwoot.arizona-rp.com/api-help-base/get?project=<p>&id=<id>",
      "metadata": {
        "category_name": "...",
        "portal_name":   "...",
        "description":   null,
        "source":        "chatwoot",
        "fetched_at":    "<rfc3339>"
      }
    }

The Chatwoot endpoint returns the entire feed in one response (no
pagination today), so this is a single-request fetch. Writes are
atomic (``.tmp`` + ``rename``) so a partial run never leaves a
half-written file in the working tree.

Exit codes
----------
* ``0`` — fetched and wrote the feed (or dry-run completed).
* ``1`` — auth/network/parse error. No partial writes committed.
* ``64`` — bad CLI args (missing project, bad out-dir, etc.).

The script intentionally uses only the stdlib so it can run on any
Python 3.10+ host without a venv (e.g. on the VM directly when running
the fetch out-of-band). This is a deliberate exception to the project's
loguru-everywhere convention (CLAUDE.md): loguru is a third-party
dependency and would break the stdlib-only constraint. The
``logging.getLogger`` calls below use stdlib ``%``-style lazy
formatting, which is the equivalent discipline for the stdlib logger.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

# stderr-bound logger so the stdout summary line stays a single,
# machine-parseable JSON document.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("fetch_chatwoot_kb")

DEFAULT_BASE_URL = "https://chatwoot.arizona-rp.com"
DEFAULT_OUT_ROOT = "data/kb"
HTTP_TIMEOUT_S = 30.0
# Article ids are treated as strings on input; sanitise anything outside
# this safelist before using one as a filename. Real ids today are
# integer-valued strings ("116"), but the API contract does not
# guarantee that.
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9._-]")


@dataclass(frozen=True)
class FetchConfig:
    project: str
    out_dir: Path
    base_url: str
    token: str
    dry_run: bool


def _safe_id(raw: str) -> str:
    """Make an article id safe for use as a filename component.

    Replaces every char outside ``[A-Za-z0-9._-]`` with ``_``.
    Also rejects the empty string, ``.``, and ``..`` to avoid any
    chance of writing outside ``out_dir``.
    """

    cleaned = _SAFE_ID_RE.sub("_", raw)
    if cleaned in {"", ".", ".."}:
        raise ValueError(f"unsafe article id: {raw!r}")
    return cleaned


def _http_get_json(url: str, token: str) -> dict[str, Any]:
    """GET ``url`` with the bearer token, parse JSON, return the body.

    Uses the verbatim header form documented by the Chatwoot endpoint
    (``authorization: <token>``, no ``Bearer`` prefix).
    """

    req = urllib.request.Request(  # noqa: S310 — explicit https URL
        url,
        method="GET",
        headers={
            "authorization": token,
            "accept": "application/json",
            "user-agent": "project-800ms-fetch-chatwoot-kb/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
            raw = resp.read()
    except urllib.error.HTTPError as err:
        # Surface response body to ease debugging (auth/quota messages
        # land here). Do not echo the request token.
        body = err.read().decode("utf-8", errors="replace") if err.fp else ""
        raise SystemExit(f"HTTP {err.code} from {url}: {body[:500]}") from None
    except urllib.error.URLError as err:
        raise SystemExit(f"network error fetching {url}: {err.reason}") from None

    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        raise SystemExit(f"non-JSON response from {url}: {err}") from None


def _normalise(
    article: dict[str, Any], project: str, base_url: str, fetched_at: str
) -> dict[str, Any]:
    """Project a Chatwoot article into the canonical KB-entry shape.

    Drops every field not listed in ``metadata`` so changes upstream
    (new fields) don't silently flow into the ingestion pipeline
    without review.
    """

    raw_id = str(article["id"])
    safe = _safe_id(raw_id)
    source_uri = f"{base_url.rstrip('/')}/api-help-base/get?" + urlencode(
        {"project": project, "id": raw_id}
    )
    return {
        "kb_entry_key": f"chatwoot:{safe}",
        "title": article.get("title") or "",
        "content": article.get("content") or "",
        "source_uri": source_uri,
        "metadata": {
            "category_name": article.get("category_name"),
            "portal_name": article.get("portal_name"),
            "description": article.get("description"),
            "source": "chatwoot",
            "fetched_at": fetched_at,
            "raw_id": raw_id,
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` via a tmp-rename so partial writes
    never leave a half-baked file behind.
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def fetch(config: FetchConfig) -> dict[str, Any]:
    """Fetch the feed and write it to disk. Returns the summary dict.

    Raises ``SystemExit`` on any unrecoverable error.
    """

    started = time.perf_counter()
    feed_url = f"{config.base_url.rstrip('/')}/api-help-base/get?" + urlencode(
        {"project": config.project}
    )
    log.info("fetching feed url=%s", feed_url)
    body = _http_get_json(feed_url, config.token)

    if not body.get("success"):
        raise SystemExit(f"feed reported success=false: {body}")
    articles = body.get("data") or []
    if not isinstance(articles, list):
        raise SystemExit(f"expected data: list, got {type(articles).__name__}")

    fetched_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    seen_ids: list[str] = []
    seen_set: set[str] = set()  # dedup so duplicate upstream IDs don't
    # inflate manifest count.
    for article in articles:
        if "id" not in article:
            log.warning("skipping article without id: %r", article.get("title"))
            continue
        safe = _safe_id(str(article["id"]))
        if safe in seen_set:
            # Duplicate id from upstream — the second copy would
            # overwrite the first via tmp+rename, leaving exactly one
            # file on disk. Reflect that on the manifest by NOT
            # double-counting; warn so the operator can chase the
            # upstream data-quality issue.
            log.warning("duplicate article id from upstream: %r", safe)
            continue
        seen_set.add(safe)
        normalised = _normalise(article, config.project, config.base_url, fetched_at)
        seen_ids.append(safe)
        target = config.out_dir / f"{safe}.json"
        if config.dry_run:
            log.info(
                "[dry-run] would write %s (%d bytes content)",
                target,
                len(normalised["content"]),
            )
        else:
            _atomic_write_json(target, normalised)
            written.append(str(target))

    manifest = {
        "project": config.project,
        "base_url": config.base_url,
        "fetched_at": fetched_at,
        "count": len(seen_ids),
        "ids": sorted(seen_ids),
    }
    if not config.dry_run:
        _atomic_write_json(config.out_dir / "_manifest.json", manifest)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    summary = {
        "project": config.project,
        "out_dir": str(config.out_dir),
        "fetched": len(seen_ids),
        "written": len(written),
        "dry_run": config.dry_run,
        "elapsed_ms": elapsed_ms,
    }
    return summary


def _build_config(args: argparse.Namespace) -> FetchConfig:
    token = os.environ.get("CHATWOOT_HELP_BASE_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "CHATWOOT_HELP_BASE_TOKEN is empty — set it in the environment "
            "(local: source infra/.env; VM: it is written by the startup "
            "script when chatwoot_help_base_token is set in terraform.tfvars)."
        )
    project = args.project.strip()
    if not project or "/" in project:
        # exit code 64: bad CLI args (matches the ingest CLI contract).
        log.error("invalid --project value: %r", args.project)
        raise SystemExit(64)
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT_ROOT) / project
    return FetchConfig(
        project=project,
        out_dir=out_dir,
        base_url=args.base_url,
        token=token,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Chatwoot help-base articles and write per-article JSON files.",
    )
    parser.add_argument(
        "--project",
        default="arizona",
        help="Chatwoot project slug (default: arizona).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Destination directory (default: data/kb/<project>/).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Chatwoot base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report counts, but do not write any files.",
    )
    args = parser.parse_args(argv)

    config = _build_config(args)
    summary = fetch(config)
    # Single-line JSON summary on stdout (parseable by ops tooling).
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

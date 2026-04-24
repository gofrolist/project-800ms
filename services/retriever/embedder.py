"""BGE-M3 embedder — preloaded singleton.

Mirrors `services/agent/models.py::load_gigaam`: the heavy model loads
once at process start, and every subsequent `encode()` reuses the cached
instance. Preloading at boot (not lazily on first request) is mandatory —
constitution Principle V; research R14.

BGE-M3 outputs 1024-dim dense vectors. No `"query:"` / `"passage:"` prefix
discipline (research R1) — callers encode text directly on either side.
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

from config import get_settings

if TYPE_CHECKING:
    # Kept under TYPE_CHECKING so unit tests that stub out the embedder
    # don't have to install sentence-transformers just to import this
    # module.
    from sentence_transformers import SentenceTransformer


# Module-level flag so /ready can probe without depending on lru_cache's
# internal API shape. Flipped on first successful load.
_loaded: bool = False


@lru_cache(maxsize=1)
def _load_bge_m3() -> "SentenceTransformer":
    """Load BGE-M3 once and cache it process-wide.

    Fails hard on load errors — a retriever without an embedder has
    nothing useful to do, and we do not want to serve `/retrieve` requests
    against a stub or NaN vectors.

    lru_cache's internal lock already serialises cache misses, so no
    separate threading.Lock is needed here.
    """
    from sentence_transformers import SentenceTransformer

    global _loaded
    settings = get_settings()
    start = time.perf_counter()
    logger.info(
        "embedder.loading model={model} device={device}",
        model=settings.embedder_model,
        device=settings.embedder_device,
    )
    model = SentenceTransformer(
        settings.embedder_model,
        device=settings.embedder_device,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "embedder.loaded model={model} dim={dim} elapsed_ms={elapsed_ms}",
        model=settings.embedder_model,
        dim=settings.embedder_dim,
        elapsed_ms=elapsed_ms,
    )
    _loaded = True
    return model


def preload() -> None:
    """Call once at service boot. Forces the singleton to exist so the
    first `/retrieve` doesn't pay the load cost. Idempotent — cached via
    `_load_bge_m3`'s lru_cache."""
    _load_bge_m3()


def is_loaded() -> bool:
    """Cheap probe for `/ready` — true after `preload()` has completed
    at least once without raising."""
    return _loaded


def _encode_sync(text_input: str) -> list[float]:
    """Synchronous encode — not exported; `encode` is the async wrapper."""
    settings = get_settings()
    model = _load_bge_m3()
    vector = model.encode(text_input, convert_to_numpy=True, normalize_embeddings=True)
    if vector.shape != (settings.embedder_dim,):
        raise ValueError(
            f"embedder shape mismatch: model emitted {vector.shape}, "
            f"expected ({settings.embedder_dim},)"
        )
    return vector.tolist()


async def encode(text_input: str) -> list[float]:
    """Embed a single text into a 1024-dim list of floats.

    Runs the blocking `SentenceTransformer.encode` via `asyncio.to_thread`
    so the event loop stays responsive to concurrent requests and to the
    `/healthz` / `/ready` probes during an embed. Per the Silero spike
    learning: `asyncio.to_thread` work is uncancellable — the retrieval
    RTT sets the barge-in floor, which is acceptable at CPU-encode
    latencies (~20–50 ms on modern x86).

    Raises `ValueError` if the emitted vector shape does not match
    `settings.embedder_dim` — catches silent config drift (wrong model
    with a different dim) before INSERT time.
    """
    return await asyncio.to_thread(_encode_sync, text_input)

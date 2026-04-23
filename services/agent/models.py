"""Shared model loading — heavy models are loaded once at startup.

GPU models (GigaAM) and CPU models (Piper, Silero VAD) are expensive to
initialize. This module loads them once and provides accessors for the
pipeline to use, so each dispatched room gets a near-instant pipeline
start instead of waiting 5-10s for model loading.

Startup fails hard if the loader raises — we do not want a degraded
agent serving traffic.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from loguru import logger

if TYPE_CHECKING:
    # Static-only Protocols that describe the parts of the heavy models
    # we actually call into. Held under TYPE_CHECKING so tests and
    # non-GPU call sites keep resolving without importing torch /
    # numpy / coqui-tts at module load time.
    import numpy as np
    import torch

    class SileroTTSModel(Protocol):
        """Runtime interface of the torch.hub-loaded Silero v5 model.

        Matches the ``apply_tts`` signature ``silero_tts.SileroTTSService``
        calls. Silero v5 exposes other methods (``save_wav``, accent
        prediction helpers) that we don't touch — keep the Protocol
        minimal so a future coqui-tts-style API churn can't accidentally
        type-check against a stale method.
        """

        def apply_tts(
            self,
            *,
            text: str,
            speaker: str,
            sample_rate: int,
            put_accent: bool = ...,
            put_yo: bool = ...,
        ) -> torch.Tensor: ...

    class CoquiXTTSModel(Protocol):
        """Runtime interface of ``TTS.api.TTS`` for XTTS v2 synthesis.

        ``TTS.api.TTS`` exposes many methods; we only call ``.tts()`` on
        the synth path. The return type is ``list[float]`` on CPU and
        ``torch.Tensor`` on CUDA — ``xtts_tts.run_tts`` duck-types on
        ``.detach()`` to handle both.
        """

        def tts(
            self,
            *,
            text: str,
            speaker_wav: str,
            language: str,
        ) -> list[float] | np.ndarray | torch.Tensor: ...


# Generic type parameter for the cached-singleton helper. Keeps the
# helper's return type tied to the engine-specific Protocol above, so
# get_silero / get_xtts retain their narrow return types downstream.
T = TypeVar("T")


_gigaam_model: object | None = None
# Keyed caches so a silent model-name swap surfaces as a clear error
# instead of returning the wrong model. ``None`` means uncached;
# ``(name, model)`` means the given model is loaded.
_silero_cached: tuple[str, SileroTTSModel] | None = None
# Same keyed-cache shape as Silero. XTTS weights are ~1.8 GB of VRAM on
# load so this singleton is genuinely heavy.
_xtts_cached: tuple[str, CoquiXTTSModel] | None = None


def _load_cached_singleton(
    cache: tuple[str, T] | None,
    *,
    model_name: str,
    label: str,
    construct: Callable[[], T],
) -> tuple[tuple[str, T], T]:
    """Load a keyed-cache singleton model with hard-swap protection.

    Shared pattern between ``load_silero`` and ``load_xtts`` (and a
    candidate for any future GPU-resident engine): keyed cache, explicit
    RuntimeError on name swap, timing log bracketing the construction.

    The caller owns the module-level cache variable — this helper takes
    the current cache tuple and returns the new one plus the model. The
    caller is responsible for assigning the returned tuple back to the
    module-level variable (simpler than passing getter/setter closures
    and keeps the global-mutation audit trail local to each load_*).

    Raises ``RuntimeError`` if ``cache`` is non-None with a different
    ``model_name`` — model-name swaps at runtime are always a config
    drift, never a normal flow.
    """
    if cache is not None:
        cached_name, cached_model = cache
        if cached_name == model_name:
            return cache, cached_model
        raise RuntimeError(
            f"load_{label.lower()} already loaded {cached_name!r}; "
            f"cannot switch to {model_name!r} without agent restart"
        )

    logger.info("Loading {label} model={model}", label=label, model=model_name)
    t0 = time.monotonic()
    model = construct()
    logger.info(
        "{label} model loaded in {elapsed:.1f}s",
        label=label,
        elapsed=time.monotonic() - t0,
    )
    return (model_name, model), model


def load_gigaam(model_name: str = "v3_e2e_rnnt") -> object:
    """Load the GigaAM model once. Subsequent calls return the cached instance.

    Weights are fetched from HuggingFace on first call into HF_HOME
    (=/home/appuser/.cache/huggingface), which is mounted as the
    `hf_cache_agent` named volume in docker-compose.yml. First-run startup
    pays a one-time download (a few hundred MB — v3_e2e_rnnt carries the
    RNNT predictor/joint network plus the punctuation + text-normalization
    head on top of the shared v3 SSL backbone); subsequent container
    restarts hit the volume cache and load from local disk.

    GigaAM does not use the shared ``_load_cached_singleton`` helper
    because its cache shape is different (single slot, not name-keyed) —
    historical carryover. Harmonizing is tracked as a cleanup but adds
    zero behavior difference today.

    Returns the gigaam.Model instance; typed as `object` because gigaam
    doesn't expose stable public type annotations.
    """
    global _gigaam_model
    if _gigaam_model is not None:
        return _gigaam_model

    # Local import so tests without gigaam installed (e.g. the agent's
    # existing test suite on a non-CUDA runner) can skip this without
    # hitting an ImportError at module scope.
    import gigaam  # noqa: PLC0415 — heavy import by design

    logger.info("Loading GigaAM model={model}", model=model_name)
    t0 = time.monotonic()
    _gigaam_model = gigaam.load_model(model_name)
    logger.info("GigaAM model loaded in {elapsed:.1f}s", elapsed=time.monotonic() - t0)
    return _gigaam_model


def get_gigaam() -> object:
    """Return the pre-loaded GigaAM model. Raises if not loaded."""
    if _gigaam_model is None:
        raise RuntimeError("GigaAM model not loaded — call load_gigaam() first")
    return _gigaam_model


def load_silero(model_name: str = "v5_cis_base") -> "SileroTTSModel":
    """Load the Silero v5 Russian TTS model once. Cached by model_name.

    Silero v5 has no PyPI package; the model is fetched via ``torch.hub``
    from ``snakers4/silero-models``. First call performs a network fetch
    into ``$TORCH_HOME/hub/`` (=/home/appuser/.cache/torch, mounted as
    the ``torch_cache_agent`` named volume in docker-compose.yml).
    Subsequent container restarts hit the volume cache and load from
    local disk without network.

    When a CUDA device is available we move the model to GPU after load
    — ``torch.hub.load`` returns a CPU-bound model by default and
    ``apply_tts`` would otherwise run on CPU even on the L4 host.

    The cache is keyed on ``model_name``: repeat calls with the same
    name return the cached singleton; calls with a different name
    raise ``RuntimeError`` rather than silently returning the wrong
    model. Switching speakers requires an agent restart. This guards
    against the scenario where a configuration edit changes the
    expected speaker but the already-loaded model keeps serving
    traffic.

    Returns the Silero model, typed as ``SileroTTSModel`` (Protocol
    over ``apply_tts``) so static type checkers can validate call sites.
    """
    global _silero_cached

    def _construct() -> "SileroTTSModel":
        # Local import — torch is a multi-hundred-MB dep and this
        # module is imported by the test suite on CPU-only CI where
        # torch is not present. Deferring the import matches gigaam.
        import torch  # noqa: PLC0415 — heavy import by design

        # torch.hub.load returns (model, example_text_map) for
        # silero_tts. The example map is unused; discard it to avoid
        # holding a reference to test-corpus strings we'll never read.
        model, _example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=model_name,
            trust_repo=True,
        )
        # Silero returns a CPU-bound model by default. Move to CUDA
        # when a device is present; fall back to CPU otherwise (CI,
        # local dev on macOS). apply_tts runs on whichever device the
        # model is bound to.
        #
        # Call .to() for its side-effects only — DO NOT reassign.
        # Silero v5 is shipped as a torch.package scripted module
        # (not a plain ``nn.Module``); its ``.to()`` implementation
        # returns ``None`` instead of ``self``, so
        # ``model = model.to(device)`` would silently nuke our handle.
        # Observed live: _silero_cached[1] ended up as None,
        # get_silero() returned None, SileroTTSService's eager bind
        # saw silero_model=None, and every synth emitted
        # ErrorFrame("TTS unavailable").
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
            logger.info("Silero model moved to CUDA")
        else:
            logger.info("Silero running on CPU (no CUDA device available)")
        return model

    _silero_cached, model = _load_cached_singleton(
        _silero_cached,
        model_name=model_name,
        label="Silero",
        construct=_construct,
    )
    return model


def get_silero() -> "SileroTTSModel":
    """Return the pre-loaded Silero TTS model. Raises if not loaded."""
    if _silero_cached is None:
        raise RuntimeError("Silero model not loaded — call load_silero() first")
    return _silero_cached[1]


# Partial-download self-heal state for XTTS (see _check_and_clean_partial_xtts_cache).
# Coqui-tts stores each model under TTS_HOME with '/' replaced by '--'.
_XTTS_SENTINEL_FILENAME = ".xtts-download-complete"


def _xtts_cache_dir(model_name: str) -> Path:
    """Return the directory coqui-tts writes XTTS weights into.

    Matches ``coqui-tts``'s internal layout: the model name has
    ``/`` replaced by ``--`` and is placed under ``TTS_HOME`` (falling
    back to ``~/.local/share/tts`` when unset). We read the env here so
    tests can point it at a tmp_path without modifying the real
    hf_cache_agent volume.
    """
    tts_home_env = os.environ.get("TTS_HOME")
    if tts_home_env:
        base = Path(tts_home_env)
    else:
        base = Path.home() / ".local" / "share" / "tts"
    return base / model_name.replace("/", "--")


def _check_and_clean_partial_xtts_cache(model_name: str) -> None:
    """Self-heal the hf_cache_agent volume when XTTS weights are partial.

    First-boot XTTS downloads ~1.8 GB into the cache dir. If the
    container is killed mid-download (spot preemption, OOM kill, compose
    restart), huggingface_hub leaves the cache directory PRESENT but the
    weight file absent. On the next boot, ``TTS(model_name)`` sees the
    directory, skips the download, and ``torch.load()`` raises
    ``FileNotFoundError`` on the missing weight — which ``main.main()``
    surfaces as ``sys.exit(3)`` and the container crashloops.

    We write a sentinel file (``.xtts-download-complete``) inside the
    cache dir AFTER ``TTS(model_name)`` returns cleanly. If a later boot
    finds the cache dir present but the sentinel missing, treat the
    cache as corrupt and nuke it so the next ``TTS()`` call re-downloads
    from scratch.

    Safe to call when the cache dir doesn't exist yet — no-op.
    """
    cache_dir = _xtts_cache_dir(model_name)
    if not cache_dir.exists():
        return
    sentinel = cache_dir / _XTTS_SENTINEL_FILENAME
    if sentinel.exists():
        # Previous download completed cleanly — leave it alone.
        return
    logger.warning(
        "XTTS cache at {path} is present but sentinel missing — treating as "
        "partial download and clearing; TTS() will re-download.",
        path=cache_dir,
    )
    shutil.rmtree(cache_dir, ignore_errors=False)


def _mark_xtts_download_complete(model_name: str) -> None:
    """Write the completion sentinel inside the XTTS cache directory.

    Called after ``TTS(model_name)`` returns cleanly so the next boot
    can recognize this as a complete download and skip the re-download
    path (see ``_check_and_clean_partial_xtts_cache``). If the cache
    directory doesn't exist (e.g. TTS() was mocked in tests, or the
    model is loaded from a hand-populated path), just skip — no harm.
    """
    cache_dir = _xtts_cache_dir(model_name)
    if not cache_dir.exists():
        return
    try:
        (cache_dir / _XTTS_SENTINEL_FILENAME).touch()
    except OSError:
        # Best-effort — a read-only mount or a permission issue
        # shouldn't bring the agent down at this point; we'd rather
        # re-download on the next boot than hard-fail the current one.
        logger.exception(
            "Failed to write XTTS download-complete sentinel under {path}",
            path=cache_dir,
        )


def load_xtts(
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> "CoquiXTTSModel":
    """Load the Coqui XTTS v2 model once. Cached by model_name.

    XTTS v2 is distributed via the maintained ``coqui-tts`` PyPI package
    (a fork of the deprecated Coqui TTS). The weights download on first
    use into ``~/.local/share/tts`` by default; in the agent container
    this maps through the ``hf_cache_agent`` volume via the ``TTS_HOME``
    env. ~1.8 GB of VRAM on the L4 GPU.

    Self-heals partial downloads: if a previous container was killed
    mid-download, the cache directory exists without the completion
    sentinel and would otherwise crashloop with FileNotFoundError. See
    ``_check_and_clean_partial_xtts_cache``.

    When CUDA is available the model is moved to GPU after load — the
    coqui ``TTS`` wrapper exposes a standard ``.to(device)`` method.
    Unlike Silero's torch.package scripted module, ``.to()`` here
    returns ``self`` so reassignment would be safe — but for parity we
    call it for side-effect only.

    The cache is keyed on ``model_name``: repeat calls with the same
    name return the cached singleton; calls with a different name raise
    ``RuntimeError`` rather than silently returning the wrong model.

    Returns the TTS instance, typed as ``CoquiXTTSModel`` (Protocol
    over ``tts``).
    """
    global _xtts_cached

    def _construct() -> "CoquiXTTSModel":
        # Local imports — torch + TTS are heavy deps; this module is
        # imported by the test suite on CPU-only CI where coqui-tts
        # is not present. Deferring matches gigaam / silero patterns.
        import torch  # noqa: PLC0415 — heavy import by design
        from TTS.api import TTS  # noqa: PLC0415 — heavy import by design

        # Self-heal BEFORE TTS() constructs. If the previous boot left
        # a partial download behind, TTS() would trust the dir and
        # torch.load() would raise FileNotFoundError under our except
        # handler in main.py, producing a permanent crashloop. Clear
        # the corrupt cache first so the constructor re-downloads.
        _check_and_clean_partial_xtts_cache(model_name)

        tts = TTS(model_name)

        # Mark the download complete AFTER TTS() returns cleanly.
        _mark_xtts_download_complete(model_name)

        # Call .to() for its side-effects only — match the Silero
        # defensive pattern even though coqui-tts's .to() does return
        # self (see TTS/api.py: ``def to(self, device):
        # self.synthesizer.tts_model.to(device); return self``). If a
        # future coqui-tts refactor changes that return contract, the
        # live cache won't silently become None.
        if torch.cuda.is_available():
            tts.to(torch.device("cuda"))
            logger.info("XTTS model moved to CUDA")
        else:
            logger.info("XTTS running on CPU (no CUDA device available)")
        return tts

    _xtts_cached, tts = _load_cached_singleton(
        _xtts_cached,
        model_name=model_name,
        label="XTTS",
        construct=_construct,
    )
    return tts


def get_xtts() -> "CoquiXTTSModel":
    """Return the pre-loaded XTTS model. Raises if not loaded."""
    if _xtts_cached is None:
        raise RuntimeError("XTTS model not loaded — call load_xtts() first")
    return _xtts_cached[1]


def _reset_models_for_tests() -> None:
    """Reset cached singletons. Tests only — do not call from runtime."""
    global _gigaam_model, _silero_cached, _xtts_cached
    _gigaam_model = None
    _silero_cached = None
    _xtts_cached = None

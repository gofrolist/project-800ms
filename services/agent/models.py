"""Shared model loading — heavy models are loaded once at startup.

GPU models (GigaAM) and CPU models (Piper, Silero VAD) are expensive to
initialize. This module loads them once and provides accessors for the
pipeline to use, so each dispatched room gets a near-instant pipeline
start instead of waiting 5-10s for model loading.

Startup fails hard if the loader raises — we do not want a degraded
agent serving traffic.
"""

from __future__ import annotations

import time

from loguru import logger


_gigaam_model: object | None = None
# Keyed cache so a silent model-name swap surfaces as a clear error
# instead of returning the wrong model. ``None`` means uncached;
# ``(name, model)`` means the given model is loaded.
_silero_cached: tuple[str, object] | None = None
# Same keyed-cache shape as Silero for consistency. ``None`` means
# uncached. XTTS weights are ~1.8 GB of VRAM on load so this singleton
# is genuinely heavy.
_xtts_cached: tuple[str, object] | None = None


def load_gigaam(model_name: str = "v3_e2e_rnnt") -> object:
    """Load the GigaAM model once. Subsequent calls return the cached instance.

    Weights are fetched from HuggingFace on first call into HF_HOME
    (=/home/appuser/.cache/huggingface), which is mounted as the
    `hf_cache_agent` named volume in docker-compose.yml. First-run startup
    pays a one-time download (a few hundred MB — v3_e2e_rnnt carries the
    RNNT predictor/joint network plus the punctuation + text-normalization
    head on top of the shared v3 SSL backbone); subsequent container
    restarts hit the volume cache and load from local disk.

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


def load_silero(model_name: str = "v5_cis_base") -> object:
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

    Returns the Silero model instance; typed as ``object`` because
    ``torch.hub.load`` returns a ``torch.nn.Module`` whose exact type
    isn't re-exported from a stable public path.
    """
    global _silero_cached
    if _silero_cached is not None:
        cached_name, cached_model = _silero_cached
        if cached_name == model_name:
            return cached_model
        raise RuntimeError(
            f"load_silero already loaded {cached_name!r}; "
            f"cannot switch to {model_name!r} without agent restart"
        )

    # Local import — torch is a multi-hundred-MB dep and this module is
    # imported by the test suite on CPU-only CI where torch is not
    # present. Deferring the import matches the gigaam pattern above.
    import torch  # noqa: PLC0415 — heavy import by design

    logger.info("Loading Silero TTS model={model}", model=model_name)
    t0 = time.monotonic()
    # torch.hub.load returns (model, example_text_map) for silero_tts.
    # The example map is unused; discard it to avoid holding a reference
    # to test-corpus strings we'll never read.
    model, _example_text = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker=model_name,
        trust_repo=True,
    )
    # Silero returns a CPU-bound model by default. Move to CUDA when a
    # device is present; fall back to CPU otherwise (CI, local dev on
    # macOS). apply_tts runs on whichever device the model is bound to.
    #
    # Call .to() for its side-effects only — DO NOT reassign. Silero v5
    # is shipped as a torch.package scripted module (not a plain
    # ``nn.Module``); its ``.to()`` implementation returns ``None``
    # instead of ``self``, so ``model = model.to(device)`` would
    # silently nuke our handle. Observed live: _silero_cached[1] ended
    # up as None, get_silero() returned None, SileroTTSService's
    # eager bind saw silero_model=None, and every synth emitted
    # ErrorFrame("TTS unavailable").
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        logger.info("Silero model moved to CUDA")
    else:
        logger.info("Silero running on CPU (no CUDA device available)")
    _silero_cached = (model_name, model)
    logger.info("Silero model loaded in {elapsed:.1f}s", elapsed=time.monotonic() - t0)
    return model


def get_silero() -> object:
    """Return the pre-loaded Silero TTS model. Raises if not loaded."""
    if _silero_cached is None:
        raise RuntimeError("Silero model not loaded — call load_silero() first")
    return _silero_cached[1]


def load_xtts(
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> object:
    """Load the Coqui XTTS v2 model once. Cached by model_name.

    XTTS v2 is distributed via the maintained ``coqui-tts`` PyPI package
    (a fork of the deprecated Coqui TTS). The weights download on first
    use into ``~/.local/share/tts`` by default; in the agent container
    this maps through the same ``hf_cache_agent`` volume as GigaAM when
    ``TTS_HOME`` is pointed at it. ~1.8 GB of VRAM on the L4 GPU.

    When CUDA is available the model is moved to GPU after load — the
    coqui ``TTS`` wrapper exposes a standard ``.to(device)`` method.
    Unlike Silero's torch.package scripted module, ``.to()`` here
    returns ``self`` so reassignment would be safe — but for parity we
    call it for side-effect only (same defensive pattern in case a
    future coqui-tts refactor changes this).

    The cache is keyed on ``model_name``: repeat calls with the same
    name return the cached singleton; calls with a different name raise
    ``RuntimeError`` rather than silently returning the wrong model.

    Returns the ``TTS.api.TTS`` instance; typed as ``object`` because
    ``coqui-tts`` does not publish stable types at a stable import path.
    """
    global _xtts_cached
    if _xtts_cached is not None:
        cached_name, cached_model = _xtts_cached
        if cached_name == model_name:
            return cached_model
        raise RuntimeError(
            f"load_xtts already loaded {cached_name!r}; "
            f"cannot switch to {model_name!r} without agent restart"
        )

    # Local imports — torch + TTS are heavy deps; this module is imported
    # by the test suite on CPU-only CI where coqui-tts is not present.
    # Deferring the import matches gigaam/silero patterns above.
    import torch  # noqa: PLC0415 — heavy import by design
    from TTS.api import TTS  # noqa: PLC0415 — heavy import by design

    logger.info("Loading XTTS model={model}", model=model_name)
    t0 = time.monotonic()
    tts = TTS(model_name)
    # Call .to() for its side-effects only — match the Silero defensive
    # pattern even though coqui-tts's .to() does return self (see
    # TTS/api.py: ``def to(self, device): self.synthesizer.tts_model.to(
    # device); return self``). If a future coqui-tts refactor changes
    # that return contract, the live cache won't silently become None.
    if torch.cuda.is_available():
        tts.to(torch.device("cuda"))
        logger.info("XTTS model moved to CUDA")
    else:
        logger.info("XTTS running on CPU (no CUDA device available)")
    _xtts_cached = (model_name, tts)
    logger.info("XTTS model loaded in {elapsed:.1f}s", elapsed=time.monotonic() - t0)
    return tts


def get_xtts() -> object:
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

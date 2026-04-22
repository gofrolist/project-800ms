"""TTS provider factory.

Central dispatch point for selecting a Pipecat TTSService implementation
based on the configured engine name. The factory is the only seam the
pipeline uses to obtain a TTS service; swapping engines is a config
change (``TTS_ENGINE`` env) rather than a code change.

Supported engines:

* ``piper`` — local CPU-bound Piper synthesis. Default.
* ``silero`` — GPU-accelerated Silero v5 Russian synthesis (Unit 3).
  Requires the agent to have preloaded the model via ``load_silero()``
  at startup; the factory resolves the shared singleton lazily.

``qwen3`` branch is added in Unit 4. Unknown engine names raise
``ValueError`` at pipeline build time — the failure surfaces on the
first dispatch, not at import time, which matches the "config flip,
redeploy, observe" operator workflow.

Voice string semantics
----------------------
``voice`` is a provider-specific identifier. Piper expects names like
``ru_RU-denis-medium``; Silero expects speaker ids like ``v5_cis_base``;
OpenAI-compatible sidecars (Qwen3) expect voice names like ``alloy``.
The factory does NOT translate or validate voice strings — each adapter
interprets the value in its own namespace. Callers (pipeline.py) pass
``overrides.voice or cfg.tts_voice`` as-is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.services.tts_service import TTSService

    from pipeline import AgentConfig


def build_tts_service(
    engine: str,
    *,
    cfg: AgentConfig,
    voice: str,
) -> TTSService:
    """Return a TTSService implementation for the requested engine.

    Args:
        engine: Engine identifier. One of ``piper`` (Unit 2), ``silero``
            (Unit 3), ``qwen3`` (Unit 4, not yet wired).
        cfg: Frozen agent config. Only the fields relevant to the chosen
            engine are read (e.g. ``piper_voices_dir`` for the piper
            branch).
        voice: Provider-specific voice identifier. Not validated by the
            factory; the adapter interprets it. For ``silero`` this is
            the speaker id (e.g. ``v5_cis_base``).

    Returns:
        An instantiated Pipecat TTSService ready to be wired into a
        pipeline.

    Raises:
        ValueError: If ``engine`` is not a known engine name. The error
            message includes the offending value.
    """
    if engine == "piper":
        # Lazy import to keep TTS-engine heavy deps out of the import graph
        # when a different engine is selected. For piper specifically this
        # is mostly future-proofing — piper is already pulled via
        # pipecat-ai[piper]. Matches the noqa pattern used in models.py.
        from pipecat.services.piper.tts import PiperTTSService  # noqa: PLC0415

        return PiperTTSService(
            settings=PiperTTSService.Settings(voice=voice),
            download_dir=cfg.piper_voices_dir,
            use_cuda=False,
        )

    if engine == "silero":
        # Lazy imports — torch + the Silero adapter are only exercised
        # when Silero is the active engine. Keeps CPU-only CI importing
        # this module cheaply and ensures non-silero deployments never
        # pay the torch.hub resolution cost at factory-dispatch time.
        from models import get_silero  # noqa: PLC0415
        from silero_tts import SileroSettings, SileroTTSService  # noqa: PLC0415

        return SileroTTSService(
            silero_model=get_silero(),
            settings=SileroSettings(speaker=voice),
        )

    raise ValueError(f"unknown TTS_ENGINE: {engine!r}")

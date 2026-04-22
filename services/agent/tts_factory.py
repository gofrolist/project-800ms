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
* ``qwen3`` — OpenAI-compatible Qwen3-TTS sidecar (Unit 4). The factory
  returns Pipecat's upstream ``OpenAITTSService`` pointed at the sidecar
  URL; the wrapper translates OpenAI-compat requests into Qwen3-TTS
  calls. No agent-side custom adapter.

Unknown engine names raise ``ValueError`` at pipeline build time — the
failure surfaces on the first dispatch, not at import time, which
matches the "config flip, redeploy, observe" operator workflow.

Voice string semantics
----------------------
``voice`` is a provider-specific identifier. Piper expects names like
``ru_RU-denis-medium``; Silero expects speaker ids like ``v5_cis_base``;
OpenAI-compatible sidecars (Qwen3) expect voice names like ``alloy``.
The factory does NOT translate voice strings across engines — each
adapter interprets the value in its own namespace. Callers (pipeline.py)
pass ``overrides.voice or cfg.tts_voice`` as-is.

The one exception is the ``qwen3`` branch: Pipecat's ``OpenAITTSService``
enforces a closed whitelist of 13 OpenAI voice names at request time
via a raw dict lookup that raises ``KeyError`` on unknown values. If an
operator flips ``TTS_ENGINE=qwen3`` while ``TTS_VOICE`` is still set to
a Piper/Silero-specific name (e.g. ``ru_RU-denis-medium``), dispatch
would crash. The factory substitutes ``QWEN3_DEFAULT_VOICE`` when the
supplied ``voice`` is outside the whitelist and emits a loguru warning;
the Qwen3 wrapper maps that name onto its internal catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pipecat.services.tts_service import TTSService

    from pipeline import AgentConfig


# Pipecat's OpenAITTSService VALID_VOICES keys as of pipecat-ai==0.0.108
# (pipecat/services/openai/tts.py:46-60). The Qwen3 wrapper's
# LANGUAGE_CODE_MAPPING accepts the same OpenAI voice names and re-maps
# them onto Qwen3's internal catalog; see infra/qwen3-tts-wrapper/README.md.
_QWEN3_VALID_VOICES: frozenset[str] = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "cedar",
        "coral",
        "echo",
        "fable",
        "marin",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
    }
)

# Default substitution when an incompatible voice flows into the Qwen3
# branch. "echo" maps to the wrapper's "Ryan" voice — the only
# English-trained voice in the Qwen3 catalog, which the wrapper's README
# calls out as the closest starting default for Russian (English phonology
# is nearer to Russian than the Chinese/Japanese-trained voices).
_QWEN3_DEFAULT_VOICE = "echo"


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

    if engine == "qwen3":
        # Lazy import — only pull Qwen3TTSService (which extends
        # pipecat.services.openai.tts.OpenAITTSService with network-error
        # redaction; see ``services/agent/qwen3_tts.py``) when the Qwen3
        # branch actually dispatches. Matches the lazy-import pattern
        # above and keeps openai-sdk + httpx out of the Piper/Silero
        # startup import graph.
        from qwen3_tts import Qwen3TTSService  # noqa: PLC0415

        if not cfg.qwen3_base_url:
            raise ValueError("qwen3 TTS engine requires QWEN3_TTS_BASE_URL to be set")
        if not cfg.qwen3_api_key:
            raise ValueError("qwen3 TTS engine requires QWEN3_TTS_API_KEY to be set")

        # Substitute out-of-whitelist voices to avoid KeyError at dispatch
        # time. The wrapper maps OpenAI voice names onto Qwen3's internal
        # catalog (see infra/qwen3-tts-wrapper/README.md); the default
        # "echo" corresponds to Qwen3's "Ryan" voice which the README
        # recommends as the Russian-synthesis starting default.
        effective_voice = voice
        if voice not in _QWEN3_VALID_VOICES:
            # Loguru does not honor the ``!r`` conversion flag in its
            # format-spec parser the way stdlib str.format does, so
            # precompute reprs and pass them as plain string kwargs.
            # Without this the literal ``!r`` would reach the log output
            # as ``{voice!r}`` rather than a quoted repr.
            logger.warning(
                "TTS_VOICE={voice_repr} is not a Pipecat OpenAITTSService "
                "whitelist voice; substituting {default_repr} for the Qwen3 "
                "sidecar dispatch. Set TTS_VOICE to one of {valid} to "
                "silence this warning.",
                voice_repr=repr(voice),
                default_repr=repr(_QWEN3_DEFAULT_VOICE),
                valid=sorted(_QWEN3_VALID_VOICES),
            )
            effective_voice = _QWEN3_DEFAULT_VOICE

        # model="tts-1-ru" routes through the wrapper's LANGUAGE_CODE_MAPPING
        # to Russian synthesis. Pipecat's OpenAITTSService passes `model` as
        # a plain string in the request body, matching OpenAI's contract;
        # the wrapper interprets the "-ru" suffix server-side. See the
        # Request/response contract table in the wrapper README for the full
        # alias list. `language=` is intentionally not a kwarg here — the
        # OpenAI-compat schema doesn't expose it; the model alias is the
        # only clean way to route language without patching Pipecat.
        return Qwen3TTSService(
            base_url=cfg.qwen3_base_url,
            api_key=cfg.qwen3_api_key,
            model="tts-1-ru",
            voice=effective_voice,
        )

    raise ValueError(f"unknown TTS_ENGINE: {engine!r}")

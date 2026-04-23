"""Pipecat voice pipeline for project-800ms.

Graph:
    LiveKit mic audio  →  Silero VAD  →  GigaAM-v3 (GPU)  →
    LLM (vLLM or external)  →  Piper TTS (CPU)  →  LiveKit speaker

Everything streams. First audio out should land ~700ms after end-of-speech
on the RTX 5080 / L4 target hardware.

The pipeline is parameterized per-session via ``PerSessionOverrides``
(persona, voice, language, llm_model). Defaults come from ``AgentConfig``;
overrides are applied at ``build_task`` time so each room can be a
different NPC speaking a different language.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from gigaam_stt import GigaAMSettings, GigaAMSTTService
from overrides import PerSessionOverrides, build_system_prompt
from transcript import (
    AssistantTranscriptForwarder,
    ErrorFrameForwarder,
    UserTranscriptForwarder,
)
from transcript_sink import TranscriptSink
from tts_factory import build_tts_service


_VALID_TTS_ENGINES: frozenset[str] = frozenset({"piper", "silero", "qwen3", "xtts"})


@dataclass(frozen=True)
class AgentConfig:
    """Runtime config for one agent instance."""

    livekit_url: str
    livekit_token: str
    room_name: str
    vllm_base_url: str
    vllm_model: str
    tts_voice: str  # e.g. "ru_RU-denis-medium"
    vllm_api_key: str
    piper_voices_dir: Path = Path("/home/appuser/.cache/piper")
    # TTS engine selector. "piper" is the day-one default; "silero" (Unit 3)
    # and "qwen3" (Unit 4) are added by the tts_factory in later units.
    # Validated at construction time (``__post_init__``); the factory still
    # raises on unknown values when build_task() calls it, but failing at
    # AgentConfig instantiation surfaces the config error on agent boot
    # instead of on the first dispatched room.
    tts_engine: str = "piper"
    # Qwen3-TTS sidecar endpoint + API key. Empty on Piper/Silero deploys —
    # the factory's qwen3 branch validates non-empty at dispatch time, so
    # non-qwen3 deploys don't need to set these. See infra/.env.example.
    qwen3_base_url: str = ""
    qwen3_api_key: str = ""
    # Qwen3-specific voice name. Takes precedence over ``tts_voice`` /
    # per-session ``overrides.voice`` when the Qwen3 engine dispatches.
    # Empty falls back to those generic values. The typical usage is a
    # ``clone:<profile>`` identifier pointing at a voice-library profile
    # baked into the wrapper image (see ``QWEN3_VARIANT=0.6B-Base`` or
    # ``1.7B-Base`` deploys). Piper/Silero sessions ignore this field.
    qwen3_tts_voice: str = ""
    # XTTS-specific voice name. Same precedence rationale as
    # ``qwen3_tts_voice``: lets ops ship a mixed deploy where Piper/Silero
    # continue to use ``TTS_VOICE=ru_RU-denis-medium`` while XTTS reads
    # ``XTTS_TTS_VOICE=clone:<profile>``. Empty falls back to the generic
    # ``tts_voice`` / per-session ``overrides.voice``. XTTS is always a
    # voice-cloning engine — non-``clone:`` values will be rejected at
    # service construction with a clear ValueError.
    xtts_tts_voice: str = ""
    # Root of the shared voice-library mount. Both the XTTS adapter and
    # the Qwen3 sidecar (via its own bind mount) look under
    # ``<dir>/profiles/<profile>/`` for ``meta.json`` + ``ref.wav``.
    # Default matches the agent container's bind-mount target in
    # infra/docker-compose.yml; operators with a different layout should
    # override via ``XTTS_VOICE_LIBRARY_DIR``. Piper/Silero sessions
    # ignore this field.
    xtts_voice_library_dir: Path = Path("/opt/voice_library")
    # When both values are set, final STT and LLM utterances are also
    # POSTed to the API's /internal/transcripts endpoint. Empty = in-UI
    # transcripts only (no DB persistence).
    api_base_url: str = ""
    agent_internal_token: str = ""

    def __post_init__(self) -> None:
        if self.tts_engine not in _VALID_TTS_ENGINES:
            raise ValueError(
                f"AgentConfig.tts_engine must be one of "
                f"{sorted(_VALID_TTS_ENGINES)!r}; got {self.tts_engine!r}"
            )


def build_task(
    cfg: AgentConfig,
    *,
    gigaam_model: object | None = None,
    overrides: PerSessionOverrides | None = None,
) -> tuple[PipelineTask, LiveKitTransport]:
    """Build the Pipecat pipeline + task for one call.

    `overrides` carries per-session knobs from the API's /dispatch payload
    (persona, voice, language, llm_model). When any field is set, it takes
    precedence over the corresponding value in `cfg`.
    """
    overrides = overrides or PerSessionOverrides()
    language = overrides.effective_language
    voice = overrides.voice or cfg.tts_voice
    llm_model = overrides.llm_model or cfg.vllm_model
    # Per-session engine override (from the demo site's three-button
    # selector). Falls back to cfg.tts_engine (from TTS_ENGINE env) when
    # unset. overrides.tts_engine is already whitelist-validated in
    # PerSessionOverrides.from_dispatch, so the factory branch below never
    # sees a bogus engine value from a client.
    tts_engine = overrides.tts_engine or cfg.tts_engine
    system_instruction = build_system_prompt(overrides.persona, language)

    transport = LiveKitTransport(
        url=cfg.livekit_url,
        token=cfg.livekit_token,
        room_name=cfg.room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
        ),
    )

    # start_secs=0.2 makes interruption snap in quickly: user has to speak
    # continuously for 200ms before the barge-in fires. Raising above 0.3
    # means the bot keeps talking past short interjections ("стой", "wait")
    # because VAD hasn't confirmed a start yet. stop_secs=0.3 keeps a bit of
    # tail buffer so we don't cut the user's last syllable.
    vad_params = VADParams(
        confidence=0.5,
        start_secs=0.2,
        stop_secs=0.3,
        min_volume=0.1,
    )
    vad_analyzer = SileroVADAnalyzer(params=vad_params)

    stt = GigaAMSTTService(
        gigaam_model=gigaam_model,
        settings=GigaAMSettings(language=language),
    )

    llm = OpenAILLMService(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        settings=OpenAILLMService.Settings(
            model=llm_model,
            system_instruction=system_instruction,
            max_tokens=128,
        ),
    )

    tts = build_tts_service(tts_engine, cfg=cfg, voice=voice)

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
        ),
    )

    sink: TranscriptSink | None = None
    if cfg.api_base_url and cfg.agent_internal_token:
        sink = TranscriptSink(
            api_base_url=cfg.api_base_url,
            internal_token=cfg.agent_internal_token,
            room=cfg.room_name,
        )

    user_transcript = UserTranscriptForwarder(transport, sink=sink)
    assistant_transcript = AssistantTranscriptForwarder(transport, sink=sink)
    # ErrorFrameForwarder sits AFTER the TTS service so that TTS synth
    # errors (missing voice profile, CUDA OOM, model-not-loaded) are
    # surfaced to the UI via the LiveKit data channel instead of the
    # user hearing silence. Upstream STT / LLM errors also flow through,
    # so any backend ErrorFrame gets shown to the caller.
    error_forwarder = ErrorFrameForwarder(transport)

    # The aggregator's LLMUserAggregatorParams(vad_analyzer=...) runs VAD
    # internally via its own VADController and broadcasts the resulting
    # VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame both
    # upstream (reaching STT for segment gating) and downstream (reaching
    # LLM/TTS for barge-in). Adding a separate VADProcessor here was the
    # pre-0.0.99 pattern and is double-work on top of the aggregator.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_transcript,
            user_agg,
            llm,
            assistant_transcript,
            tts,
            error_forwarder,
            transport.output(),
            assistant_agg,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
    )

    return task, transport

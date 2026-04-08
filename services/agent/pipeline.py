"""Pipecat voice pipeline for project-800ms.

Graph:
    LiveKit mic audio  →  Silero VAD  →  Faster-Whisper (GPU)  →
    Qwen2.5-7B (vLLM)  →  Piper TTS (CPU, in-process)  →  LiveKit speaker

Everything streams. First audio out should land ~700ms after end-of-speech on
the RTX 5080 / L4 target hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import Model as WhisperModel
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are spoken aloud, so: "
    "keep them short (1-3 sentences), conversational, and never use markdown, "
    "bullet points, or emojis. If the user speaks Russian, reply in Russian. "
    "If the user speaks English, reply in English."
)


@dataclass(frozen=True)
class AgentConfig:
    """Runtime config for one agent instance. Immutable — build a new one to change."""

    livekit_url: str
    livekit_token: str
    room_name: str
    vllm_base_url: str
    vllm_model: str
    tts_voice: str  # e.g. "ru_RU-ruslan-medium"
    whisper_model: WhisperModel = WhisperModel.LARGE_V3_TURBO
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"


def build_task(cfg: AgentConfig) -> tuple[PipelineTask, LiveKitTransport]:
    """Build the Pipecat pipeline + task for one call.

    Returns (task, transport) so the caller can attach event handlers
    (greeting on participant-join, teardown on disconnect) before running.
    """
    transport = LiveKitTransport(
        url=cfg.livekit_url,
        token=cfg.livekit_token,
        room_name=cfg.room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = WhisperSTTService(
        model=cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        # Don't force language — let Whisper auto-detect per utterance
        # so EN and RU in the same conversation both work.
        no_speech_prob=0.4,
    )

    # vLLM speaks the OpenAI API — point the OpenAI client at it.
    llm = OpenAILLMService(
        base_url=cfg.vllm_base_url,
        api_key="not-used",  # vLLM doesn't check, but the client requires a non-empty string
        model=cfg.vllm_model,
        settings=OpenAILLMService.Settings(
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(
            voice=cfg.tts_voice,
        ),
        use_cuda=False,  # CPU is plenty for Piper; keep GPU for LLM + Whisper
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,
            transport.output(),
            assistant_agg,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,  # barge-in
        ),
    )

    return task, transport

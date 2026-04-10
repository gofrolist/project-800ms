"""Pipecat voice pipeline for project-800ms.

Graph:
    LiveKit mic audio  →  Silero VAD  →  Faster-Whisper (GPU)  →
    LLM (vLLM)  →  Piper TTS (CPU)  →  LiveKit speaker

Everything streams. First audio out should land ~700ms after end-of-speech on
the RTX 5080 / L4 target hardware.

Whisper auto-detects language per utterance. The LLM always replies in
Russian. A single Russian Piper voice handles output.
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
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import Model as WhisperModelSize
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from faster_whisper import WhisperModel

from stt_filter import FilteredWhisperSTTService
from transcript import AssistantTranscriptForwarder, UserTranscriptForwarder

SYSTEM_PROMPT = (
    "Ты голосовой помощник. Твои ответы озвучиваются через синтез речи, "
    "поэтому строго соблюдай правила: "
    "Отвечай коротко, 1-3 предложения, разговорный тон. "
    "Никогда не используй маркдаун, звёздочки, списки, "
    "тире, скобки, кавычки, эмодзи и специальные символы. "
    "Используй только обычные слова и базовую пунктуацию (точка, запятая, "
    "вопросительный знак, восклицательный знак). "
    "Всегда отвечай только на русском языке. "
    "Никогда не используй слова или символы из других языков, "
    "включая английский, китайский и любые другие. "
    "Все имена и названия транслитерируй на русский."
)


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
    whisper_model: WhisperModelSize = WhisperModelSize.LARGE
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"


def build_task(
    cfg: AgentConfig, *, whisper_model: WhisperModel | None = None
) -> tuple[PipelineTask, LiveKitTransport]:
    """Build the Pipecat pipeline + task for one call."""
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

    vad_params = VADParams(
        confidence=0.5,
        start_secs=0.3,
        stop_secs=0.3,
        min_volume=0.1,
    )
    vad_analyzer = SileroVADAnalyzer(params=vad_params)
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

    stt = FilteredWhisperSTTService(
        model=whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        settings=FilteredWhisperSTTService.Settings(
            model=cfg.whisper_model.value,
            language="ru",
            no_speech_prob=0.4,
            min_avg_logprob=-0.7,
            max_compression_ratio=2.4,
        ),
    )

    llm = OpenAILLMService(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        settings=OpenAILLMService.Settings(
            model=cfg.vllm_model,
            system_instruction=SYSTEM_PROMPT,
            max_tokens=128,
        ),
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=cfg.tts_voice),
        download_dir=cfg.piper_voices_dir,
        use_cuda=False,
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
        ),
    )

    user_transcript = UserTranscriptForwarder(transport)
    assistant_transcript = AssistantTranscriptForwarder(transport)

    pipeline = Pipeline(
        [
            transport.input(),
            vad_processor,
            stt,
            user_transcript,
            user_agg,
            llm,
            assistant_transcript,
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
            allow_interruptions=True,
        ),
    )

    return task, transport

"""Pipecat voice pipeline for project-800ms.

Graph:
    LiveKit mic audio  →  Silero VAD  →  Faster-Whisper (GPU)  →
    Qwen2.5-7B (vLLM)  →  Bilingual TTS router  →  LiveKit speaker
                                       │
                                       ├──→ Piper EN voice (CPU)
                                       └──→ Piper RU voice (CPU)

Everything streams. First audio out should land ~700ms after end-of-speech on
the RTX 5080 / L4 target hardware.

Whisper auto-detects language per utterance, the LLM is told to reply in the
same language, and the TTS router below picks the matching Piper voice based
on a cheap cyrillic-vs-latin classifier (see `lang.py`). Two voice models are
loaded simultaneously (~50 MB each, CPU-resident).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TextFrame, TTSSpeakFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import Model as WhisperModel
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from lang import Language, LanguageRouter

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
    # Two Piper voices, one per supported output language. Both are loaded
    # at startup; the bilingual router below picks per utterance.
    tts_voice_en: str  # e.g. "en_US-amy-medium"
    tts_voice_ru: str  # e.g. "ru_RU-ruslan-medium"
    # vLLM doesn't authenticate by default, but the OpenAI client requires
    # a non-empty string. Override via VLLM_API_KEY when vLLM is fronted by
    # an auth proxy (e.g. an api gateway in prod).
    vllm_api_key: str = "not-used"
    whisper_model: WhisperModel = WhisperModel.LARGE_V3_TURBO
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"


class LanguageFilter(FrameProcessor):
    """Drops TextFrames / TTSSpeakFrames whose language doesn't match.

    Sits at the head of one branch of a `ParallelPipeline`. Both branches
    receive every frame; this filter drops anything destined for the
    other branch's voice. Non-text frames (lifecycle, audio, control)
    pass through unchanged so the downstream TTS can keep its state in
    sync with the rest of the pipeline.

    Stateful: short LLM-streamed chunks like "." or " " carry no language
    signal of their own, so we remember the last detected language and
    use it as the fallback. Both branches see the same frames in the
    same order, so their states stay in lock-step.
    """

    def __init__(self, target: Language) -> None:
        super().__init__()
        self._target = target
        self._router = LanguageRouter()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (TextFrame, TTSSpeakFrame)) and frame.text:
            if self._router.route(frame.text) != self._target:
                # Drop: this frame belongs to the other language's branch.
                return
        await self.push_frame(frame, direction)


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
            # Force the resampler + downstream VAD to 16 kHz mono. Silero VAD
            # only supports 8 kHz / 16 kHz; if we leave this None, the StartFrame
            # propagates LiveKit's native 48 kHz to the VAD analyzer, while the
            # transport resamples actual audio to 16 kHz — mismatch → no detection.
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            # NOTE: don't pass vad_analyzer here — it's deprecated since 0.0.101
            # and the modern transport input does NOT emit VADUserStarted/Stopped
            # frames even if it's set. We use a dedicated VADProcessor below.
        ),
    )

    # Single VAD instance, shared between the upstream VADProcessor (which
    # gates Whisper) and the LLM user aggregator (which uses VAD frames for
    # turn-taking). Two analyzers would load the Silero ONNX model twice and
    # could drift in their speech/silence decisions. Tuned for low-gain mics
    # (browser AGC off / quiet input): defaults of confidence=0.7,
    # min_volume=0.6 are both too aggressive.
    vad_params = VADParams(
        confidence=0.3,
        start_secs=0.15,
        stop_secs=0.6,
        min_volume=0.05,
    )
    vad_analyzer = SileroVADAnalyzer(params=vad_params)
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

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
        api_key=cfg.vllm_api_key,
        model=cfg.vllm_model,
        settings=OpenAILLMService.Settings(
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    # Two Piper voices, one per language. CPU-only — Piper is fast enough
    # on CPU and we want to keep the GPU free for vLLM + Whisper.
    tts_en = PiperTTSService(
        settings=PiperTTSService.Settings(voice=cfg.tts_voice_en),
        use_cuda=False,
    )
    tts_ru = PiperTTSService(
        settings=PiperTTSService.Settings(voice=cfg.tts_voice_ru),
        use_cuda=False,
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad_processor,  # emits VADUserStarted/Stopped frames so Whisper can segment
            stt,
            user_agg,
            llm,
            # Bilingual TTS routing. Both branches receive every frame;
            # each LanguageFilter drops the frames meant for the other
            # voice, so only one Piper actually synthesises any given
            # sentence. The merged audio comes out the other side and
            # flows to transport.output() unchanged.
            ParallelPipeline(
                [LanguageFilter(Language.EN), tts_en],
                [LanguageFilter(Language.RU), tts_ru],
            ),
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

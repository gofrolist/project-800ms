# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Request/Response schemas for OpenAI-compatible API.
"""

from .schemas import (
    OpenAISpeechRequest,
    NormalizationOptions,
    ModelInfo,
    VoiceInfo,
)

__all__ = [
    "OpenAISpeechRequest",
    "NormalizationOptions",
    "ModelInfo",
    "VoiceInfo",
]

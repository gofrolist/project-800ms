# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Canonical OpenAI → Qwen voice mapping.

Local patch (adversarial/ADV-005 in ce-code-review). Upstream had two
independent copies of this table: one in the router's non-streaming
path, one hard-coded inside ``optimized_backend.generate_speech_streaming``.
The streaming path referenced voices (``Sophia``, ``Isabella``,
``Lily``) that don't exist in the model's voice catalog, so any
streaming request against those OpenAI aliases would map to an invalid
voice and fall back to defaults silently.

Centralising the table here lets both call paths import the same
object, so adding or renaming a voice in one spot can't drift the
other. The router re-exports ``VOICE_MAPPING`` under its original name
to keep the old API surface intact for anyone grep-hunting the repo.
"""

# OpenAI voice name → Qwen3 internal voice id. Must map to voices that
# actually exist in the config.yaml voices list for the deployed
# model. The authoritative list at the pinned upstream SHA is:
#   aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
VOICE_MAPPING: dict[str, str] = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Serena",
    "nova": "Aiden",
    "onyx": "Eric",
    "shimmer": "Dylan",
}

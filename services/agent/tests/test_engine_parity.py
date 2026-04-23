"""Cross-service parity check: the valid TTS engine set must agree between
the agent (two frozensets) and the API (Pydantic Literal).

The API lives in a separate Python environment (3.14) and can't be imported
from the agent test suite directly. Instead we parse the literal values
out of the API's schemas.py source file with a minimal regex — not a full
AST walk, because the declaration is a single-line Literal that's obvious
by shape.

Failure mode this guards against: someone adds a 5th engine to
``apps/api/schemas.py::TtsEngine`` but forgets to update
``services/agent/overrides.py`` or ``services/agent/pipeline.py`` (or vice
versa). The previous state — three separate declarations maintained "by
hand" per the comment in ``schemas.py`` — would silently let the API
accept a value the agent drops, or the agent accept a value the API
rejects, with no cross-service test catching the drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from overrides import _VALID_TTS_ENGINES as OVERRIDES_VALID_TTS_ENGINES
from pipeline import _VALID_TTS_ENGINES as PIPELINE_VALID_TTS_ENGINES


def _parse_api_literal_values() -> frozenset[str]:
    """Parse ``TtsEngine = Literal[...]`` from apps/api/schemas.py.

    Returns the frozenset of string literal values. Falls back to raising
    AssertionError with a clear message if the regex doesn't match — a
    structural refactor of the Literal (e.g., splitting across lines or
    switching to a computed type) would need this helper updated too.
    """
    repo_root = Path(__file__).resolve().parents[3]
    schemas_path = repo_root / "apps" / "api" / "schemas.py"
    if not schemas_path.is_file():
        # Not fatal — apps/api may be excluded in some checkouts; let the
        # test skip loudly rather than breaking the suite.
        import pytest

        pytest.skip(f"apps/api/schemas.py not found at {schemas_path}; skipping parity check")
    source = schemas_path.read_text(encoding="utf-8")
    match = re.search(r"TtsEngine\s*=\s*Literal\[([^\]]+)\]", source)
    assert match, (
        f"Couldn't locate 'TtsEngine = Literal[...]' in {schemas_path}. "
        f"If the declaration was refactored, update this helper."
    )
    # Pull the quoted strings out. Matches both single- and double-quoted.
    values = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
    assert values, f"TtsEngine Literal in {schemas_path} has no values"
    return frozenset(values)


def test_agent_whitelists_agree() -> None:
    """Both agent whitelists must carry the same engines.

    ``overrides.py`` validates per-session dispatch bodies; ``pipeline.py``
    validates the AgentConfig at boot. A drift between the two would let a
    session pass ``overrides.tts_engine`` checking but fail when
    ``AgentConfig.__post_init__`` runs — a non-obvious failure mode.
    """
    assert OVERRIDES_VALID_TTS_ENGINES == PIPELINE_VALID_TTS_ENGINES, (
        f"overrides.py _VALID_TTS_ENGINES={sorted(OVERRIDES_VALID_TTS_ENGINES)!r} "
        f"diverges from pipeline.py _VALID_TTS_ENGINES="
        f"{sorted(PIPELINE_VALID_TTS_ENGINES)!r}"
    )


def test_api_and_agent_whitelists_agree() -> None:
    """The API's TtsEngine Literal must match the agent whitelists.

    The API rejects unknown ``tts_engine`` values at request validation
    with a Pydantic Literal check. The agent's whitelist rejects the same
    set downstream. If they drift, one side will accept a value the other
    rejects, producing either 422s on legitimate requests or silent drops
    downstream.
    """
    api_values = _parse_api_literal_values()
    assert api_values == OVERRIDES_VALID_TTS_ENGINES, (
        f"API TtsEngine Literal={sorted(api_values)!r} diverges from "
        f"agent _VALID_TTS_ENGINES={sorted(OVERRIDES_VALID_TTS_ENGINES)!r}. "
        f"Update apps/api/schemas.py and both agent _VALID_TTS_ENGINES "
        f"frozensets together, or add a shared source of truth."
    )

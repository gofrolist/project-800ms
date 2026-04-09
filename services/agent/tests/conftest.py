"""Test fixtures for the agent service.

The agent's `pipeline.py` and `main.py` import pipecat (which transitively
loads CUDA libs at runtime), so most CI runners can't import them. Tests
here target leaf modules with no heavy imports — currently `env.py` only.
"""

from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

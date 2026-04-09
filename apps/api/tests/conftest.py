"""Test fixtures for the api service.

Sets fake LiveKit credentials *before* the production modules are imported,
because `settings.py` instantiates `Settings()` at module-import time and
would otherwise raise a ValidationError.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Required env before any application import.
os.environ.setdefault("LIVEKIT_API_KEY", "test-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-secret-min-32-chars-long-xxxxx")

# Make the service root importable as if `python main.py` were run from there.
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

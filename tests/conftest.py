"""Make `agent_stack` importable without requiring an editable install,
so `pytest` works straight after cloning. Prefer `pip install -e .` for
real development; this is a convenience fallback."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

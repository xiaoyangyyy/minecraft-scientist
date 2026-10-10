"""Post-freeze development diagnostics; no feedback into scientific planners."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the declared legacy bootstrap, without changing any frozen learner file.
import research.compact_campaigns  # noqa: F401

"""Development-only integration of generated structures and paid science."""

import sys
from pathlib import Path

# Direct `pytest research/structural_campaigns` imports this package by its
# directory name; make sibling namespace research modules available as well.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

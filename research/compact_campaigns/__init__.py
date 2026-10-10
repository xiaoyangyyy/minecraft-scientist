"""Lossless development evidence storage; not a new learner or reduced benchmark."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    ROOT,
    ROOT / "scripts",
    ROOT / "research" / "adaptive_automata",
    ROOT / "research" / "online_automata",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

"""Exact public-program behavioral equivalence, not a new learning algorithm."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .audit import verify_certificate
from .search import compare

__all__ = ["compare", "verify_certificate"]

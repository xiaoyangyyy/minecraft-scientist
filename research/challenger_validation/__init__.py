"""Public-model mutation validation baseline, not an originality claim."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Legacy research adapters use top-level imports. Establish their paths at the
# package boundary, before import sorting can place RecordedEpoch ahead of the
# shared runtime. This does not import fixtures, private tables or evaluators.
for directory in (
    ROOT / "scripts",
    ROOT / "research" / "adaptive_automata",
    ROOT / "research" / "online_automata",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

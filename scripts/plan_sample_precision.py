"""Print deterministic pre-data precision/power sensitivity JSON; no world input."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.sample_design.report import sensitivity_report

if __name__ == "__main__":
    print(json.dumps(sensitivity_report(), sort_keys=True, indent=2, allow_nan=False))

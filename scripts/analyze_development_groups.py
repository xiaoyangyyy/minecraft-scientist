"""Source-pinned, independently audited paired leakage-group diagnostics."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.group_analysis.campaign import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path)
    parser.add_argument("--validation-manifest-sha256")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261010)
    report = run(**vars(parser.parse_args()))
    print(
        {
            kind: {
                "fixtures": value["fixtures"],
                "leakage_groups": value["leakage_groups"],
                "rows": value["rows"],
            }
            for kind, value in report["analyses"].items()
        },
        flush=True,
    )

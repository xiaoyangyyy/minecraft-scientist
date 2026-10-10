"""Lossless full-development paid validation ablation; no private oracle planner."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.compact_campaigns.validation import run, verify_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    arguments = vars(parser.parse_args())
    if arguments.pop("verify_only"):
        arguments["directory"] = arguments.pop("output")
        print(verify_output(**arguments), flush=True)
    else:
        run(**arguments)

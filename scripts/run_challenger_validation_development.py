"""Paired paid-prefix validation ablation, complete development groups only."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.challenger_validation.campaign import run, verify_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = vars(parser.parse_args())
    if args.pop("verify_only"):
        args["directory"] = args.pop("output")
        print(verify_output(**args), flush=True)
    else:
        run(**args)

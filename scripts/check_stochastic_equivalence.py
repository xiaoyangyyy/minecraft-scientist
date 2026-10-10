"""Create or independently check the complete public stochastic-model proofs."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.stochastic_equivalence.campaign import run, verify_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.verify_only:
        if args.manifest_sha256 is None:
            parser.error("externally pinned manifest required")
        print(
            verify_output(args.output, manifest_sha256=args.manifest_sha256), flush=True
        )
    else:
        if args.manifest_sha256 is not None:
            parser.error("manifest pin only applies to verification")
        report = run(output=args.output, seed=args.seed)
        print(
            {
                "public_model_pairs": sum(r["pairs"] for r in report["rows"]),
                "physical_trials": 0,
            },
            flush=True,
        )

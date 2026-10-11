"""Real full noisy-family online shadow matrix or complete independent audit."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.noisy_adequacy.campaign import run, verify_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=4)
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
        result = run(
            output=args.output, seed=args.seed, budget=args.budget, horizon=args.horizon
        )
        print(
            {
                "runs": len(result["rows"]),
                "accepted_theories": None,
                "registry_changed": False,
            },
            flush=True,
        )

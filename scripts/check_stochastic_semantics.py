"""Run supplied-model stochastic component execution and independent raw replay."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.stochastic_ir.campaign import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    report = run(**vars(parser.parse_args()))
    print(
        {
            "execution_fixtures": len(report["rows"]),
            "resets": sum(r["resets"] for r in report["rows"]),
            "atomic_actions": sum(r["atomic_actions"] for r in report["rows"]),
            "discovery_completed": False,
        },
        flush=True,
    )

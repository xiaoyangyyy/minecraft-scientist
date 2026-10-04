"""Bounded temporary-data timing diagnostic; does not change test/model gates."""

import argparse
import json
import tempfile
import time
from pathlib import Path


def measure(root):
    directory = Path(tempfile.mkdtemp(prefix="msc-runtime-io-", dir=root))
    started = time.perf_counter()
    payload = b"minecraft-scientist temporary timing sample\n"
    reads = 0
    for group in range(20):
        folder = directory / f"group-{group:02d}"
        folder.mkdir()
        for item in range(5):
            path = folder / f"sample-{item:02d}.txt"
            path.write_bytes(payload)
            assert path.stat().st_size == len(payload)
            assert path.read_bytes() == payload
            reads += 1
    return {
        "directory": str(directory),
        "files": reads,
        "seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "bounded-runtime-io-diagnostic-v1",
        "samples": [measure(args.output), measure("/tmp")],
        "scope": "100 newly created small files per filesystem; no fixture reuse/deletion or scientific claim",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))

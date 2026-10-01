"""Fetch the official LoFTR outdoor weights into a new research-artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

URL = "https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
MAX_BYTES = 128 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "loftr_outdoor.ckpt.partial"
    digest = hashlib.sha256()
    count = 0
    try:
        with urlopen(URL, timeout=30) as response, path.open("xb") as output:
            if not response.geturl().startswith("https://cmp.felk.cvut.cz/"):
                raise ValueError("unexpected checkpoint redirect host")
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > MAX_BYTES:
                    raise ValueError("checkpoint download exceeded 128 MiB")
                digest.update(chunk)
                output.write(chunk)
        path = path.rename(args.output / "loftr_outdoor.ckpt")
        payload = {
            "url": URL,
            "bytes": count,
            "sha256": digest.hexdigest(),
            "checkpoint": path.name,
            "license_reference": "https://github.com/zju3dv/LoFTR/blob/master/LICENSE",
        }
    except Exception as error:
        payload = {
            "url": URL,
            "bytes_received": count,
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        (args.output / "download.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        raise
    (args.output / "download.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

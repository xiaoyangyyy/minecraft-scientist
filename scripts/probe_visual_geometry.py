"""Offline RGB geometry development probe; never loads evaluator telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.exploration.visual_geometry import (
    VisualGeometryConfig,
    estimate_visual_goal_bearing,
)


def _frames(directory: Path) -> list[np.ndarray]:
    result = []
    for path in sorted(directory.glob("view-*.png")):
        with Image.open(path) as image:
            result.append(np.asarray(image.convert("RGB")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite geometry evidence")
    source_paths = (
        Path(__file__),
        Path(estimate_visual_goal_bearing.__code__.co_filename),
    )
    source_bytes = {path.name: path.read_bytes() for path in source_paths}
    report_bytes = (args.run / "report.json").read_bytes()
    report = json.loads(report_bytes)
    config = VisualGeometryConfig()
    goal = _frames(args.run / "goal-reference")
    roles = [item["target_scan_role"] for item in report["candidate_results"]]
    source_role = report["candidate_results"][0]["replay"]["localizations"][-1][
        "scan_role"
    ]
    roles = [source_role, *roles]
    image_paths = tuple(
        sorted((args.run / "goal-reference").glob("view-*.png"))
    ) + tuple(
        path
        for role in roles
        for path in sorted((args.run / "scans" / role).glob("view-*.png"))
    )
    image_hashes = {
        path.relative_to(args.run).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in image_paths
    }
    evidence = {}
    for role in roles:
        result = estimate_visual_goal_bearing(
            goal, _frames(args.run / "scans" / role), config=config
        )
        evidence[role] = result.model_dump(mode="json")
        print(
            json.dumps(
                {"role": role, **result.model_dump(mode="json", exclude={"pairs"})}
            ),
            flush=True,
        )
    payload = {
        "suite": "visual_geometry_development",
        "config": config.model_dump(mode="json"),
        "evaluator_loaded": False,
        "results": evidence,
        "source_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in source_bytes.items()
        },
        "input_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "input_image_sha256": image_hashes,
    }
    if any(path.read_bytes() != source_bytes[path.name] for path in source_paths):
        raise RuntimeError("source changed during geometry development")
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest()
        != image_hashes[path.relative_to(args.run).as_posix()]
        for path in image_paths
    ):
        raise RuntimeError("input images changed during geometry development")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / "source-snapshot"
    snapshot.mkdir(exist_ok=False)
    for name, data in source_bytes.items():
        (snapshot / name).write_bytes(data)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

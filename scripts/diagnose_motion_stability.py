"""Freeze and probe pixel-only sensitivity of previously audited dense matches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_dense_correspondence import _numeric_equal

from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.motion_stability import (
    MotionStabilityConfig,
    assess_motion_stability,
)
from minecraft_scientist.exploration.two_view_motion import (
    TwoViewMotionConfig,
    estimate_two_view_motion,
)
from minecraft_scientist.exploration.visual_geometry import _opencv


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_sources() -> dict[str, Path]:
    return {
        name: Path(__file__).resolve().parent.parent / path
        for name, path in {
            "dense_correspondence.py": "src/minecraft_scientist/exploration/dense_correspondence.py",
            "two_view_motion.py": "src/minecraft_scientist/exploration/two_view_motion.py",
            "motion_stability.py": "src/minecraft_scientist/exploration/motion_stability.py",
            "visual_geometry.py": "src/minecraft_scientist/exploration/visual_geometry.py",
            "visual_odometry.py": "src/minecraft_scientist/exploration/visual_odometry.py",
            "diagnose_motion_stability.py": "scripts/diagnose_motion_stability.py",
            "audit_dense_correspondence.py": "scripts/audit_dense_correspondence.py",
        }.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    input_paths = {
        name: args.development / name
        for name in ("config.json", "report.json", "audit.json")
    }
    input_hashes = {name: file_hash(path) for name, path in input_paths.items()}
    original_config, original_report, original_audit = [
        json.loads(input_paths[name].read_bytes()) for name in input_paths
    ]
    if (
        not original_audit["valid"]
        or original_report["failure"] is not None
        or original_report["config_sha256"] != input_hashes["config.json"]
    ):
        raise ValueError("sensitivity inputs require valid audited dense predictions")
    sources = runtime_sources()
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    source_hashes = {name: file_hash(snapshot / name) for name in sources}
    config = MotionStabilityConfig()
    cv2 = _opencv()
    manifest = {
        "scope": "single-seed development sensitivity; no navigation promotion",
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "stability": config.model_dump(mode="json"),
        "opencv_version": cv2.__version__,
        "evaluation_thresholds": original_config["evaluation_thresholds"],
    }
    config_path = args.output / "config.json"
    config_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_hash = file_hash(config_path)
    camera = np.asarray(original_config["camera_intrinsics"])
    motion_config = TwoViewMotionConfig.model_validate(original_config["motion"])
    results = []
    for index, item in enumerate(original_report["results"]):
        matches = DenseCorrespondences.model_validate(item["correspondences"])
        base = estimate_two_view_motion(
            matches, camera, config=motion_config, seed=index
        )
        if not _numeric_equal(base.model_dump(mode="json"), item["motion"]):
            raise ValueError("cached base motion does not replay in this runtime")
        result = assess_motion_stability(
            matches, camera, config=config, motion_config=motion_config, seed=index
        )
        results.append({"pair": item["pair"], "motion": result.model_dump(mode="json")})
        print(
            json.dumps(
                {
                    "pair": item["pair"]["id"],
                    "supported": result.supported,
                    "supported_fraction": result.supported_fraction,
                    "direction_95": result.direction_deviation_95_degrees,
                    "rotation_95": result.rotation_deviation_95_degrees,
                }
            ),
            flush=True,
        )
    unchanged = (
        file_hash(config_path) == config_hash
        and all(
            file_hash(path) == input_hashes[name] for name, path in input_paths.items()
        )
        and all(
            file_hash(path) == source_hashes[name] for name, path in sources.items()
        )
    )
    report = {
        "config_sha256": config_hash,
        "results": results,
        "sources_and_inputs_unchanged": unchanged,
        "evaluator_file_loaded": False,
        "navigation_accepted": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

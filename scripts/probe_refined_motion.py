"""Preregister RGB refinement of frozen dense matches without reading telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from diagnose_motion_stability import file_hash
from diagnose_motion_stability import runtime_sources as sensitivity_sources
from PIL import Image

from minecraft_scientist.exploration.correspondence_refinement import (
    CorrespondenceRefinementConfig,
    refine_correspondences,
)
from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.motion_stability import (
    MotionStabilityConfig,
    assess_motion_stability,
)
from minecraft_scientist.exploration.two_view_motion import TwoViewMotionConfig
from minecraft_scientist.exploration.visual_geometry import _opencv


def runtime_sources() -> dict[str, Path]:
    paths = sensitivity_sources()
    paths["correspondence_refinement.py"] = Path(
        refine_correspondences.__code__.co_filename
    )
    paths["probe_refined_motion.py"] = Path(__file__)
    return paths


def frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def infer_pair(route: Path, original: dict, config: dict, index: int) -> dict:
    pair = original["pair"]
    refined = refine_correspondences(
        frame(route / "frames" / f"frame-{pair['first']:05d}.png"),
        frame(route / "frames" / f"frame-{pair['second']:05d}.png"),
        DenseCorrespondences.model_validate(original["correspondences"]),
        config=CorrespondenceRefinementConfig.model_validate(config["refinement"]),
    )
    motion = assess_motion_stability(
        refined.correspondences,
        np.asarray(config["camera_intrinsics"]),
        config=MotionStabilityConfig.model_validate(config["stability"]),
        motion_config=TwoViewMotionConfig.model_validate(config["motion"]),
        seed=index,
    )
    return {
        "pair": pair,
        "refined": refined.model_dump(mode="json"),
        "motion": motion.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {
        name: args.development / name
        for name in ("config.json", "report.json", "audit.json")
    }
    original_config, original_report, original_audit = [
        json.loads(inputs[name].read_bytes()) for name in inputs
    ]
    hashes = {name: file_hash(path) for name, path in inputs.items()}
    if (
        original_audit["valid"] is not True
        or original_report["config_sha256"] != hashes["config.json"]
        or file_hash(args.route / "report.json")
        != original_config["input_report_sha256"]
    ):
        raise ValueError("refinement requires the original audited route and matches")
    frame_paths = {
        name: args.route / "frames" / name for name in original_config["frame_sha256"]
    }
    if any(
        file_hash(path) != original_config["frame_sha256"][name]
        for name, path in frame_paths.items()
    ):
        raise ValueError("refinement frame hash mismatch")
    sources = runtime_sources()
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    source_hashes = {name: file_hash(snapshot / name) for name in sources}
    config = {
        "scope": "single-seed development refinement; not navigation acceptance",
        "input_sha256": hashes,
        "source_sha256": source_hashes,
        "frame_sha256": original_config["frame_sha256"],
        "route_report_sha256": original_config["input_report_sha256"],
        "refinement": CorrespondenceRefinementConfig().model_dump(mode="json"),
        "stability": MotionStabilityConfig().model_dump(mode="json"),
        "motion": original_config["motion"],
        "camera_intrinsics": original_config["camera_intrinsics"],
        "evaluation_thresholds": original_config["evaluation_thresholds"],
        "opencv_version": _opencv().__version__,
    }
    config_path = args.output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_hash = file_hash(config_path)
    results = []
    for index, original in enumerate(original_report["results"]):
        result = infer_pair(args.route, original, config, index)
        results.append(result)
        print(
            json.dumps(
                {
                    "pair": result["pair"]["id"],
                    "refined_matches": len(
                        result["refined"]["correspondences"]["points0"]
                    ),
                    "base_supported": result["motion"]["candidate"]["supported"],
                    "stable_supported": result["motion"]["supported"],
                    "direction_95": result["motion"]["direction_deviation_95_degrees"],
                    "rotation_95": result["motion"]["rotation_deviation_95_degrees"],
                }
            ),
            flush=True,
        )
    unchanged = (
        file_hash(config_path) == config_hash
        and all(file_hash(path) == hashes[name] for name, path in inputs.items())
        and all(
            file_hash(path) == source_hashes[name] for name, path in sources.items()
        )
        and all(
            file_hash(path) == config["frame_sha256"][name]
            for name, path in frame_paths.items()
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

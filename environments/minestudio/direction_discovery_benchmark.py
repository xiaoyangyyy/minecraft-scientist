"""Run a preregistered suite for RGB discovery of opaque move controls."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--mapping-seed", type=int, default=700)
    parser.add_argument("--mapping-seed-stride", type=int, default=2)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cases < 2:
        parser.error("at least two independently permuted cases are required")
    if args.mapping_seed_stride == 0:
        parser.error("mapping seed stride must be nonzero")
    output = args.output or ROOT / f"runs/direction-discovery-suite-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    source_files = (
        "environments/minestudio/direction_discovery_benchmark.py",
        "environments/minestudio/direction_discovery_probe.py",
        "environments/minestudio/navigation_probe.py",
        "src/minecraft_scientist/protocols/models.py",
        "src/minecraft_scientist/minecraft/minestudio.py",
        "src/minecraft_scientist/minecraft/visual_evidence.py",
    )
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files
    }
    save(
        output / "runtime-manifest.json",
        {
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("minestudio", "numpy", "pydantic", "pillow")
            },
            "source_sha256": source_hashes,
        },
    )
    snapshot = output / "source-snapshot"
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    cases = [
        {
            "seed": args.seed + index,
            "mapping_seed": args.mapping_seed + index * args.mapping_seed_stride,
        }
        for index in range(args.cases)
    ]
    settings = {
        "candidate_count": 4,
        "move_frames": 4,
        "score_threshold": 150.0,
        "max_moves": 20,
        "minimum_displacement": 5.0,
        "maximum_displacement": 9.5,
        "minimum_selected_index_variety": 2,
    }
    save(
        output / "preregistered.json",
        {
            "cases": cases,
            "settings": settings,
            "policy": (
                "test every opaque MOVE on reset-identical RGB trials, select the "
                "unique maximum score delta, then repeat it until first threshold"
            ),
            "required_success_rate": 1.0,
        },
    )
    preregistration_sha256 = hashlib.sha256(
        (output / "preregistered.json").read_bytes()
    ).hexdigest()

    results = []
    for index, case in enumerate(cases):
        directory = output / f"case-{index}"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("direction_discovery_probe.py")),
                "--seed",
                str(case["seed"]),
                "--mapping-seed",
                str(case["mapping_seed"]),
                "--move-frames",
                str(settings["move_frames"]),
                "--score-threshold",
                str(settings["score_threshold"]),
                "--max-moves",
                str(settings["max_moves"]),
                "--output",
                str(directory),
            ],
            check=True,
            timeout=600,
        )
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        calibration = json.loads(
            (directory / "calibration.json").read_text(encoding="utf-8")
        )
        displacement = report.get("evaluator_displacement")
        selected_index = next(
            trial_index
            for trial_index, trial in enumerate(calibration)
            if trial["handle_id"] == report["selected_handle"]
        )
        successful = all(
            (
                report.get("candidate_count") == settings["candidate_count"],
                report.get("selected_is_goal_directed") is True,
                report.get("unique_score_winner") is True,
                report.get("stopped_on_visual_threshold") is True,
                isinstance(displacement, (int, float)),
                settings["minimum_displacement"]
                <= displacement
                <= settings["maximum_displacement"],
            )
        )
        if not successful:
            raise RuntimeError(f"direction-discovery case failed: {directory}")
        results.append(
            {
                "case": index,
                "seed": case["seed"],
                "mapping_seed": case["mapping_seed"],
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "calibration_sha256": hashlib.sha256(
                    (directory / "calibration.json").read_bytes()
                ).hexdigest(),
                "evaluator_sha256": hashlib.sha256(
                    (directory / "evaluator-audit.json").read_bytes()
                ).hexdigest(),
                "successful": successful,
                "selected_handle_index": selected_index,
                "moves": report["moves"],
                "displacement": displacement,
            }
        )

    success_rate = sum(item["successful"] for item in results) / len(results)
    selected_index_variety = len(
        {item["selected_handle_index"] for item in results}
    )
    report = {
        "preregistration_sha256": preregistration_sha256,
        "cases": results,
        "success_rate": success_rate,
        "selected_index_variety": selected_index_variety,
        "opaque_direction_discovery": True,
        "scope": "Reset-controlled straight corridor with four opaque move handles.",
    }
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)
    if (
        success_rate != 1.0
        or selected_index_variety < settings["minimum_selected_index_variety"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

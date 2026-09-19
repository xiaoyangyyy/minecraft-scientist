"""Run a preregistered three-position active camera-search benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("negative", "center", "positive")


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"runs/camera-search-suite-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    source_files = (
        "environments/minestudio/camera_search_benchmark.py",
        "environments/minestudio/camera_search_probe.py",
        "src/minecraft_scientist/protocols/models.py",
        "src/minecraft_scientist/minecraft/minestudio.py",
        "src/minecraft_scientist/minecraft/visual_evidence.py",
    )
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files
    }
    save(output / "runtime-manifest.json", {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("minestudio", "numpy", "pydantic", "pillow")
        },
        "source_sha256": source_hashes,
    })
    snapshot = output / "source-snapshot"
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    cases = [
        {"target_view": view, "seed": args.seed + index}
        for index, view in enumerate(VIEWS)
    ]
    save(output / "preregistered.json", {
        "cases": cases,
        "view_set": list(VIEWS),
        "decision_rule": "argmax opaque central-camera brightness score",
        "required_accuracy": 1.0,
        "required_successful_look_actions_per_case": 5,
    })
    preregistration_sha256 = hashlib.sha256(
        (output / "preregistered.json").read_bytes()
    ).hexdigest()

    results = []
    for index, case in enumerate(cases):
        directory = output / f"case-{index}"
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("camera_search_probe.py")),
            "--seed", str(case["seed"]),
            "--target-view", case["target_view"],
            "--output", str(directory),
        ], check=True, timeout=180)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("camera_actions_valid") is not True:
            raise RuntimeError(f"camera action control failed: {directory}")
        results.append({
            "case": index,
            "seed": case["seed"],
            "report_sha256": hashlib.sha256(
                (directory / "report.json").read_bytes()
            ).hexdigest(),
            "correct": report.get("correct") is True,
            "camera_actions": report.get("camera_actions"),
            "camera_actions_valid": report.get("camera_actions_valid") is True,
        })

    accuracy = sum(result["correct"] for result in results) / len(results)
    report = {
        "preregistration_sha256": preregistration_sha256,
        "cases": results,
        "accuracy": accuracy,
        "all_camera_actions_valid": all(
            result["camera_actions"] == 5 and result["camera_actions_valid"]
            for result in results
        ),
        "active_camera_search": True,
        "scope": "Three fixture-defined yaw views and one brightness-defined marker.",
    }
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)
    if accuracy != 1.0 or not report["all_camera_actions_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Run a preregistered multi-seed closed-loop visual navigation benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cases < 2:
        parser.error("at least two independently seeded cases are required")
    output = args.output or ROOT / f"runs/navigation-suite-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    source_files = (
        "environments/minestudio/navigation_benchmark.py",
        "environments/minestudio/navigation_probe.py",
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

    cases = [{"seed": args.seed + index} for index in range(args.cases)]
    settings = {
        "score_threshold": 150.0,
        "move_frames": 4,
        "max_moves": 20,
        "minimum_displacement": 5.0,
        "maximum_displacement": 9.5,
    }
    save(output / "preregistered.json", {
        "cases": cases,
        "settings": settings,
        "policy": "repeat opaque MOVE until central RGB score first reaches threshold",
        "required_success_rate": 1.0,
    })
    preregistration_sha256 = hashlib.sha256(
        (output / "preregistered.json").read_bytes()
    ).hexdigest()

    results = []
    for index, case in enumerate(cases):
        directory = output / f"case-{index}"
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("navigation_probe.py")),
            "--seed", str(case["seed"]),
            "--score-threshold", str(settings["score_threshold"]),
            "--move-frames", str(settings["move_frames"]),
            "--max-moves", str(settings["max_moves"]),
            "--output", str(directory),
        ], check=True, timeout=180)
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        displacement = report.get("evaluator_displacement")
        successful = all((
            report.get("stopped_on_visual_threshold") is True,
            report.get("all_move_actions_valid") is True,
            isinstance(displacement, (int, float)),
            settings["minimum_displacement"] <= displacement
            <= settings["maximum_displacement"],
        ))
        if not successful:
            raise RuntimeError(f"navigation case failed: {directory}")
        results.append({
            "case": index,
            "seed": case["seed"],
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "successful": successful,
            "moves": report["moves"],
            "displacement": displacement,
        })

    success_rate = sum(item["successful"] for item in results) / len(results)
    report = {
        "preregistration_sha256": preregistration_sha256,
        "cases": results,
        "success_rate": success_rate,
        "closed_loop_visual_navigation": True,
        "scope": "Straight fixture corridor with one opaque forward handle.",
    }
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)
    if success_rate != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Bounded movement-duration ablation over the unchanged natural RGB capturer.

The capture entry point lacks duration CLI arguments. This wrapper supplies one
validated config through a process-local constructor hook, records that actual
constructor call, and restores the alias/argv even on startup or capture failure.
No SDK, camera profile, terrain or control mapping is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(capture, config: MotionProbeConfig, output: Path):
    directory = output.with_name(output.name + ".duration-profile")
    if output.exists() or directory.exists():
        raise FileExistsError("refusing to overwrite duration acquisition or profile")
    original_constructor, original_argv = capture.MotionProbeConfig, sys.argv
    # Only the current canonical constructor can be wrapped. Hooks cannot nest.
    if original_constructor is not MotionProbeConfig:
        raise ValueError("duration wrapper requires the canonical capture constructor")
    sources = capture.source_paths() | {Path(__file__).name: Path(__file__)}
    source_hashes = {name: digest(path) for name, path in sources.items()}
    manifest = {
        "capture_config": config.model_dump(mode="json"),
        "source_sha256": source_hashes,
        "scope": "fixed movement-duration development; no telemetry-based action selection",
    }
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    manifest_path = directory / "config.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen = digest(manifest_path)
    calls, failure = [], None

    def configured(**kwargs):
        actual = original_constructor(**kwargs, movement_ticks=config.movement_ticks)
        if actual != config:
            raise ValueError(
                "capture constructor differs from preregistered duration config"
            )
        calls.append(actual.model_dump(mode="json"))
        return actual

    try:
        capture.MotionProbeConfig = configured
        sys.argv = [
            capture.__file__,
            "--seed",
            str(config.seed),
            "--observation-size",
            str(config.observation_size),
            "--time",
            config.time_of_day,
            "--weather",
            config.weather,
            "--output",
            str(output),
        ]
        capture.main()
    except (Exception, SystemExit) as error:  # noqa: BLE001 -- failed acquisition is retained
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        capture.MotionProbeConfig, sys.argv = original_constructor, original_argv
    checks = {
        "single_declared_constructor": calls == [config.model_dump(mode="json")],
        "process_settings_restored": capture.MotionProbeConfig is original_constructor
        and sys.argv is original_argv,
        "preregistration_unchanged": digest(manifest_path) == frozen,
        "source_bytes_unchanged": all(
            digest(path) == source_hashes[name] for name, path in sources.items()
        ),
        "capture_completed": failure is None
        and (output / "report.json").is_file()
        and (output / "config.json").is_file(),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "config_sha256": frozen,
        "constructor_calls": calls,
        "failure": failure,
        "capture_config_sha256": digest(output / "config.json")
        if (output / "config.json").is_file()
        else None,
        "capture_report_sha256": digest(output / "report.json")
        if (output / "report.json").is_file()
        else None,
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--observation-size", type=int, default=448)
    parser.add_argument("--time", choices=("noon", "midnight"), default="noon")
    parser.add_argument("--weather", choices=("clear", "thunder"), default="clear")
    parser.add_argument("--short-move", type=int, default=6)
    parser.add_argument("--long-move", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = MotionProbeConfig(
        seed=args.seed,
        observation_size=args.observation_size,
        time_of_day=args.time,
        weather=args.weather,
        movement_ticks=(args.short_move, args.long_move),
    )
    import capture_controlled_rgb_motion as capture

    report = run(capture, config, args.output)
    print(
        json.dumps({"profile_passed": report["passed"], "failure": report["failure"]})
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

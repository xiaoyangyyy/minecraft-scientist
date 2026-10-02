"""Execute a publicly replayed prospective plan; never select using telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from active_capture_common import raw_bindings
from controlled_motion_common import digest
from plan_active_rgb_motion import validate_plan


def run(capture, plan_directory, output):
    profile = output.with_name(output.name + ".active-profile")
    if output.exists() or profile.exists():
        raise FileExistsError("refusing to overwrite active capture or profile")
    config = validate_plan(plan_directory)
    original_constructor, original_argv = capture.MotionProbeConfig, sys.argv
    with raw_bindings(capture, source_function="source_paths"):
        sources = capture.source_paths()
    paths = {
        "plan_config": plan_directory / "config.json",
        "plan_result": plan_directory / "plan.json",
    }
    manifest = {
        "plan_directory": plan_directory.resolve().relative_to(ROOT).as_posix(),
        "plan_sha256": {name: digest(path) for name, path in paths.items()},
        "capture_config": config.model_dump(mode="json"),
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "scope": "prospective public-RGB action selection; no online adaptation or navigation acceptance",
    }
    profile.mkdir(parents=True, exist_ok=False)
    snapshot = profile / "source-snapshot"
    snapshot.mkdir()
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
    manifest_path = profile / "config.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen, calls, failure = digest(manifest_path), [], None
    try:
        with raw_bindings(capture, source_function="source_paths"):
            constructor = capture.MotionProbeConfig

            def configured(**kwargs):
                actual = constructor(**kwargs, handle_pairs=config.handle_pairs)
                if actual != config:
                    raise ValueError(
                        "active constructor differs from publicly selected plan"
                    )
                calls.append(actual.model_dump(mode="json"))
                return actual

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
    except (Exception, SystemExit) as error:  # noqa: BLE001 -- preserve partial failures
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        sys.argv = original_argv
    checks = {
        "single_declared_constructor": calls == [config.model_dump(mode="json")],
        "process_settings_restored": capture.MotionProbeConfig is original_constructor
        and sys.argv is original_argv,
        "preregistration_unchanged": digest(manifest_path) == frozen,
        "source_bytes_unchanged": all(
            digest(path) == manifest["source_sha256"][name]
            for name, path in sources.items()
        ),
        "plan_bytes_unchanged": all(
            digest(path) == manifest["plan_sha256"][name]
            for name, path in paths.items()
        ),
        "capture_completed": failure is None
        and (output / "config.json").is_file()
        and (output / "report.json").is_file(),
    }
    result = {
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
    (profile / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import capture_controlled_rgb_motion as capture

    result = run(capture, args.plan, args.output)
    print(
        json.dumps(
            {"profile_passed": result["passed"], "failure": result["failure"]},
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

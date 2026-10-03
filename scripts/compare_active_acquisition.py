"""Preregister two bounded live acquisition strategies and retain both outcomes.

Same development seed and action/fitting caps, separate fresh rendered sessions.
This is not an identical-image/compute comparison or a statistical success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from minecraft_scientist.exploration.active_causal_map import ActiveMapPolicy

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("fixed_cap", "past_support")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    package = ROOT / "src/minecraft_scientist"
    paths = {
        "package/" + path.relative_to(package).as_posix(): path
        for path in sorted(package.rglob("*.py"))
    }
    for name in (
        "scripts/compare_active_acquisition.py",
        "scripts/audit_active_causal_map.py",
        "scripts/audit_runtime_camera.py",
        "environments/minestudio/capture_runtime_camera.py",
        "environments/minestudio/camera_audit_agent/CameraAudit.java",
        "environments/minestudio/camera_audit_agent/MANIFEST.MF",
        "environments/minestudio/capture_controlled_rgb_motion.py",
        "environments/minestudio/navigation_probe.py",
        "pyproject.toml",
        "uv.lock",
    ):
        paths[name] = ROOT / name
    return paths


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def execute(argv, logfile, seconds):
    if sys.platform != "linux":
        raise RuntimeError(
            "live comparison requires the installed optional Linux runtime"
        )
    # GNU timeout owns this process group; no process-name/broad machine cleanup.
    with logfile.open("wb") as log:
        return subprocess.run(
            ["/usr/bin/timeout", "-k", "30s", f"{seconds}s", *argv],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=seconds + 60,
        ).returncode


def condition_summary(strategy, raw, capture_code, audit_code, *, seed):
    declared = (
        json.loads((raw / "config.json").read_bytes())
        if (raw / "config.json").exists()
        else {}
    )
    capture = (
        json.loads((raw / "report.json").read_bytes())
        if (raw / "report.json").exists()
        else {}
    )
    audited = (
        json.loads((raw / "audit.json").read_bytes())
        if (raw / "audit.json").exists()
        else {}
    )
    active = (
        (json.loads((raw / "active-acquisition.json").read_bytes()) or {}).get(
            "acquisition"
        )
        if (raw / "active-acquisition.json").exists()
        else None
    )
    active = active or {}
    evaluated = audited.get("evaluation") or {}
    camera, body = evaluated.get("rendered") or {}, evaluated.get("body") or {}
    policy_matched = (
        declared.get("seed") == seed
        and declared.get("active_policy")
        == ActiveMapPolicy(control_strategy=strategy).model_dump(mode="json")
        == active.get("policy")
        and audited.get("control_strategy") == strategy
    )
    valid = (
        capture_code == 0
        and capture.get("completed") is True
        and audit_code == 0
        and audited.get("valid") is True
        and policy_matched
    )
    return {
        "strategy": strategy,
        "capture_returncode": capture_code,
        "audit_returncode": audit_code,
        "capture_completed": capture.get("completed") is True,
        "audit_valid": audited.get("valid") is True,
        "policy_and_seed_matched": policy_matched,
        "integrity_passed": valid,
        "terminal": active.get("terminal"),
        "failure": capture.get("failure") or active.get("failure"),
        "first_control_end_tick": active.get("first_control_end_tick"),
        "fit_count": active.get("fit_count"),
        "acquisition_attempts": len(active.get("acquisition_attempts", [])),
        "requested_policy_actions": len(active.get("steps", [])),
        "pixel_motion_supported": valid and active.get("terminal") == "validated",
        "useful_rendered_motion": valid
        and camera.get("translated") is True
        and camera.get("correct") is True,
        "useful_body_motion": valid
        and body.get("translated") is True
        and body.get("correct") is True,
        "evaluation": evaluated if valid else None,
        "navigation_accepted": False,
    }


def run(output, engine, *, seed=201, runner=None):
    if output.exists():
        raise FileExistsError("refusing to overwrite active acquisition comparison")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("comparison requires a bounded nonnegative seed")
    engine = engine.resolve(strict=True)
    runner = runner or execute
    code = sources()
    config = {
        "schema": "active-acquisition-comparison-v1",
        "seed": seed,
        "conditions": [
            {
                "strategy": strategy,
                "policy": ActiveMapPolicy(control_strategy=strategy).model_dump(
                    mode="json"
                ),
            }
            for strategy in STRATEGIES
        ],
        "scope": "two fresh same-seed development sessions; shared budget caps, not identical images or compute; no held-out or statistical navigation claim",
        "engine_sha256": digest(engine),
        "source_sha256": {name: digest(path) for name, path in code.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, path in code.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(output / "config.json", config)
    config_hash = digest(output / "config.json")
    conditions = []
    for strategy in STRATEGIES:
        directory = output / strategy
        directory.mkdir()
        raw = directory / "raw"
        capture_code, audit_code, failure = None, None, None
        try:
            if not all(
                digest(path) == config["source_sha256"][name]
                for name, path in code.items()
            ):
                raise ValueError("frozen sources changed before condition dispatch")
            capture_code = runner(
                [
                    sys.executable,
                    str(ROOT / "environments/minestudio/capture_runtime_camera.py"),
                    "--active-causal",
                    "--acquisition-strategy",
                    strategy,
                    "--seed",
                    str(seed),
                    "--output",
                    str(raw),
                ],
                directory / "capture.log",
                900,
            )
            if capture_code == 0:
                audit_code = runner(
                    [
                        sys.executable,
                        str(ROOT / "scripts/audit_active_causal_map.py"),
                        str(raw),
                        "--engine-jar",
                        str(engine),
                        "--output",
                        str(raw / "audit.json"),
                    ],
                    directory / "audit.log",
                    300,
                )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            failure = {"type": type(error).__name__, "message": str(error)}
        try:
            result = condition_summary(
                strategy, raw, capture_code, audit_code, seed=seed
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            result = {
                "strategy": strategy,
                "integrity_passed": False,
                "failure": {"type": type(error).__name__, "message": str(error)},
                "navigation_accepted": False,
            }
        if failure:
            result["dispatch_failure"] = failure
            result["integrity_passed"] = False
        conditions.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    checks = {
        "all_registered_conditions_retained": [item["strategy"] for item in conditions]
        == list(STRATEGIES),
        "all_captures_and_audits_complete": all(
            item["integrity_passed"] for item in conditions
        ),
        "sources_unchanged": all(
            digest(path) == config["source_sha256"][name] for name, path in code.items()
        ),
        "engine_unchanged": digest(engine) == config["engine_sha256"],
        "preregistration_unchanged": digest(output / "config.json") == config_hash,
        "source_snapshots_verified": all(
            digest(output / "source-snapshot" / name) == value
            for name, value in config["source_sha256"].items()
        ),
    }
    report = {
        "integrity_passed": all(checks.values()),
        "checks": checks,
        "conditions": conditions,
        "registered_episode_count": len(STRATEGIES),
        "navigation_accepted": False,
        "budget_caps_matched": True,
        "compute_matched": False,
        "images_matched": False,
        "statistical_claim": False,
        "config_sha256": config_hash,
        "artifact_sha256": {
            path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--engine-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output.resolve(), args.engine_jar, seed=args.seed)
    print(
        json.dumps(
            {"integrity_passed": result["integrity_passed"], "checks": result["checks"]}
        )
    )
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

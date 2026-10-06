"""Known-public replay of a new monitor; never a fresh control/arrival result.

The original complete rejected future is preserved. A counterfactual WAIT may
consume an old receipt only if its kind/handle match; otherwise recovery is
unobserved rather than invented. No evaluator pose files are opened; the saved
audit is consulted only for source/artifact attestation, not numerical scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from minecraft_scientist.exploration import public_map_risk as risk
from minecraft_scientist.exploration.map_recovery import _digest
from minecraft_scientist.exploration.settled_map_control import PublicControlStep


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordedControlGap(ValueError):
    pass


def recorded_receipt(steps, images, cursor, kind, handle):
    """Never reinterpret an old action as an unobserved counterfactual one."""
    if type(cursor) is not int or not 0 <= cursor < len(steps):
        raise RecordedControlGap("no remaining observed action for requested control")
    step = steps[cursor]
    if (step["requested_kind"], step["requested_handle"]) != (kind, handle):
        raise RecordedControlGap(
            "original actions cannot simulate the new requested control"
        )
    return PublicControlStep(
        frame=images[step["absolute_frame"]],
        executed_kind=step["executed_kind"],
        executed_handle=step["executed_handle"],
        compliant=step["compliant"],
        ticks=step["ticks"],
    ), cursor + 1


def replay(route, output):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.is_relative_to(route):
        raise FileExistsError("requires fresh output outside original capture")
    inputs = {}

    def bound(name):
        if (
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
        ):
            raise ValueError("replay requires explicit relative public/source paths")
        path = route / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(route):
            raise ValueError("replay input escapes the original capture")
        inputs[path] = digest(path)
        return path

    paths = [
        bound(name)
        for name in (
            "config.json",
            "report.json",
            "keyframe-acquisition.json",
            "audit.json",
            "audit.source.py",
        )
    ]
    config, capture, public, audit = [
        json.loads(path.read_bytes()) for path in paths[:4]
    ]
    if (
        config["schema"]
        != "walking-oriented-long-wait-safe-seed-stable-keyframe-camera-projection-v1"
        or capture["completed"] is not True
        or capture["failure"] is not None
        or capture["config_sha256"] != inputs[paths[0]]
        or capture["artifact_sha256"][paths[2].name] != inputs[paths[2]]
        or not capture["checks"]
        or not all(value is True for value in capture["checks"].values())
        or audit["valid"] is not True
        or not audit["checks"]
        or not all(value is True for value in audit["checks"].values())
        or audit["audit_snapshot_sha256"] != inputs[paths[4]]
    ):
        raise ValueError("requires an unchanged independently audited walking capture")
    acquisition = public["acquisition"]
    frozen, future = acquisition["frozen"], acquisition["future"]
    if (
        frozen is None
        or future is None
        or _digest(frozen) != acquisition["frozen_sha256"]
    ):
        raise ValueError("requires the original sealed map and full consumed future")
    if acquisition["terminal"] != audit["policy_terminal"]:
        raise ValueError("original terminal differs from saved independent audit")
    start, end = frozen["frozen_at_frame"], future["absolute_frame"]
    steps = [step for step in acquisition["steps"] if step["absolute_frame"] > start]
    if [step["absolute_frame"] for step in steps] != list(range(start + 1, end + 1)):
        raise ValueError("replay cannot omit or reorder original future actions")

    source_root = Path(risk.__file__).resolve().parents[3]
    package = source_root / "src/minecraft_scientist"
    extra = {
        "capture_runtime_camera.py": "environments/minestudio/capture_runtime_camera.py",
        "CameraAudit.java": "environments/minestudio/camera_audit_agent/CameraAudit.java",
        "MANIFEST.MF": "environments/minestudio/camera_audit_agent/MANIFEST.MF",
        "probe-recorder.py": "environments/minestudio/capture_controlled_rgb_motion.py",
        "position-audit.py": "environments/minestudio/navigation_probe.py",
    }
    source_paths = {
        "new/replay.source.py": Path(__file__),
        "new/monitor.source.py": Path(risk.__file__),
    }
    original_project_count = 0
    for name, expected in config["source_sha256"].items():
        snapshot = bound("source-snapshot/" + name)
        if inputs[snapshot] != expected:
            raise ValueError("original capture source snapshot changed")
        if name.startswith("package/"):
            path = package / name[len("package/") :]
        elif name in extra:
            path = source_root / extra[name]
        elif name in ("sdk-entry.py", "sdk-launcher.py"):
            continue  # External SDK binding is snapshot-only, never a new installation claim.
        else:
            raise ValueError("unknown original source binding")
        if (
            not path.resolve(strict=True).is_relative_to(source_root)
            or digest(path) != expected
        ):
            raise ValueError("new monitor changed an original capture dependency")
        original_project_count += 1
        source_paths["original/" + name] = path
    sources = {name: digest(path) for name, path in source_paths.items()}
    images = {}
    for slot in frozen["index"]["frame_ordinals"] + list(range(start + 1, end + 1)):
        name = f"frames/frame-{slot:05d}.png"
        path = bound(name)
        if inputs[path] != capture["artifact_sha256"][name]:
            raise ValueError("public RGB differs from original capture ledger")
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError("replay cannot transform original RGB")
            images[slot] = np.array(image)
    declaration = {
        "schema": "known-public-map-risk-replay-registration-v1",
        "capture_path": str(route),
        "source_root": str(source_root),
        "original_source_snapshot_count": len(config["source_sha256"]),
        "original_current_project_count": original_project_count,
        "source_sha256": sources,
        "input_sha256": {
            path.relative_to(route).as_posix(): value for path, value in inputs.items()
        },
        "policy": risk.MapRiskPolicy().model_dump(mode="json"),
        "scope": "Counterfactual component replay of known public observations, not a new native launch, intervention, navigation, independent generalization or promotion of an earlier original future.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_sha = digest(output / "config.json")
    for name, path in source_paths.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    monitor = risk.FrozenMapMonitor(
        frozen, [images[slot] for slot in frozen["index"]["frame_ordinals"]]
    )
    cursor = 0

    def receipt(step):
        return PublicControlStep(
            frame=images[step["absolute_frame"]],
            executed_kind=step["executed_kind"],
            executed_handle=step["executed_handle"],
            compliant=step["compliant"],
            ticks=step["ticks"],
        )

    while cursor < len(steps) and monitor.preflight(
        steps[cursor]["requested_kind"], steps[cursor]["requested_handle"]
    ):
        step = steps[cursor]
        monitor.observe(
            receipt(step),
            kind=step["requested_kind"],
            handle=step["requested_handle"],
            absolute_frame=step["absolute_frame"],
        )
        cursor += 1
    detected = monitor.journal
    gap = []

    def replay_wait(kind, handle):
        nonlocal cursor
        try:
            response, cursor = recorded_receipt(steps, images, cursor, kind, handle)
            return response
        except RecordedControlGap:
            gap.append(
                {
                    "next_original_frame": steps[cursor]["absolute_frame"]
                    if cursor < len(steps)
                    else None,
                    "requested_kind": kind,
                    "requested_handle": handle,
                    "reason": "unobserved matching counterfactual control; no invented receipt",
                }
            )
            raise

    recovery = None
    if monitor.state in ("challenged", "support_lost", "geometry_failed"):
        recovery = monitor.recover_observation(replay_wait).journal
    checks = {
        "all_known_inputs_unchanged": all(
            digest(path) == value for path, value in inputs.items()
        ),
        "active_sources_unchanged": all(
            digest(source_paths[name]) == value for name, value in sources.items()
        ),
        "saved_sources_match": all(
            digest(output / "source-snapshot" / name) == value
            for name, value in sources.items()
        ),
        "original_frozen_map_unchanged": _digest(frozen)
        == acquisition["frozen_sha256"],
        "configuration_unchanged": digest(output / "config.json") == config_sha,
    }
    result = {
        "schema": "known-public-map-risk-replay-v1",
        "config_sha256": config_sha,
        "checks": checks,
        "passed": all(checks.values()),
        "monitor": detected,
        "recovery": recovery,
        "counterfactual_control_gaps": gap,
        "unconsumed_original_future": steps[cursor:],
        "original_future": {
            "interval": [start, end],
            "terminal": acquisition["terminal"],
            "future_tracks": future["index"]["tracks"],
            "candidates": future["candidates"],
        },
        "original_result_replaced": False,
        "fresh_native_launch": False,
        "motion_validated": False,
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise ValueError("known replay input/source changed during analysis")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.route, args.output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "monitor_terminal": report["monitor"]["terminal"],
                "first_problem_frame": report["monitor"]["records"][-1][
                    "absolute_frame"
                ]
                if report["monitor"]["terminal"] != "observable"
                else None,
                "recovery_terminal": (report["recovery"] or {}).get("terminal"),
                "counterfactual_control_gaps": len(
                    report["counterfactual_control_gaps"]
                ),
                "navigation_accepted": False,
            }
        )
    )

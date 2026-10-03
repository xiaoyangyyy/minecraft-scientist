"""Public-only joint-map replay followed by independent rendered/body evaluation.

Every preregistered bootstrap and new-control pixel test is retained. Matrices
are evaluator-only, never calibration or fitting input. No navigation promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_runtime_camera as runtime
import numpy as np

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.evaluation.rendered_camera import score_rendered_motion
from minecraft_scientist.exploration.joint_multiview_map import (
    JointMapConfig,
    JointMapEvidence,
    fit_joint_map,
    validate_joint_future,
)
from minecraft_scientist.exploration.online_landmark_probe import (
    calibrate_reference_camera,
)
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    track_frame_chain,
)

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "two preregistered development trajectories; all bootstrap candidates; rendered and body truth scored separately; no navigation acceptance or held-out generalization"
THRESHOLDS = {
    "minimum_translation": 0.15,
    "maximum_direction_error": 20,
    "maximum_rotation_error": 5,
}


def sources():
    package = ROOT / "src/minecraft_scientist"
    return {
        "package/" + path.relative_to(package).as_posix(): path
        for path in sorted(package.rglob("*.py"))
    } | {path.name: path for path in (Path(__file__), Path(runtime.__file__))}


def inputs(route):
    # Listing the private ledger names is harmless; opening their bytes is not.
    paths = [
        route / name
        for name in ("config.json", "report.json", "events.jsonl", "camera-audit.jar")
    ]
    paths += list((route / "frames").glob("*.png"))
    paths += list((route / "evaluator-native").glob("*.png"))
    paths += [path for path in (route / "source-snapshot").rglob("*") if path.is_file()]
    return {path.relative_to(route).as_posix(): path for path in paths}


def unchanged_pixels(previous, current, retained):
    if previous is None or not retained:
        return
    before = {identity: slot for slot, identity in enumerate(previous.track_ids)}
    after = {identity: slot for slot, identity in enumerate(current.track_ids)}
    ids = sorted(retained)
    if not np.allclose(
        np.array(previous.pixels)[:, [before[identity] for identity in ids]],
        np.array(current.pixels)[:-1, [after[identity] for identity in ids]],
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("chronological track prefix revised past pixels")


def infer(route, engine):
    public = runtime.public_data(route, engine)
    if not public["valid"]:
        raise ValueError(
            "public capture validation failed: " + json.dumps(public["checks"])
        )
    config = public["config"]
    images = [
        runtime.load_image(route / "frames" / f"frame-{slot:05d}.png")
        for slot in range(public["count"])
    ]
    calibration = calibrate_reference_camera(
        [images[slot] for slot in config["reference_indices"]],
        tuple(config["render_size"]),
    )
    camera = np.array(calibration["camera_intrinsics"])
    temporal = TemporalTrackConfig(maximum_frames=64)
    results = []
    # All past maps in BOTH trials are frozen before ANY future fitting input.
    pending = []
    for number, (first, end, future_end) in enumerate(config["trial_intervals"]):
        retained, previous, prefixes = None, None, []
        for absolute in range(first + 1, end + 1):
            current = track_frame_chain(images[first : absolute + 1], config=temporal)
            retained = (
                set(current.track_ids)
                if retained is None
                else retained & set(current.track_ids)
            )
            unchanged_pixels(previous, current, retained)
            previous = current
            prefixes.append(current.model_dump(mode="json"))
        past = retain_tracks(previous, retained)
        candidates = [
            {
                "map": fit_joint_map(
                    past,
                    camera,
                    bootstrap_slot=slot,
                    seed=2000 + 5 * number,
                    config=JointMapConfig(),
                ).model_dump(mode="json"),
                "future": None,
            }
            for slot in range(2, past.frame_count)
        ]
        record = {
            "trial": number,
            "anchor_frame": first,
            "fitting_end_frame": end,
            "future_end_frame": future_end,
            "past_tracks": past.model_dump(mode="json"),
            "past_prefixes": prefixes,
            "candidates": candidates,
        }
        results.append(record)
        pending.append((record, past, previous, retained))
    frozen_maps = json.dumps([item["candidates"] for item in results], sort_keys=True)
    future_results = []
    for record, past, previous, retained in pending:
        first, end, future_end = (
            record[key]
            for key in ("anchor_frame", "fitting_end_frame", "future_end_frame")
        )
        prefixes = []
        for absolute in range(end + 1, future_end + 1):
            current = track_frame_chain(images[first : absolute + 1], config=temporal)
            retained &= set(current.track_ids)
            unchanged_pixels(previous, current, retained)
            previous = current
            prefixes.append(current.model_dump(mode="json"))
        extended = retain_tracks(previous, retained)
        record["future_tracks"], record["future_prefixes"] = (
            extended.model_dump(mode="json"),
            prefixes,
        )
        future_results.append(
            [
                validate_joint_future(
                    JointMapEvidence.model_validate(item["map"]),
                    past,
                    extended,
                    target_slots=(end + 2 - first, future_end - first),
                )
                for item in record["candidates"]
            ]
        )
    if (
        json.dumps([item["candidates"] for item in results], sort_keys=True)
        != frozen_maps
    ):
        raise ValueError("future observations mutated frozen candidates")
    for record, futures in zip(results, future_results, strict=True):
        for item, future in zip(record["candidates"], futures, strict=True):
            item["future"] = future
    if not all(
        runtime.digest(path) == value for path, value in public["frozen"].items()
    ):
        raise ValueError("public bytes changed during inference")
    return {"calibration": calibration, "trials": results, "navigation_accepted": False}


def policy():
    return {
        "schema": "rendered-motion-development-v1",
        "scope": SCOPE,
        "joint_config": JointMapConfig().model_dump(mode="json"),
        "temporal_config": TemporalTrackConfig(maximum_frames=64).model_dump(
            mode="json"
        ),
        "seeds": [2000, 2005],
        "bootstrap_rule": "all past offsets 2..10 in both trials before any future input",
        "thresholds": THRESHOLDS,
    }


def run(route, engine, output, *, infer_fn=None, policy_fn=None, sources_fn=None):
    infer_fn, policy_fn, sources_fn = (
        infer_fn or infer,
        policy_fn or policy,
        sources_fn or sources,
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite rendered-motion experiment")
    paths, code = inputs(route), sources_fn()
    config = policy_fn() | {
        "input_sha256": {name: runtime.digest(path) for name, path in paths.items()},
        "source_sha256": {name: runtime.digest(path) for name, path in code.items()},
        "engine_sha256": runtime.digest(engine),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, path in code.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "completed": False,
        "failure": None,
        "prediction": None,
        "navigation_accepted": False,
        "config_sha256": runtime.digest(output / "config.json"),
    }
    try:
        result["prediction"] = infer_fn(route, engine)
        result["completed"] = True
    except Exception as error:  # noqa: BLE001 -- retain rejected development runs
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["input_sources_unchanged"] = (
        all(
            runtime.digest(path) == config["input_sha256"][name]
            for name, path in paths.items()
        )
        and all(
            runtime.digest(path) == config["source_sha256"][name]
            for name, path in code.items()
        )
        and runtime.digest(engine) == config["engine_sha256"]
        and runtime.digest(output / "config.json") == result["config_sha256"]
    )
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def prediction(pose):
    if pose is None:
        return {"supported": False}
    shift = np.array(pose["translation"])
    if np.linalg.norm(shift) <= 1e-12:
        return {"supported": False}
    return {
        "supported": True,
        "relative_rotation": pose["rotation"],
        "unit_translation": (shift / np.linalg.norm(shift)).tolist(),
    }


def relative_pose(first, second):
    rotation = np.array(second["rotation"]) @ np.array(first["rotation"]).T
    shift = np.array(second["translation"]) - rotation @ np.array(first["translation"])
    return {"rotation": rotation.tolist(), "translation": shift.tolist()}


def evaluate_future(pose, future, first, end, score):
    """Keep gated policy outcomes AND rejected fitted-pose diagnostics distinct."""
    records = []
    for check, later in zip(future["frame_checks"], future["poses"], strict=True):
        absolute = first + check["frame_slot"]
        usable = later if check["supported"] else None
        records.append(
            {
                "absolute_frame": absolute,
                "pixel_check": check,
                "from_anchor": score(usable, first, absolute),
                "from_frozen_endpoint": score(
                    relative_pose(pose, usable) if usable is not None else None,
                    end,
                    absolute,
                ),
                "diagnostic_pose_from_anchor": score(later, first, absolute),
                "diagnostic_pose_from_frozen_endpoint": score(
                    relative_pose(pose, later) if later is not None else None,
                    end,
                    absolute,
                ),
                "diagnostic_is_policy_acceptance": False,
            }
        )
    return records


def registered_trials(predicted):
    return [item["trial"] for item in predicted["trials"]] == [0, 1] and all(
        [item["map"]["bootstrap_slot"] for item in trial["candidates"]]
        == list(range(2, 11))
        for trial in predicted["trials"]
    )


def audit(
    route,
    engine,
    experiment,
    *,
    infer_fn=None,
    policy_fn=None,
    sources_fn=None,
    denominator_fn=None,
):
    infer_fn, policy_fn, sources_fn = (
        infer_fn or infer,
        policy_fn or policy,
        sources_fn or sources,
    )
    denominator_fn = denominator_fn or registered_trials
    config = json.loads((experiment / "config.json").read_bytes())
    result = json.loads((experiment / "report.json").read_bytes())
    paths, code = inputs(route), sources_fn()
    checks = {
        "complete_public_prediction": result["completed"] is True
        and result["failure"] is None
        and result["input_sources_unchanged"] is True
        and result["navigation_accepted"] is False,
        "registered_policy": all(
            config.get(name) == value for name, value in policy_fn().items()
        ),
        "config_identity": result["config_sha256"]
        == runtime.digest(experiment / "config.json"),
        "current_sources": config["source_sha256"]
        == {name: runtime.digest(path) for name, path in code.items()},
        "complete_public_inputs": config["input_sha256"]
        == {name: runtime.digest(path) for name, path in paths.items()},
        "engine_identity": config["engine_sha256"] == runtime.digest(engine),
        "source_snapshots": config["source_sha256"]
        == {
            path.relative_to(experiment / "source-snapshot").as_posix(): runtime.digest(
                path
            )
            for path in (experiment / "source-snapshot").rglob("*")
            if path.is_file()
        },
    }
    frozen = {
        path: runtime.digest(path)
        for path in list(paths.values())
        + list(code.values())
        + [engine, experiment / "config.json", experiment / "report.json"]
        + [
            path
            for path in (experiment / "source-snapshot").rglob("*")
            if path.is_file()
        ]
    }
    if all(checks.values()):
        # Exact same-runtime numerical/structural replay, including every failure.
        checks["entire_public_inference_replayed"] = json.dumps(
            infer_fn(route, engine), sort_keys=True, allow_nan=False
        ) == json.dumps(result["prediction"], sort_keys=True, allow_nan=False)
        checks["all_registered_trials_and_candidates_retained"] = denominator_fn(
            result["prediction"]
        )
        checks["public_bytes_stable_before_private_access"] = all(
            runtime.digest(path) == value for path, value in frozen.items()
        )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # Only now may the evaluator access renderer matrices/player coordinates.
    private_names = ("evaluator-positions.json", "evaluator-camera.jsonl")
    capture_ledger = json.loads((route / "report.json").read_bytes())["artifact_sha256"]
    frozen.update(
        {route / name: runtime.digest(route / name) for name in private_names}
    )
    checks["private_bytes_match_capture_before_alignment"] = all(
        frozen[route / name] == capture_ledger[name] for name in private_names
    )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    physical = runtime.audit(route, engine)
    checks["independent_render_alignment"] = physical["valid"]
    if not physical["valid"]:
        return {
            "valid": False,
            "checks": checks,
            "evaluation": None,
            "camera_audit": physical,
        }
    positions = json.loads((route / "evaluator-positions.json").read_bytes())[
        "positions"
    ]
    cameras = physical["rendered_cameras"]

    def score(pose, first, second):
        predicted = prediction(pose)
        return {
            "rendered": score_rendered_motion(
                predicted, cameras[first], cameras[second], **THRESHOLDS
            ),
            "body": score_relative_motion(
                predicted, positions[first], positions[second], **THRESHOLDS
            ),
        }

    evaluation = []
    for trial in result["prediction"]["trials"]:
        first, end = trial["anchor_frame"], trial["fitting_end_frame"]
        records = []
        for candidate in trial["candidates"]:
            fitted, future = candidate["map"], candidate["future"]
            pose = fitted["poses"][-1] if fitted["supported"] else None
            future_scores = []
            if pose is not None:
                future_scores = evaluate_future(pose, future, first, end, score)
            records.append(
                {
                    "bootstrap_slot": fitted["bootstrap_slot"],
                    "supported": fitted["supported"],
                    "reason": fitted["reason"],
                    "past_motion": score(pose, first, end),
                    "future_status": future["status"],
                    "future_motion": future_scores,
                }
            )
        evaluated = {"trial": trial["trial"], "candidates": records}
        if "controller_decision" in trial:
            decision = trial["controller_decision"]
            evaluated["controller_gate"] = decision["instruction"]
            evaluated["controller_motion"] = score(
                {
                    "rotation": decision["relative_rotation"],
                    "translation": decision["unit_translation"],
                }
                if decision["unit_translation"] is not None
                else None,
                end,
                decision["absolute_frame"],
            )
        evaluation.append(evaluated)
    checks["all_bytes_stable_through_scoring"] = all(
        runtime.digest(path) == value for path, value in frozen.items()
    )
    valid = all(checks.values())
    return {
        "valid": valid,
        "checks": checks,
        "evaluation": evaluation if valid else None,
        "camera_audit": physical,
        "private_sha256": {name: frozen[route / name] for name in private_names},
        "navigation_accepted": False,
        "scope": config["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--engine-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit is None:
        result = run(args.route, args.engine_jar, args.output)
        print(
            json.dumps(
                {
                    "completed": result["completed"],
                    "failure": result["failure"],
                    "sources_unchanged": result["input_sources_unchanged"],
                }
            )
        )
        if not result["completed"] or not result["input_sources_unchanged"]:
            raise SystemExit(1)
    else:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite rendered-motion audit")
        result = audit(args.route, args.engine_jar, args.audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"valid": result["valid"], "checks": result["checks"]}))
        if not result["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

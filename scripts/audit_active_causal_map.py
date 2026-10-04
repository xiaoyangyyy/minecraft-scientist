"""Replay every live acquisition action before opening private camera evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_runtime_camera as runtime
import numpy as np

from minecraft_scientist.evaluation.gauge_registration import score_gauge_truth
from minecraft_scientist.evaluation.relative_motion import (
    relative_motion_truth,
    score_relative_motion,
)
from minecraft_scientist.evaluation.rendered_camera import (
    rendered_motion_truth,
    score_rendered_motion,
)
from minecraft_scientist.exploration.active_causal_map import (
    ActiveMapPolicy,
    acquire_causal_map,
)
from minecraft_scientist.exploration.map_gauge_bridge import GaugeBridgePolicy
from minecraft_scientist.exploration.map_recovery import (
    MapRecoveryPolicy,
    recover_public_map,
)
from minecraft_scientist.exploration.online_landmark_probe import (
    calibrate_reference_camera,
)
from minecraft_scientist.exploration.rolling_map_control import (
    RollingMapPolicy,
    maintain_public_maps,
)
from minecraft_scientist.exploration.settled_map_control import (
    PublicControlStep,
    SettledControlPolicy,
    run_settled_controls,
)
from minecraft_scientist.exploration.shared_anchor_map import SharedAnchorPolicy
from minecraft_scientist.exploration.visibility_acquisition import PublicProbeStep
from minecraft_scientist.protocols import TransitionRecord

THRESHOLDS = {
    "minimum_translation": 0.15,
    "maximum_direction_error": 20,
    "maximum_rotation_error": 5,
}


def _settled_registration(config):
    """A separate declared protocol; never enlarge the old capture budget."""
    return config.get("schema") in (
        "settled-causal-camera-projection-v1",
        "recovery-causal-camera-projection-v1",
        "maintenance-causal-camera-projection-v1",
        "registered-maintenance-camera-projection-v1",
        "registered-shared-anchor-camera-projection-v1",
    )


def _recovery_registration(config):
    return config.get("schema") in (
        "recovery-causal-camera-projection-v1",
        "maintenance-causal-camera-projection-v1",
        "registered-maintenance-camera-projection-v1",
        "registered-shared-anchor-camera-projection-v1",
    )


def _maintenance_registration(config):
    return config.get("schema") in (
        "maintenance-causal-camera-projection-v1",
        "registered-maintenance-camera-projection-v1",
        "registered-shared-anchor-camera-projection-v1",
    )


def _gauge_registration(config):
    return config.get("schema") in (
        "registered-maintenance-camera-projection-v1",
        "registered-shared-anchor-camera-projection-v1",
    )


def _shared_registration(config):
    return config.get("schema") == "registered-shared-anchor-camera-projection-v1"


def _frame_limit(config):
    if _maintenance_registration(config):
        return 201
    if _recovery_registration(config):
        return 193
    return 105 if _settled_registration(config) else 61


def registration(directory, config, report, definition):
    declared = config.get("active_policy")
    if not isinstance(declared, dict):
        raise TypeError("active registration requires a declared public policy")
    strategy = declared.get("control_strategy")
    if strategy not in ("past_support", "fixed_cap"):
        raise ValueError(
            "active registration requires an explicit known control strategy"
        )
    settings = ActiveMapPolicy(control_strategy=strategy)
    settled = _settled_registration(config)
    continuation = SettledControlPolicy()
    recovering = _recovery_registration(config)
    recovery_policy = MapRecoveryPolicy()
    maintaining = _maintenance_registration(config)
    maintenance_policy = RollingMapPolicy()
    gauging = _gauge_registration(config)
    gauge_policy = GaugeBridgePolicy()
    sharing = _shared_registration(config)
    shared_policy = SharedAnchorPolicy()
    baseline = [list(item) for item in definition.SCHEDULE[:48]]
    registered = {
        "valid": False,
        "schedule": baseline,
        "artifacts": ["active-acquisition.json"],
        "capture_checks": ["active_policy_completed"],
    }
    if settled:
        registered["capture_checks"].append("settled_policy_completed")
    if recovering:
        registered["capture_checks"].append("recovery_policy_completed")
    if maintaining:
        registered["capture_checks"].append("maintenance_policy_completed")
    if gauging:
        registered["capture_checks"].append("gauge_policy_completed")
    if sharing:
        registered["capture_checks"].append("shared_anchor_policy_completed")
    if (
        sharing
        and json.dumps(config.get("shared_anchor_policy"), sort_keys=True)
        != json.dumps(shared_policy.model_dump(mode="json"), sort_keys=True)
    ) or (not sharing and "shared_anchor_policy" in config):
        return registered
    if (
        gauging
        and json.dumps(config.get("gauge_policy"), sort_keys=True)
        != json.dumps(gauge_policy.model_dump(mode="json"), sort_keys=True)
    ) or (not gauging and "gauge_policy" in config):
        return registered
    if (
        maintaining
        and json.dumps(config.get("maintenance_policy"), sort_keys=True)
        != json.dumps(maintenance_policy.model_dump(mode="json"), sort_keys=True)
    ) or (not maintaining and "maintenance_policy" in config):
        return registered
    if (
        recovering
        and json.dumps(config.get("recovery_policy"), sort_keys=True)
        != json.dumps(recovery_policy.model_dump(mode="json"), sort_keys=True)
    ) or (not recovering and "recovery_policy" in config):
        return registered
    if settled and (
        strategy != "past_support"
        or json.dumps(config.get("settled_policy"), sort_keys=True)
        != json.dumps(continuation.model_dump(mode="json"), sort_keys=True)
    ):
        return registered
    if not (
        config["schema"]
        in (
            "active-causal-camera-projection-v2",
            "settled-causal-camera-projection-v1",
            "recovery-causal-camera-projection-v1",
            "maintenance-causal-camera-projection-v1",
            "registered-maintenance-camera-projection-v1",
            "registered-shared-anchor-camera-projection-v1",
        )
        and (settled or "settled_policy" not in config)
        and config["schedule"] == baseline
        and json.dumps(config["active_policy"], sort_keys=True)
        == json.dumps(settings.model_dump(mode="json"), sort_keys=True)
        and type(config["active_seed"]) is int
        and config["active_seed"] == 2000
        and config["reference_indices"] == list(definition.REFERENCE_INDICES)
        and config["trial_intervals"] == []
        and type(report["frame_count"]) is int
        and 50 <= report["frame_count"] <= _frame_limit(config)
    ):
        return registered
    count = report["frame_count"]
    paths = [
        directory / name
        for name in (
            "config.json",
            "report.json",
            "events.jsonl",
            "active-acquisition.json",
        )
    ]
    paths += [directory / "frames" / f"frame-{slot:05d}.png" for slot in range(count)]
    frozen = {path: runtime.digest(path) for path in paths}
    images = [
        runtime.load_image(directory / "frames" / f"frame-{slot:05d}.png")
        for slot in range(count)
    ]
    if any(image.shape != (448, 448, 3) for image in images):
        raise ValueError("active replay needs registered public RGB dimensions")
    events = [
        TransitionRecord.model_validate_json(line)
        for line in (directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    saved = json.loads((directory / "active-acquisition.json").read_bytes())
    calibration = calibrate_reference_camera(
        [images[slot] for slot in definition.REFERENCE_INDICES], (1280, 720)
    )
    cursor = 48
    requested = list(baseline)

    def execute(handle):
        nonlocal cursor
        if cursor >= len(events) or cursor + 1 >= len(images):
            raise ValueError(
                "saved run ended before the public policy requested an action"
            )
        event = events[cursor]
        requested.append(["move", handle])
        targets = [
            obj.object_id
            for obj in event.state_before.objects
            if tuple(kind.value for kind in obj.affordances) == ("move",)
        ]
        actual = (
            targets.index(event.executed_action.target_id)
            if event.executed_action.target_id in targets
            else -1
        )
        compliant = (
            event.executed_action.kind.value == "move"
            and event.intervention_compliance is True
            and event.event.success is True
            and event.controller_failure is None
            and event.event.duration == 1
        )
        cursor += 1
        return PublicProbeStep(
            images[cursor].copy(), actual, compliant, event.executed_action.ticks
        )

    replayed = {
        "calibration": calibration,
        "acquisition": acquire_causal_map(
            images[48].copy(),
            execute,
            calibration["camera_intrinsics"],
            anchor_frame=48,
            seed=config["active_seed"],
            policy=settings,
        ),
    }
    if settled:
        # Only already consumed acquisition frames enter the checkpoint. Later
        # frames become visible one at a time through executed public receipts.
        acquired_frames = [image.copy() for image in images[48 : cursor + 1]]

        def execute_control(kind, handle):
            nonlocal cursor
            if cursor >= len(events) or cursor + 1 >= len(images):
                raise ValueError("saved run ended before a settled control request")
            event = events[cursor]
            requested.append([kind, handle])
            targets = [
                obj.object_id
                for obj in event.state_before.objects
                if tuple(action.value for action in obj.affordances) == ("move",)
            ]
            actual = (
                targets.index(event.executed_action.target_id)
                if event.executed_action.target_id in targets
                else None
            )
            compliant = (
                event.intervention_compliance is True
                and event.event.success is True
                and event.controller_failure is None
                and event.event.duration == 1
            )
            cursor += 1
            return PublicControlStep(
                images[cursor].copy(),
                event.executed_action.kind.value,
                actual,
                compliant,
                event.executed_action.ticks,
            )

        replayed["settled"] = run_settled_controls(
            replayed["acquisition"],
            acquired_frames,
            execute_control,
            policy=continuation,
        )
        if recovering:
            parent = {
                key: replayed[key] for key in ("calibration", "acquisition", "settled")
            }
            replayed["recovery"] = recover_public_map(
                parent,
                images[cursor].copy(),
                execute_control,
                policy=recovery_policy,
            )
            if maintaining:
                recovered = replayed["recovery"]
                epoch_frames = (
                    images[recovered["new_anchor_frame"] : cursor + 1]
                    if recovered["acquisition"]
                    else []
                )
                replayed["maintenance"] = maintain_public_maps(
                    recovered,
                    epoch_frames,
                    execute_control,
                    calibration["camera_intrinsics"],
                    policy=maintenance_policy,
                    gauge_policy=gauge_policy if gauging else None,
                    shared_policy=shared_policy if sharing else None,
                )
    registered.update(
        valid=(
            json.dumps(saved, sort_keys=True, allow_nan=False)
            == json.dumps(replayed, sort_keys=True, allow_nan=False)
            and cursor == len(events) == count - 1
            and replayed["acquisition"]["failure"] is None
            and (not settled or replayed["settled"]["failure"] is None)
            and (not recovering or replayed["recovery"]["failure"] is None)
            and (not maintaining or replayed["maintenance"]["failure"] is None)
            and all(runtime.digest(path) == value for path, value in frozen.items())
        ),
        schedule=requested,
    )
    return registered


def _score_acquisition(active, positions, cameras):
    window, decision = active["snapshot"], active["controller_decision"]
    if window is None or decision is None or decision["instruction"] == "collect":
        return None
    first = active["anchor_frame"] + window["past"]["frame_count"] - 1
    last = decision["absolute_frame"]
    predicted = (
        {
            "supported": True,
            "relative_rotation": decision["relative_rotation"],
            "unit_translation": decision["unit_translation"],
        }
        if decision["instruction"] == "refresh"
        else {"supported": False}
    )
    return {
        "interval": [first, last],
        "rendered": score_rendered_motion(
            predicted, cameras[first], cameras[last], **THRESHOLDS
        ),
        "body": score_relative_motion(
            predicted, positions[first], positions[last], **THRESHOLDS
        ),
    }


def _score_continuation(continuation, positions, cameras):
    trials = []
    for trial in continuation["trials"]:
        first, last = trial["start_frame"], trial["end_frame"]
        scores = None
        if last is not None:
            if not (
                type(first) is int
                and type(last) is int
                and 0 <= first < last < len(positions)
            ):
                raise ValueError("settled trial has invalid public frame endpoints")
            predicted = trial["motion"] or {"supported": False}
            scores = {
                "rendered": score_rendered_motion(
                    predicted, cameras[first], cameras[last], **THRESHOLDS
                ),
                "body": score_relative_motion(
                    predicted, positions[first], positions[last], **THRESHOLDS
                ),
            }
        trials.append(
            {
                "handle": trial["handle"],
                "interval": [first, last],
                "move_end_frame": trial["move_end_frame"],
                "settle": trial["settle"],
                "motion": trial["motion"],
                "scores": scores,
            }
        )
    return {
        "terminal": continuation["terminal"],
        "initial_settle": continuation["initial_settle"],
        "executed_actions": len(continuation["steps"]),
        "trials": trials,
        "pose_domain": continuation["pose_domain"],
        "metric_scale_known": False,
        "navigation_accepted": False,
    }


def _score_gauge_links(maintenance, positions, cameras):
    links = []
    for link in maintenance["gauge_bridges"]:
        parent, child = [
            maintenance["epochs"][link[key]]["frozen"]["snapshot"]
            for key in ("parent_epoch", "child_epoch")
        ]
        pmodels, cmodels = [
            {item["bootstrap_slot"]: item for item in snapshot["maps"]}
            for snapshot in (parent, child)
        ]
        first, last = parent["anchor_frame"], child["anchor_frame"]
        scores = []
        for pair in link["evidence"]["map_pairs"]:
            result = {
                "parent_bootstrap": pair["parent_bootstrap"],
                "child_bootstrap": pair["child_bootstrap"],
                "public_supported": pair["supported"],
                "rendered": None,
                "body": None,
            }
            if pair["similarity"] is not None:
                pm, cm = (
                    pmodels[pair["parent_bootstrap"]],
                    cmodels[pair["child_bootstrap"]],
                )
                for domain, observations, truth in (
                    ("rendered", cameras, rendered_motion_truth),
                    ("body", positions, relative_motion_truth),
                ):
                    result[domain] = score_gauge_truth(
                        pair["similarity"],
                        truth(observations[first], observations[last]),
                        parent_baseline=truth(
                            observations[first],
                            observations[first + pm["bootstrap_slot"]],
                        )["distance"],
                        child_baseline=truth(
                            observations[last],
                            observations[last + cm["bootstrap_slot"]],
                        )["distance"],
                        parent_gauge_baseline=float(
                            np.linalg.norm(
                                pm["poses"][pm["bootstrap_slot"]]["translation"]
                            )
                        ),
                        child_gauge_baseline=float(
                            np.linalg.norm(
                                cm["poses"][cm["bootstrap_slot"]]["translation"]
                            )
                        ),
                        policy=maintenance["gauge_policy"],
                    )
            scores.append(result)
        links.append(
            {
                "parent_epoch": link["parent_epoch"],
                "child_epoch": link["child_epoch"],
                "anchor_interval": [first, last],
                "public_supported": link["evidence"]["supported"],
                "reference_pair": link["evidence"]["reference_pair"],
                "matched_tracks": len(link["evidence"]["track_pairs"]),
                "candidate_scores": scores,
            }
        )
    return {
        "links": links,
        "supported_local_links": maintenance["supported_local_gauge_links"],
        "global_gauge_aligned": False,
        "metric_scale_known": False,
        "navigation_accepted": False,
    }


def audit(directory, engine):
    report = json.loads((directory / "report.json").read_bytes())
    config = json.loads((directory / "config.json").read_bytes())
    count = report["frame_count"]
    if type(count) is not int or not 50 <= count <= _frame_limit(config):
        raise ValueError("active capture frame count exceeds registration")
    # Freeze public inputs across the nested camera audit, not after it. Private
    # ledger names may be inspected here, but their bytes remain unopened.
    paths = [
        directory / name
        for name in (
            "config.json",
            "report.json",
            "events.jsonl",
            "active-acquisition.json",
            "camera-audit.jar",
        )
    ]
    paths += [
        directory / folder / f"frame-{slot:05d}.png"
        for folder in ("frames", "evaluator-native")
        for slot in range(count)
    ]
    paths += [
        path for path in (directory / "source-snapshot").rglob("*") if path.is_file()
    ]
    paths += list((runtime.ROOT / "src/minecraft_scientist").rglob("*.py"))
    paths += [engine, Path(__file__), Path(runtime.__file__)]
    frozen = {path: runtime.digest(path) for path in paths}
    # runtime.audit cannot read any private bytes until registration's complete
    # chronological public replay, image resizing and action receipt checks pass.
    physical = runtime.audit(directory, engine, registration=registration)
    if not physical["valid"]:
        return {
            "valid": False,
            "camera_audit": physical,
            "evaluation": None,
            "settled_evaluation": None,
            "recovery_evaluation": None,
            "maintenance_evaluation": None,
            "gauge_evaluation": None,
            "shared_anchor_evaluation": None,
            "navigation_accepted": False,
        }
    if not all(runtime.digest(path) == value for path, value in frozen.items()):
        return {
            "valid": False,
            "camera_audit": physical,
            "evaluation": None,
            "settled_evaluation": None,
            "recovery_evaluation": None,
            "maintenance_evaluation": None,
            "gauge_evaluation": None,
            "shared_anchor_evaluation": None,
            "failure": {
                "type": "ValueError",
                "message": "public bytes changed across camera audit",
            },
            "navigation_accepted": False,
        }
    extra = directory / "active-acquisition.json"
    config = json.loads((directory / "config.json").read_bytes())
    private = [
        directory / name
        for name in ("evaluator-positions.json", "evaluator-camera.jsonl")
    ]
    frozen.update({path: runtime.digest(path) for path in private})
    if any(frozen[path] != report["artifact_sha256"][path.name] for path in private):
        return {
            "valid": False,
            "camera_audit": physical,
            "evaluation": None,
            "settled_evaluation": None,
            "recovery_evaluation": None,
            "maintenance_evaluation": None,
            "gauge_evaluation": None,
            "shared_anchor_evaluation": None,
            "failure": {
                "type": "ValueError",
                "message": "private bytes changed after camera audit",
            },
            "navigation_accepted": False,
        }
    public_run = json.loads(extra.read_bytes())
    active = public_run["acquisition"]
    positions = json.loads((directory / "evaluator-positions.json").read_bytes())[
        "positions"
    ]
    cameras = physical["rendered_cameras"]
    evaluation = _score_acquisition(active, positions, cameras)
    settled_evaluation = None
    if _settled_registration(config):
        settled_evaluation = _score_continuation(
            public_run["settled"], positions, cameras
        )
    recovery_evaluation = None
    if _recovery_registration(config):
        recovered = public_run["recovery"]
        stability = None
        if recovered.get("new_anchor_frame") is not None:
            intervals = [entry["interval"] for entry in recovered["flow_prefixes"][-3:]]
            stability = {
                "public_intervals": intervals,
                "maximum_body_step_translation": max(
                    relative_motion_truth(positions[first], positions[last])["distance"]
                    for first, last in intervals
                ),
                "maximum_rendered_step_translation": max(
                    rendered_motion_truth(cameras[first], cameras[last])["distance"]
                    for first, last in intervals
                ),
                "diagnostic_only": True,
            }
        recovery_evaluation = {
            "terminal": recovered["terminal"],
            "parent_sha256": recovered["parent_sha256"],
            "wait_actions": len(recovered["wait_steps"]),
            "new_anchor_frame": recovered.get("new_anchor_frame"),
            "stability_diagnostic": stability,
            "acquisition": _score_acquisition(
                recovered["acquisition"], positions, cameras
            )
            if recovered["acquisition"]
            else None,
            "settled": _score_continuation(recovered["settled"], positions, cameras)
            if recovered["settled"]
            else None,
            "global_gauge_aligned": False,
            "navigation_accepted": False,
        }
    maintenance_evaluation = None
    if _maintenance_registration(config):
        maintenance = public_run["maintenance"]
        epochs = []
        for epoch in maintenance["epochs"]:
            fitted = epoch["frozen"]
            epochs.append(
                {
                    "number": epoch["number"],
                    "parent_sha256": epoch["parent_sha256"],
                    "terminal": epoch["terminal"],
                    "fit_attempts": len(epoch["attempts"]),
                    "fitted_candidates": sum(
                        len(item["snapshot"]["maps"]) for item in epoch["attempts"]
                    ),
                    "validation_handle": epoch["validation_handle"],
                    "evaluation": _score_acquisition(
                        {
                            "snapshot": fitted["snapshot"],
                            "anchor_frame": fitted["anchor_frame"],
                            "controller_decision": epoch["decision"],
                        },
                        positions,
                        cameras,
                    )
                    if fitted
                    else None,
                }
            )
        maintenance_evaluation = {
            "terminal": maintenance["terminal"],
            "executed_actions": len(maintenance["steps"]),
            "epochs": epochs,
            "global_gauge_aligned": False,
            "navigation_accepted": False,
        }
    gauge_evaluation = (
        _score_gauge_links(public_run["maintenance"], positions, cameras)
        if _gauge_registration(config)
        else None
    )
    shared_evaluation = None
    if _shared_registration(config):
        maintenance = public_run["maintenance"]
        shared_evaluation = {
            "versions": [
                {
                    "parent_epoch": version["parent_epoch"],
                    "child_epoch": version["child_epoch"],
                    "status": version["status"],
                    "terminal": version["terminal"],
                    "evaluation": _score_acquisition(
                        {
                            "snapshot": version["frozen"],
                            "anchor_frame": version["frozen"]["anchor_frame"],
                            "controller_decision": version["decision"],
                        },
                        positions,
                        cameras,
                    )
                    if version["frozen"] is not None
                    else None,
                }
                for version in maintenance["shared_anchor_versions"]
            ],
            "shadow_validated_versions": maintenance[
                "shadow_validated_shared_versions"
            ],
            "global_gauge_aligned": False,
            "metric_scale_known": False,
            "navigation_accepted": False,
        }
    checks = {
        "capture_and_public_policy_independently_replayed": physical["valid"],
        "artifacts_still_match_capture_ledger": all(
            frozen[directory / name] == value
            for name, value in report["artifact_sha256"].items()
        ),
        "bytes_stable_through_scoring": all(
            runtime.digest(path) == value for path, value in frozen.items()
        ),
    }
    # Invalid late mutation discards every physical score, not just a green flag.
    valid = all(checks.values())
    return {
        "valid": valid,
        "checks": checks,
        "camera_audit": physical,
        "policy_terminal": active["terminal"],
        "fit_count": active["fit_count"],
        "acquisition_attempts": len(active["acquisition_attempts"]),
        "executed_acquisition_actions": len(active["steps"]),
        "first_control_end_tick": active["first_control_end_tick"],
        "control_strategy": active["policy"]["control_strategy"],
        "evaluation": evaluation if valid else None,
        "settled_evaluation": settled_evaluation if valid else None,
        "recovery_evaluation": recovery_evaluation if valid else None,
        "maintenance_evaluation": maintenance_evaluation if valid else None,
        "gauge_evaluation": gauge_evaluation if valid else None,
        "shared_anchor_evaluation": shared_evaluation if valid else None,
        "navigation_accepted": False,
        "scope": config["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--engine-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.output.with_suffix(".source.py")
    if args.output.exists() or snapshot.exists():
        raise FileExistsError("refusing to overwrite active acquisition audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    try:
        result = audit(args.route, args.engine_jar)
    except (KeyError, ValueError, OSError, TypeError) as error:
        result = {
            "valid": False,
            "failure": {"type": type(error).__name__, "message": str(error)},
            "navigation_accepted": False,
        }
    result["audit_snapshot_sha256"] = runtime.digest(snapshot)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "valid",
                    "policy_terminal",
                    "failure",
                    "evaluation",
                    "settled_evaluation",
                    "recovery_evaluation",
                    "maintenance_evaluation",
                    "gauge_evaluation",
                    "shared_anchor_evaluation",
                )
            }
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

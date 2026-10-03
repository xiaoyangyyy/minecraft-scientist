"""Past-only joint-map development replay and separate public-first physical audit.

Historical RGB/receipt integrity is rechecked, not relabeled as current-source
controller acquisition. Every registered bootstrap is retained; no best private
pose or failed-future refit is used. This is not held-out navigation acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest, rgb

from minecraft_scientist.evaluation.relative_motion import score_relative_motion
from minecraft_scientist.exploration.joint_multiview_map import (
    JointMapConfig,
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
from minecraft_scientist.protocols import TransitionRecord

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "single historical development world; all past-only bootstrap candidates and future tests retained; local optimization not uniqueness; no navigation promotion or generalization"


def sources():
    package = ROOT / "src/minecraft_scientist"
    return {
        "package/" + path.relative_to(package).as_posix(): path
        for path in sorted(package.rglob("*.py"))
    } | {
        path.name: path
        for path in (
            Path(__file__),
            Path(_numeric_equal.__code__.co_filename),
            Path(rgb.__code__.co_filename),
        )
    }


def inputs(route):
    paths = [route / name for name in ("config.json", "report.json", "events.jsonl")]
    paths += list((route / "frames").glob("*.png"))
    paths += list((route / "goal-reference").glob("*.png"))
    paths += list((route / "source-snapshot").rglob("*.py"))
    return {path.relative_to(route).as_posix(): path for path in paths}


def public_data(route):
    """Read no evaluator file; preserve archival source rather than forging currentness."""
    manifest = json.loads((route / "config.json").read_bytes())
    raw = json.loads((route / "report.json").read_bytes())
    if (
        manifest["schema"] != "visibility-rgb-acquisition-v3"
        or raw["passed"] is not True
        or raw["failure"] is not None
        or raw["navigation_accepted"] is not False
        or raw["config_sha256"] != digest(route / "config.json")
        or set(raw["checks"])
        != {
            "execution_completed",
            "action_frame_alignment",
            "bounded_frames",
            "telemetry_alignment",
            "preregistration_unchanged",
            "input_sources_unchanged",
            "public_log_clean",
        }
        or any(value is not True for value in raw["checks"].values())
        or len(raw["trials"]) != 8
        or not 1 <= raw["frame_count"] <= manifest["maximum_frames"] <= 257
    ):
        raise ValueError("incomplete historical adaptive acquisition")
    snapshots = {
        path.relative_to(route / "source-snapshot").as_posix(): path
        for path in (route / "source-snapshot").rglob("*.py")
    }
    if set(snapshots) != set(manifest["source_sha256"]) or any(
        digest(path) != manifest["source_sha256"][name]
        for name, path in snapshots.items()
    ):
        raise ValueError("historical acquisition snapshots changed")
    frame_names = {f"frame-{slot:05d}.png" for slot in range(raw["frame_count"])}
    if (
        set(raw["frame_sha256"]) != frame_names
        or {path.name for path in (route / "frames").glob("*.png")} != frame_names
        or len(raw["rgb_sha256"]) != raw["frame_count"]
    ):
        raise ValueError("historical RGB ledger incomplete")
    images = []
    size = manifest["world"]["observation_size"]
    for slot in range(raw["frame_count"]):
        path = route / "frames" / f"frame-{slot:05d}.png"
        image = rgb(path)
        if (
            digest(path) != raw["frame_sha256"][path.name]
            or hashlib.sha256(image.tobytes()).hexdigest() != raw["rgb_sha256"][slot]
            or image.shape != (size, size, 3)
        ):
            raise ValueError("historical frame bytes, RGB or size changed")
        images.append(image)
    text = (route / "events.jsonl").read_text(encoding="utf-8")
    if any(
        token in text
        for token in ("location_stats", "xpos", "ypos", "zpos", '"yaw"', '"pitch"')
    ):
        raise ValueError("private telemetry in public data")
    events = [TransitionRecord.model_validate_json(line) for line in text.splitlines()]
    timeline, cursor = [], 0
    for event in events:
        end = cursor + event.event.duration
        if (
            not 0 < event.event.duration == event.executed_action.ticks
            or end >= len(images)
            or event.intervention_compliance is not True
            or event.event.success is not True
            or event.controller_failure is not None
            or event.state_before.agent_state["frame_sha256"]
            != raw["rgb_sha256"][cursor]
            or event.state_after.agent_state["frame_sha256"] != raw["rgb_sha256"][end]
        ):
            raise ValueError("noncompliant/misaligned public receipt")
        timeline.append(
            {
                "start_frame": cursor,
                "end_frame": end,
                "action": event.executed_action.model_dump(mode="json"),
            }
        )
        cursor = end
    if timeline != raw["timeline"] or cursor + 1 != len(images):
        raise ValueError("public receipts do not cover every frame")
    references = raw["reference_indices"]
    if (
        len(references) != 12
        or {path.name for path in (route / "goal-reference").glob("*.png")}
        != {f"view-{slot:02d}.png" for slot in range(12)}
        or any(
            digest(route / "goal-reference" / f"view-{number:02d}.png")
            != raw["frame_sha256"][f"frame-{slot:05d}.png"]
            for number, slot in enumerate(references)
        )
    ):
        raise ValueError("public reference views incomplete")
    calibration = calibrate_reference_camera(
        [images[slot] for slot in references], tuple(manifest["world"]["render_size"])
    )
    if not _numeric_equal(calibration, raw["landmark_calibration"]):
        raise ValueError("public calibration replay changed")
    # Trial fitting/shadow bounds must be actual one-tick public MOVE intervals.
    slots = {interval["start_frame"]: interval for interval in timeline}
    for number, trial in enumerate(raw["trials"]):
        first, end, shadow = (
            trial["anchor_frame"],
            trial["end_frame"],
            trial["second_leg_shadow"],
        )
        if (
            type(first) is not int
            or type(end) is not int
            or not 0 <= first < end < len(images)
            or shadow["freeze_frame"] != end
            or not end <= shadow["end_frame"] < len(images)
        ):
            raise ValueError("trial/shadow fitting boundary changed")
        moves = [slots.get(slot) for slot in range(first, shadow["end_frame"])]
        if any(
            interval is None
            or interval["end_frame"] != interval["start_frame"] + 1
            or interval["action"]["kind"] != "move"
            for interval in moves
        ):
            raise ValueError("trial/shadow is not aligned one-tick public movement")
        if (
            number
            and first <= raw["trials"][number - 1]["second_leg_shadow"]["end_frame"]
        ):
            raise ValueError("trial ordering/denominator changed")
    return raw, images, np.array(calibration["camera_intrinsics"])


def infer(route):
    raw, images, camera = public_data(route)
    config, temporal = JointMapConfig(), TemporalTrackConfig(maximum_frames=64)
    results = []
    for number, trial in enumerate(raw["trials"]):
        first, end, shadow = (
            trial["anchor_frame"],
            trial["end_frame"],
            trial["second_leg_shadow"],
        )
        record = {
            "trial": number,
            "anchor_frame": first,
            "fitting_end_frame": end,
            "original_terminal": trial["terminal"],
            "candidates": [],
            "future_end_frame": shadow["end_frame"],
            "source_aborted": trial["terminal"] != "finished",
        }
        results.append(record)
        if record["source_aborted"]:
            continue
        retained, previous, past, prefixes = None, None, None, []
        for absolute in range(first + 1, end + 1):
            current = track_frame_chain(images[first : absolute + 1], config=temporal)
            retained = (
                set(current.track_ids)
                if retained is None
                else retained & set(current.track_ids)
            )
            if previous is not None:
                before = {
                    identity: index for index, identity in enumerate(previous.track_ids)
                }
                after = {
                    identity: index for index, identity in enumerate(current.track_ids)
                }
                ids = sorted(retained)
                if ids and not np.allclose(
                    np.array(previous.pixels)[
                        :, [before[identity] for identity in ids]
                    ],
                    np.array(current.pixels)[
                        :-1, [after[identity] for identity in ids]
                    ],
                    rtol=0,
                    atol=1e-6,
                ):
                    raise ValueError("joint development prefix changed previous pixels")
            prefixes.append(current.model_dump(mode="json"))
            previous = current
        past = retain_tracks(previous, retained)
        record["past_tracks"], record["past_prefixes"] = (
            past.model_dump(mode="json"),
            prefixes,
        )
        # Every bootstrap is fitted/frozen before any later RGB is tracked.
        for bootstrap in range(2, past.frame_count):
            fitted = fit_joint_map(
                past,
                camera,
                bootstrap_slot=bootstrap,
                seed=2000 + 5 * number,
                config=config,
            )
            record["candidates"].append(
                {"map": fitted.model_dump(mode="json"), "future": None}
            )
        frozen = json.dumps(record["candidates"], sort_keys=True)
        if shadow["terminal"] != "completed" or shadow["end_frame"] - end != 4:
            continue
        future_prefixes = []
        for absolute in range(end + 1, shadow["end_frame"] + 1):
            current = track_frame_chain(images[first : absolute + 1], config=temporal)
            retained &= set(current.track_ids)
            future_prefixes.append(current.model_dump(mode="json"))
            previous = current
        extended = retain_tracks(previous, retained)
        record["future_tracks"], record["future_prefixes"] = (
            extended.model_dump(mode="json"),
            future_prefixes,
        )
        targets = (end + 2 - first, end + 4 - first)
        maps = [item["map"] for item in record["candidates"]]
        from minecraft_scientist.exploration.joint_multiview_map import JointMapEvidence

        futures = [
            validate_joint_future(
                JointMapEvidence.model_validate(item),
                past,
                extended,
                target_slots=targets,
            )
            for item in maps
        ]
        if json.dumps(record["candidates"], sort_keys=True) != frozen:
            raise ValueError("future tracking mutated frozen joint candidates")
        for item, future in zip(record["candidates"], futures, strict=True):
            item["future"] = future
    return results


def run(route, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite joint-map development")
    paths, runtime = inputs(route), sources()
    config = {
        "schema": "joint-map-development-v1",
        "scope": SCOPE,
        "source_sha256": {name: digest(path) for name, path in runtime.items()},
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "config": JointMapConfig().model_dump(mode="json"),
        "seeds": [2000 + 5 * number for number in range(8)],
        "bootstrap_rule": "every past offset 2..terminal; no winner selected",
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, path in runtime.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "completed": False,
        "failure": None,
        "navigation_accepted": False,
        "config_sha256": digest(output / "config.json"),
        "results": [],
    }
    try:
        result["results"] = infer(route)
        result["completed"] = True
    except Exception as error:  # noqa: BLE001 -- retain development failures
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["input_sources_unchanged"] = all(
        digest(path) == config["input_sha256"][name] for name, path in paths.items()
    ) and all(
        digest(path) == config["source_sha256"][name] for name, path in runtime.items()
    )
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def audit(route, experiment):
    config = json.loads((experiment / "config.json").read_bytes())
    predicted = json.loads((experiment / "report.json").read_bytes())
    paths, runtime = inputs(route), sources()
    checks = {
        "complete_prediction": predicted["completed"] is True
        and predicted["failure"] is None
        and predicted["input_sources_unchanged"] is True
        and predicted["navigation_accepted"] is False,
        "config_identity": predicted["config_sha256"]
        == digest(experiment / "config.json"),
        "complete_current_sources": config["source_sha256"]
        == {name: digest(path) for name, path in runtime.items()},
        "complete_public_inputs": config["input_sha256"]
        == {name: digest(path) for name, path in paths.items()},
        "snapshots_verified": {
            path.relative_to(experiment / "source-snapshot").as_posix(): digest(path)
            for path in (experiment / "source-snapshot").rglob("*.py")
        }
        == config["source_sha256"],
        "registered_policy": config["schema"] == "joint-map-development-v1"
        and config["scope"] == SCOPE
        and config["config"] == JointMapConfig().model_dump(mode="json")
        and config["seeds"] == [2000 + 5 * number for number in range(8)]
        and config["bootstrap_rule"]
        == "every past offset 2..terminal; no winner selected",
    }
    frozen = {
        path: digest(path)
        for path in list(paths.values())
        + list(runtime.values())
        + [experiment / "config.json", experiment / "report.json"]
        + list((experiment / "source-snapshot").rglob("*.py"))
    }
    if all(checks.values()):
        checks["all_public_joint_maps_and_future_pixels_replayed"] = _numeric_equal(
            infer(route), predicted["results"]
        )
        checks["all_eight_trials_retained"] = [
            item["trial"] for item in predicted["results"]
        ] == list(range(8))
        checks["public_bytes_stable_before_private_scoring"] = all(
            digest(path) == value for path, value in frozen.items()
        )
    if not all(checks.values()):
        return {"valid": False, "checks": checks, "evaluation": None}
    # First privileged read, only after the complete public replay above.
    evaluator = route / "evaluator-audit.json"
    evaluator_hash = digest(evaluator)
    positions = json.loads(evaluator.read_bytes())["positions"]
    raw = json.loads((route / "report.json").read_bytes())
    if len(positions) != raw["frame_count"] or any(
        not isinstance(point, dict)
        or any(
            type(point.get(key)) not in (int, float) or not np.isfinite(point[key])
            for key in ("xpos", "ypos", "zpos", "yaw", "pitch")
        )
        or not -90 <= point["pitch"] <= 90
        for point in positions
    ):
        raise ValueError("private evaluator alignment changed")
    evaluation = []

    def score(pose, first, second):
        translation = np.array(pose["translation"])
        return score_relative_motion(
            {
                "supported": True,
                "relative_rotation": pose["rotation"],
                "unit_translation": (
                    translation / max(np.linalg.norm(translation), 1e-12)
                ).tolist(),
            },
            positions[first],
            positions[second],
            minimum_translation=0.15,
            maximum_direction_error=20,
            maximum_rotation_error=5,
        )

    for record in predicted["results"]:
        scores = []
        for item in record["candidates"]:
            fitted = item["map"]
            local_score, future_scores = None, []
            if fitted["supported"]:
                pose = fitted["poses"][-1]
                local_score = score(
                    pose, record["anchor_frame"], record["fitting_end_frame"]
                )
                future = item["future"]
                if future is not None:
                    for check, later in zip(
                        future["frame_checks"], future["poses"], strict=True
                    ):
                        if later is None or not check["supported"]:
                            continue
                        absolute = record["anchor_frame"] + later["frame_slot"]
                        future_scores.append(
                            {
                                "absolute_frame": absolute,
                                "from_anchor": score(
                                    later, record["anchor_frame"], absolute
                                ),
                                "from_frozen_endpoint": score(
                                    relative_pose(pose, later),
                                    record["fitting_end_frame"],
                                    absolute,
                                ),
                            }
                        )
            scores.append(
                {
                    "bootstrap_slot": fitted["bootstrap_slot"],
                    "local_motion": local_score,
                    "future_motion": future_scores,
                    "future_status": None
                    if item["future"] is None
                    else item["future"]["status"],
                }
            )
        evaluation.append({"trial": record["trial"], "candidates": scores})
    checks["public_bytes_stable_through_scoring"] = all(
        digest(path) == value for path, value in frozen.items()
    )
    checks["private_bytes_stable_through_scoring"] = digest(evaluator) == evaluator_hash
    valid = all(checks.values())
    return {
        "valid": valid,
        "checks": checks,
        "evaluation": evaluation if valid else None,
        "evaluator_sha256": evaluator_hash,
        "navigation_accepted": False,
        "scope": SCOPE,
    }


def relative_pose(first, second):
    """World-to-camera composition; baseline and future share one unscaled gauge."""
    rotation = np.array(second["rotation"]) @ np.array(first["rotation"]).T
    translation = np.array(second["translation"]) - rotation @ np.array(
        first["translation"]
    )
    return {"rotation": rotation.tolist(), "translation": translation.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit is not None:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite joint-map audit")
        result = audit(args.route, args.audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({"valid": result["valid"], "evaluation": result["evaluation"]})
        )
        if not result["valid"]:
            raise SystemExit(1)
    else:
        result = run(args.route, args.output)
        print(
            json.dumps(
                {
                    "completed": result["completed"],
                    "failure": result["failure"],
                    "trials": len(result["results"]),
                }
            )
        )
        if not result["completed"] or not result["input_sources_unchanged"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

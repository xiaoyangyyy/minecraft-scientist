"""Censor fixed-trajectory RGB replay at its first causal new-control request.

This cannot estimate a counterfactual switched path. It records when the online
controller would intervene, then stops reading that trial's subsequent images.
Public acquisition verification may replay the whole existing session separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import active_motion_pipeline as active
import pose_competition_common as original
from audit_dense_correspondence import _numeric_equal
from controlled_motion_common import digest, rgb

from minecraft_scientist.exploration.temporal_tracks import track_frame_chain
from minecraft_scientist.exploration.visibility_acquisition import (
    VisibilityController,
    VisibilityPolicy,
)


def sources():
    return active.runtime_sources() | {
        "pose_competition_common.py": Path(original.__file__),
        "visibility_acquisition.py": Path(
            VisibilityController.observe.__code__.co_filename
        ),
        Path(__file__).name: Path(__file__),
        "audit_dense_correspondence.py": Path(_numeric_equal.__code__.co_filename),
    }


def configuration(route, experiment):
    return {
        "schema": "causal-visibility-requests-v1",
        "route": route.resolve().relative_to(original.planner.ROOT).as_posix(),
        "prior_predictions": experiment.resolve()
        .relative_to(original.planner.ROOT)
        .as_posix(),
        "policy": VisibilityPolicy().model_dump(mode="json"),
        "scope": "all sixteen movement legs; point-only prefix decisions; censor at first requested intervention or fixed leg end; not executed adaptive control, counterfactual imagery, map validation or navigation",
    }


def replay(route, policy, *, records=None):
    _, raw, checks = active.validate_acquisition(route)
    timeline = raw["timeline"]
    declared = sorted(
        {
            item["scheduled"]["motion_handle_index"]
            for item in timeline
            if item["scheduled"]["kind"] == "move"
        }
    )
    records = [] if records is None else records
    # Startup/reference rotations are not movement trials. Preserve every move.
    moves = [item for item in timeline if item["scheduled"]["kind"] == "move"]
    for number, interval in enumerate(moves):
        handle = interval["scheduled"]["motion_handle_index"]
        alternative = next(
            (
                item["scheduled"]["motion_handle_index"]
                for item in moves[number + 1 :]
                if item["scheduled"]["motion_handle_index"] != handle
            ),
            next(value for value in declared if value != handle),
        )
        controller = VisibilityController((handle, alternative), policy=policy)
        anchor, endpoint = interval["start_frame"], interval["end_frame"]
        frames = [rgb(route / "frames" / f"frame-{anchor:05d}.png")]
        record = {
            "trial": number,
            "anchor_frame": anchor,
            "fixed_end_frame": endpoint,
            "handles": [handle, alternative],
            "prefixes": [],
            "status": "censored_existing_leg_end",
            "last_frame_read": anchor,
        }
        records.append(record)  # A partial current trial survives exceptions.
        for tick in range(1, min(endpoint - anchor, policy.maximum_first_ticks) + 1):
            slot = anchor + tick
            frames.append(rgb(route / "frames" / f"frame-{slot:05d}.png"))
            record["last_frame_read"] = slot
            tracks = track_frame_chain(frames, config=policy.temporal)
            decision = controller.observe(tracks)
            record["prefixes"].append(
                {
                    "tracks": tracks.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                }
            )
            if decision.instruction in ("switch", "abort"):
                record["status"] = (
                    "request_different_control"
                    if decision.instruction == "switch"
                    else "abort_insufficient_support"
                )
                break
        # Never execute/read the counterfactual leg from this fixed trajectory.
    return records, checks


def run(route, experiment, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite visibility request evidence")
    paths, runtime = original.inputs(route, experiment), sources()
    manifest = configuration(route, experiment) | {
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "source_sha256": {name: digest(path) for name, path in runtime.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name, path in runtime.items():
        (snapshot / name).write_bytes(path.read_bytes())
    path = output / "config.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen = digest(path)
    result = {
        "completed": False,
        "failure": None,
        "config_sha256": frozen,
        "records": [],
        "navigation_accepted": False,
    }
    try:
        _, checks = replay(route, VisibilityPolicy(), records=result["records"])
        result.update(completed=True, acquisition_checks=checks)
    except Exception as error:  # noqa: BLE001 -- preserve every partial trial
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["inputs_sources_unchanged"] = (
        digest(path) == frozen
        and all(
            digest(path) == manifest["input_sha256"][name]
            for name, path in paths.items()
        )
        and all(
            digest(path) == manifest["source_sha256"][name]
            for name, path in runtime.items()
        )
    )
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def verify(route, experiment, output):
    manifest = json.loads((output / "config.json").read_bytes())
    predicted = json.loads((output / "report.json").read_bytes())
    paths, runtime = original.inputs(route, experiment), sources()
    checks = {
        "completed": predicted["completed"] is True
        and predicted["failure"] is None
        and predicted["inputs_sources_unchanged"] is True,
        "configuration": {
            key: value
            for key, value in manifest.items()
            if key not in ("input_sha256", "source_sha256")
        }
        == configuration(route, experiment),
        "manifest_identity": predicted["config_sha256"]
        == digest(output / "config.json"),
        "inputs_current": manifest["input_sha256"]
        == {name: digest(path) for name, path in paths.items()},
        "sources_current": manifest["source_sha256"]
        == {name: digest(path) for name, path in runtime.items()},
        "snapshot_identity": {
            path.name for path in (output / "source-snapshot").glob("*.py")
        }
        == set(runtime)
        and all(
            digest(output / "source-snapshot" / name) == value
            for name, value in manifest["source_sha256"].items()
        ),
        "no_navigation": predicted["navigation_accepted"] is False,
    }
    if all(checks.values()):
        bound = {
            path: digest(path)
            for path in list(paths.values())
            + list(runtime.values())
            + [output / "config.json", output / "report.json"]
            + list((output / "source-snapshot").glob("*.py"))
        }
        replayed, acquisition = replay(route, VisibilityPolicy())
        checks["all_causal_prefixes_replayed"] = _numeric_equal(
            replayed, predicted["records"]
        )
        checks["acquisition_checks_replayed"] = (
            acquisition == predicted["acquisition_checks"]
        )
        checks["all_bytes_stable"] = all(
            digest(path) == value for path, value in bound.items()
        )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "scope": "public-only request replay; no private scoring or adaptive trajectory success",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.route, args.experiment, args.output)
        audit = args.output / "public-replay.json"
        if audit.exists():
            raise FileExistsError("refusing to overwrite public replay report")
        audit.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result))
        if not result["valid"]:
            raise SystemExit(1)
    else:
        result = run(args.route, args.experiment, args.output)
        print(
            json.dumps(
                {
                    "completed": result["completed"],
                    "failure": result["failure"],
                    "statuses": dict(
                        Counter(record["status"] for record in result["records"])
                    ),
                }
            )
        )
        if not result["completed"] or not result["inputs_sources_unchanged"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

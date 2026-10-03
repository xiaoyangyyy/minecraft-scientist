"""Replay causal two-tick gates and refreshed local maps from public RGB only.

This simulates chronological estimator input on a registered capture. It is not
an online game-controller/navigation acceptance run. All refresh/reject windows,
initializations and simultaneous body/camera physical scores remain visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import probe_rendered_motion as common

from minecraft_scientist.exploration.causal_map_window import (
    CausalMapWindow,
    CausalWindowPolicy,
)
from minecraft_scientist.exploration.online_landmark_probe import (
    calibrate_reference_camera,
)
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    track_frame_chain,
)

SCOPE = "causal-prefix development replay; two original trajectories and two windows each; explicit refresh after every horizon including rejection; no navigation acceptance, metric gauge stitching or held-out generalization"


def sources():
    return common.sources() | {Path(__file__).name: Path(__file__)}


def policy():
    return {
        "schema": "causal-window-development-v1",
        "scope": SCOPE,
        "controller": CausalWindowPolicy().model_dump(mode="json"),
        "temporal": TemporalTrackConfig(maximum_frames=64).model_dump(mode="json"),
        "seeds": [2000, 2005],
        "windows_per_trial": 2,
        "fitting_ticks": 10,
        "horizon": 2,
        "bootstrap_rule": "every past bootstrap; future validation uses only its current prefix",
        "thresholds": common.THRESHOLDS,
    }


def fit_tracks(images, first, end):
    retained, previous = None, None
    prefixes = []
    for absolute in range(first + 1, end + 1):
        current = track_frame_chain(
            images[first : absolute + 1], config=TemporalTrackConfig(maximum_frames=64)
        )
        retained = (
            set(current.track_ids)
            if retained is None
            else retained & set(current.track_ids)
        )
        common.unchanged_pixels(previous, current, retained)
        previous = current
        prefixes.append(current.model_dump(mode="json"))
    return retain_tracks(previous, retained), prefixes


def infer(route, engine):
    public = common.runtime.public_data(route, engine)
    if not public["valid"]:
        raise ValueError("causal replay requires verified public capture")
    config = public["config"]
    images = [
        common.runtime.load_image(route / "frames" / f"frame-{slot:05d}.png")
        for slot in range(public["count"])
    ]
    calibration = calibrate_reference_camera(
        [images[slot] for slot in config["reference_indices"]],
        tuple(config["render_size"]),
    )
    camera, records = np.array(calibration["camera_intrinsics"]), []
    for number, (first, end, future_end) in enumerate(config["trial_intervals"]):
        if future_end - end != 4 or end - first != 10:
            raise ValueError(
                "causal development registration requires two two-tick windows"
            )
        past, fitting_prefixes = fit_tracks(images, first, end)
        controller = CausalMapWindow(
            past, camera, anchor_frame=first, seed=2000 + 5 * number
        )
        for window in range(2):
            anchor, fitting_end = first + window * 2, end + window * 2
            for absolute in range(fitting_end + 1, fitting_end + 3):
                current = track_frame_chain(
                    images[anchor : absolute + 1],
                    config=TemporalTrackConfig(maximum_frames=64),
                )
                decision = controller.advance(current)
            snapshot = controller.snapshot()
            record = {
                "trial": len(records),
                "source_trial": number,
                "window": window,
                "anchor_frame": anchor,
                "fitting_end_frame": fitting_end,
                "future_end_frame": decision.absolute_frame,
                "fitting_prefixes": fitting_prefixes,
                "snapshot": snapshot,
                "snapshot_sha256": controller.digest(),
                "controller_decision": decision.model_dump(mode="json"),
                "candidates": [
                    {"map": fitted, "future": checked["future"]}
                    for fitted, checked in zip(
                        snapshot["maps"],
                        decision.model_dump(mode="json")["candidates"],
                        strict=True,
                    )
                ],
            }
            records.append(record)
            if window == 0:
                past, fitting_prefixes = fit_tracks(
                    images, anchor + 2, decision.absolute_frame
                )
                controller = controller.refresh(past, anchor_frame=anchor + 2)
    if not all(
        common.runtime.digest(path) == value for path, value in public["frozen"].items()
    ):
        raise ValueError("public inputs changed during causal replay")
    return {"calibration": calibration, "trials": records, "navigation_accepted": False}


def registered(predicted):
    records = predicted["trials"]
    return (
        [item["trial"] for item in records] == [0, 1, 2, 3]
        and [(item["source_trial"], item["window"]) for item in records]
        == [(0, 0), (0, 1), (1, 0), (1, 1)]
        and all(
            [candidate["map"]["bootstrap_slot"] for candidate in item["candidates"]]
            == list(range(2, 11))
            and item["controller_decision"]["elapsed_ticks"] == 2
            and item["controller_decision"]["navigation_accepted"] is False
            for item in records
        )
    )


def run(route, engine, output):
    return common.run(
        route, engine, output, infer_fn=infer, policy_fn=policy, sources_fn=sources
    )


def audit(route, engine, experiment):
    return common.audit(
        route,
        engine,
        experiment,
        infer_fn=infer,
        policy_fn=policy,
        sources_fn=sources,
        denominator_fn=registered,
    )


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
            raise FileExistsError("refusing to overwrite causal-window audit")
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

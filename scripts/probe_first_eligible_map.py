"""Development replay of first-public-eligible map acquisition.

Every attempted prefix and bootstrap is retained; the first past prefix with at
least two supported maps is latched before using any validation pixels. This
replays existing controls, not an adaptive live controller or navigation success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import probe_rendered_motion as common

from minecraft_scientist.exploration.causal_map_window import CausalMapWindow
from minecraft_scientist.exploration.online_landmark_probe import (
    calibrate_reference_camera,
)
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    track_frame_chain,
)

SCOPE = "first-public-eligible-prefix development replay on two existing fixed trajectories; all attempts retained; no adaptive action dispatch, navigation or held-out generalization"
MINIMUM_TICKS = 3
MAXIMUM_TICKS = 10
HORIZON = 2


def sources():
    return common.sources() | {Path(__file__).name: Path(__file__)}


def policy():
    return {
        "schema": "first-eligible-map-development-v1",
        "scope": SCOPE,
        "minimum_ticks": MINIMUM_TICKS,
        "maximum_ticks": MAXIMUM_TICKS,
        "horizon": HORIZON,
        "minimum_supported_past_maps": 2,
        "seeds": [2000, 2005],
        "thresholds": common.THRESHOLDS,
        "selection": "latch first public past prefix with two supported maps; if exhausted retain last rejected prefix; never select by future pixels or physical labels",
    }


def acquire(images, camera, first, *, seed):
    previous, retained, attempts = None, None, []
    for tick in range(1, MAXIMUM_TICKS + 1):
        current = track_frame_chain(
            images[first : first + tick + 1],
            config=TemporalTrackConfig(maximum_frames=64),
        )
        retained = (
            set(current.track_ids)
            if retained is None
            else retained & set(current.track_ids)
        )
        common.unchanged_pixels(previous, current, retained)
        previous = retain_tracks(current, retained)
        if tick < MINIMUM_TICKS:
            continue
        controller = CausalMapWindow(previous, camera, anchor_frame=first, seed=seed)
        snapshot = controller.snapshot()
        eligible = sum(item["supported"] for item in snapshot["maps"]) >= 2
        attempts.append(
            {
                "absolute_frame": first + tick,
                "eligible": eligible,
                "snapshot": snapshot,
                "snapshot_sha256": controller.digest(),
            }
        )
        if eligible:
            break
    return controller, attempts, first + tick


def infer(route, engine):
    public = common.runtime.public_data(route, engine)
    if not public["valid"]:
        raise ValueError("first-eligible replay requires verified public capture")
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
        if end - first != MAXIMUM_TICKS or future_end - end < HORIZON:
            raise ValueError("capture does not cover the registered acquisition budget")
        controller, attempts, selected_end = acquire(
            images, camera, first, seed=2000 + 5 * number
        )
        frozen_attempts = json.dumps(attempts, sort_keys=True)
        for absolute in range(selected_end + 1, selected_end + HORIZON + 1):
            current = track_frame_chain(
                images[first : absolute + 1],
                config=TemporalTrackConfig(maximum_frames=64),
            )
            decision = controller.advance(current)
        if json.dumps(attempts, sort_keys=True) != frozen_attempts:
            raise ValueError("future observations changed past acquisition attempts")
        snapshot = controller.snapshot()
        records.append(
            {
                "trial": number,
                "anchor_frame": first,
                "fitting_end_frame": selected_end,
                "future_end_frame": decision.absolute_frame,
                "acquisition_attempts": attempts,
                "acquisition_eligible": attempts[-1]["eligible"],
                "acquisition_fit_count": sum(
                    len(attempt["snapshot"]["maps"]) for attempt in attempts
                ),
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
        )
    if not all(
        common.runtime.digest(path) == value for path, value in public["frozen"].items()
    ):
        raise ValueError("public inputs changed during acquisition replay")
    return {"calibration": calibration, "trials": records, "navigation_accepted": False}


def registered(predicted):
    if [item["trial"] for item in predicted["trials"]] != [0, 1]:
        return False
    for trial in predicted["trials"]:
        attempts, first = trial["acquisition_attempts"], trial["anchor_frame"]
        ticks = trial["fitting_end_frame"] - first
        if not (
            MINIMUM_TICKS <= ticks <= MAXIMUM_TICKS
            and [item["absolute_frame"] for item in attempts]
            == list(range(first + MINIMUM_TICKS, first + ticks + 1))
            and all(not item["eligible"] for item in attempts[:-1])
            and (attempts[-1]["eligible"] or ticks == MAXIMUM_TICKS)
            and trial["acquisition_eligible"] == attempts[-1]["eligible"]
            and trial["future_end_frame"] == trial["fitting_end_frame"] + HORIZON
            and [item["map"]["bootstrap_slot"] for item in trial["candidates"]]
            == list(range(2, ticks + 1))
            and trial["acquisition_fit_count"]
            == sum(len(item["snapshot"]["maps"]) for item in attempts)
        ):
            return False
        for attempt in attempts:
            past = attempt["snapshot"]
            if not (
                [item["bootstrap_slot"] for item in past["maps"]]
                == list(range(2, past["past"]["frame_count"]))
                and attempt["eligible"]
                == (sum(item["supported"] for item in past["maps"]) >= 2)
                and not past["prefixes"]
                and not past["decisions"]
            ):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--engine-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit is None:
        result = common.run(
            args.route,
            args.engine_jar,
            args.output,
            infer_fn=infer,
            policy_fn=policy,
            sources_fn=sources,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("completed", "failure", "input_sources_unchanged")
                }
            )
        )
        if not result["completed"] or not result["input_sources_unchanged"]:
            raise SystemExit(1)
    else:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite first-eligible audit")
        result = common.audit(
            args.route,
            args.engine_jar,
            args.audit,
            infer_fn=infer,
            policy_fn=policy,
            sources_fn=sources,
            denominator_fn=registered,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"valid": result["valid"], "checks": result["checks"]}))
        if not result["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

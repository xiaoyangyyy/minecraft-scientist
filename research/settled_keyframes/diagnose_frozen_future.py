"""Diagnose known frozen-map support loss using public pixels, not new controls.

Keep every original selected past view and every consumed future image. Compare
the original sparse future tracker with a complete one-frame future bridge using
exactly the original first-image seeds and tracking gates. This is retrospective
research evidence: never a replacement map, successful validation or navigation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from minecraft_scientist.exploration import (
    correspondence_refinement,
    dense_correspondence,
    joint_multiview_map,
    multiview_motion,
    safe_track_seeds,
    second_leg_shadow,
    temporal_tracks,
    visual_geometry,
    visual_odometry,
)
from minecraft_scientist.exploration.joint_multiview_map import (
    JointMapEvidence,
    validate_joint_future,
)
from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    TemporalTracks,
    diagnose_frame_chain,
    track_frame_chain,
)

SCHEMA = "walking-oriented-long-wait-safe-seed-stable-keyframe-camera-projection-v1"
DEPENDENCIES = (
    temporal_tracks,
    safe_track_seeds,
    correspondence_refinement,
    dense_correspondence,
    visual_geometry,
    second_leg_shadow,
    joint_multiview_map,
    multiview_motion,
    visual_odometry,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedules(acquisition):
    """No view selection: preserve original past and the whole executed future."""
    frozen, future = acquisition["frozen"], acquisition["future"]
    if frozen is None or future is None:
        raise ValueError("requires a previously frozen map and consumed future")
    past = frozen["index"]["frame_ordinals"]
    end, anchor = future["absolute_frame"], acquisition["anchor_frame"]
    steps = acquisition["steps"]
    if (
        type(anchor) is not int
        or not 48 <= anchor <= 81
        or type(end) is not int
        or not anchor < end <= 288
        or type(past) is not list
        or not 6 <= len(past) <= 12
        or any(type(slot) is not int for slot in past)
        or past != sorted(set(past))
        or not anchor <= past[0] < past[-1] < end
        or frozen["frozen_at_frame"] != past[-1]
        or future["index"]["frame_ordinals"] != past + [end]
        or type(steps) is not list
        or [step["absolute_frame"] for step in steps]
        != list(range(anchor + 1, end + 1))
        or any(
            type(step["absolute_frame"]) is not int
            or step["compliant"] is not True
            or type(step["ticks"]) is not int
            or step["ticks"] != 1
            or step["requested_kind"] not in ("move", "wait")
            or (
                step["requested_kind"] == "wait"
                and step["requested_handle"] is not None
            )
            or (
                step["requested_kind"] == "move"
                and (
                    type(step["requested_handle"]) is not int
                    or not 0 <= step["requested_handle"] < 4
                )
            )
            or step["requested_kind"] != step["executed_kind"]
            or step["requested_handle"] != step["executed_handle"]
            for step in steps
        )
    ):
        raise ValueError("requires original ordered views and complete actual clock")
    bridge = past + list(range(past[-1] + 1, end + 1))
    if len(bridge) > 128:
        raise ValueError("complete future bridge exceeds bounded diagnostic budget")
    return past, past + [end], bridge


def summary(diagnosis, ordinals):
    tracks = diagnosis.tracks
    return {
        "frame_ordinals": ordinals,
        "seed_count": tracks.seed_count,
        "surviving_counts": list(tracks.surviving_counts),
        "final_survivors": len(tracks.track_ids),
        "first_zero_frame": next(
            (
                slot
                for slot, count in zip(ordinals, tracks.surviving_counts)
                if count == 0
            ),
            None,
        ),
        "loss_by_reason": dict(
            sorted(
                Counter(
                    item.reason for item in diagnosis.rejections for _ in item.track_ids
                ).items()
            )
        ),
        "loss_events": [
            {
                "absolute_frame": ordinals[item.image_index],
                "reason": item.reason,
                "track_ids": list(item.track_ids),
            }
            for item in diagnosis.rejections
        ],
        "tracks": tracks.model_dump(mode="json"),
    }


def analyze(acquisition, frame):
    past, sparse, bridge = schedules(acquisition)
    detector = acquisition["frozen"]["detector"]
    first = frame(past[0])
    seeds = safe_seeds(first)
    if (
        detector["registration"] != "first-image-only-safe-support-mask-v1"
        or detector["seed_image_frame"] != past[0]
        or detector["seed_image_sha256"] != hashlib.sha256(first.tobytes()).hexdigest()
        or detector["seed_pixels"] != seeds.tolist()
    ):
        raise ValueError("cannot replace the original first-image seed denominator")
    seeds.setflags(write=False)
    tracker = TemporalTrackConfig(maximum_frames=128)
    initial = diagnose_frame_chain(
        [frame(slot) for slot in past], config=tracker, seed_pixels=seeds
    )
    if (
        initial.tracks.model_dump(mode="json")
        != acquisition["frozen"]["index"]["tracks"]
    ):
        raise ValueError("past tracking does not replay original frozen evidence")
    original = diagnose_frame_chain(
        [frame(slot) for slot in sparse], config=tracker, seed_pixels=seeds
    )
    retained = retain_tracks(original.tracks, set(initial.tracks.track_ids))
    if retained.model_dump(mode="json") != acquisition["future"]["index"]["tracks"]:
        raise ValueError("sparse future does not replay original failed evidence")
    continued = diagnose_frame_chain(
        [frame(slot) for slot in bridge], config=tracker, seed_pixels=seeds
    )
    if any(
        item.tracks.seed_count != len(seeds) for item in (initial, original, continued)
    ):
        raise ValueError("diagnostic cannot replenish identities")
    return {
        "original_past_replayed": True,
        "original_future_replayed": True,
        "sparse": summary(original, sparse),
        "complete_future_bridge": summary(continued, bridge),
        "original_terminal": acquisition["terminal"],
        "original_result_replaced": False,
        "map_fitted": False,
        "future_validation_accepted": False,
        "navigation_accepted": False,
    }


def prefix_geometry(acquisition, frame):
    """Test every prefix against every frozen model without selecting a winner."""
    past_ordinals, _, bridge = schedules(acquisition)
    frozen = acquisition["frozen"]
    past = TemporalTracks.model_validate(frozen["index"]["tracks"])
    models = [JointMapEvidence.model_validate(item) for item in frozen["maps"]]
    required = frozen["required_bootstraps"]
    if (
        not required
        or required != sorted(set(required))
        or required != [model.bootstrap_slot for model in models if model.supported]
        or len({model.bootstrap_slot for model in models}) != len(models)
    ):
        raise ValueError("cannot select alternative frozen bootstrap winners")
    seeds = np.array(frozen["detector"]["seed_pixels"], dtype=np.float32)
    seeds.setflags(write=False)
    past_frames = [frame(slot) for slot in past_ordinals]
    controls = {step["absolute_frame"]: step for step in acquisition["steps"]}
    rows = []
    for ordinal in bridge[len(past_ordinals) :]:
        tracked = track_frame_chain(
            past_frames + [frame(ordinal)],
            config=TemporalTrackConfig(maximum_frames=128),
            seed_pixels=seeds,
        )
        extended = retain_tracks(tracked, set(past.track_ids))
        evidence = [
            {
                "bootstrap_slot": model.bootstrap_slot,
                "required": model.bootstrap_slot in required,
                "future": validate_joint_future(
                    model, past, extended, target_slots=(len(past_ordinals),)
                ),
            }
            for model in models
        ]
        rows.append(
            {
                "absolute_frame": ordinal,
                "requested_kind": controls[ordinal]["requested_kind"],
                "requested_handle": controls[ordinal]["requested_handle"],
                "original_seed_count": extended.seed_count,
                "retained_past_id_count": len(extended.track_ids),
                "candidates": evidence,
                "all_required_pixel_tests_consistent": all(
                    item["future"]["status"] == "consistent"
                    for item in evidence
                    if item["required"]
                ),
                "prefix_selected_for_control": False,
                "future_validation_accepted": False,
                "navigation_accepted": False,
            }
        )
    return rows


def probe(route, output, *, all_future_prefixes=False):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if type(all_future_prefixes) is not bool:
        raise ValueError("prefix diagnosis requires an explicit boolean registration")
    if output.exists() or output.is_relative_to(route):
        raise FileExistsError("requires new output outside original capture")
    before = {}

    def bound(name):
        path = route / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(route):
            raise ValueError("public input cannot escape original capture")
        before[path] = digest(path)
        return path

    config_path, report_path = bound("config.json"), bound("report.json")
    config, capture = [
        json.loads(path.read_bytes()) for path in (config_path, report_path)
    ]
    public_path, audit_path = bound("keyframe-acquisition.json"), bound("audit.json")
    audit_source = bound("audit.source.py")
    public, audit = [
        json.loads(path.read_bytes()) for path in (public_path, audit_path)
    ]
    if (
        config["schema"] != SCHEMA
        or capture["completed"] is not True
        or capture["failure"] is not None
        or capture["config_sha256"] != before[config_path]
        or capture["artifact_sha256"][public_path.name] != before[public_path]
        or not capture["checks"]
        or not all(value is True for value in capture["checks"].values())
        or audit["valid"] is not True
        or not audit["checks"]
        or not all(value is True for value in audit["checks"].values())
        or audit["audit_snapshot_sha256"] != before[audit_source]
    ):
        raise ValueError("requires an intact independently audited walking capture")
    # Audit is read only for attestation. Never read evaluator poses or use scores.
    acquisition = public["acquisition"]
    schedules(acquisition)
    sources = {"diagnostic.source.py": Path(__file__)}
    for module in DEPENDENCIES:
        source = Path(module.__file__)
        name = "package/exploration/" + source.name
        snapshot = bound("source-snapshot/" + name)
        if config["source_sha256"][name] != digest(source) or before[
            snapshot
        ] != digest(source):
            raise ValueError("tracking dependency differs from original capture")
        sources[source.name] = source
    source_hashes = {name: digest(path) for name, path in sources.items()}
    frames = {}

    def frame(slot):
        if slot not in frames:
            name = f"frames/frame-{slot:05d}.png"
            path = bound(name)
            if capture["artifact_sha256"][name] != before[path]:
                raise ValueError("public image differs from capture ledger")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError("requires original RGB")
                frames[slot] = np.array(image)
            if frames[slot].shape != (448, 448, 3) or frames[slot].dtype != np.uint8:
                raise ValueError("cannot change original public sensor")
        return frames[slot]

    output.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        (output / name).write_bytes(source.read_bytes())
    result = analyze(acquisition, frame)
    result["all_future_prefixes"] = (
        prefix_geometry(acquisition, frame) if all_future_prefixes else None
    )
    checks = {
        "public_inputs_unchanged": all(
            digest(path) == value for path, value in before.items()
        ),
        "sources_unchanged": all(
            digest(sources[name]) == value for name, value in source_hashes.items()
        ),
        "original_past_replayed": result["original_past_replayed"],
        "original_future_replayed": result["original_future_replayed"],
    }
    result.update(
        schema="known-frozen-future-public-diagnostic-v1",
        scope="Known public observations only. Complete consumed future, original seeds/gates; no private numerical input, new actions, view reselection, refit or replacement of original rejection. Diagnostic is not independent generalization or navigation proof.",
        input_sha256={
            path.relative_to(route).as_posix(): value for path, value in before.items()
        },
        source_sha256=source_hashes,
        checks=checks,
        passed=all(checks.values()),
    )
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise ValueError("diagnostic input or source changed during analysis")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-future-prefixes", action="store_true")
    args = parser.parse_args()
    result = probe(
        args.route, args.output, all_future_prefixes=args.all_future_prefixes
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "sparse_survivors": result["sparse"]["final_survivors"],
                "bridge_survivors": result["complete_future_bridge"]["final_survivors"],
                "bridge_first_zero_frame": result["complete_future_bridge"][
                    "first_zero_frame"
                ],
                "navigation_accepted": False,
            }
        )
    )


if __name__ == "__main__":
    main()

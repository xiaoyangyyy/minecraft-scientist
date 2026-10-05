"""Replay every actual safe-seed endpoint pair against its complete raw bridge.

Known public data only. This does not select new views, fit a map, change gates,
read evaluator poses, or establish out-of-sample prediction/navigation success.
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
    safe_track_seeds,
    temporal_tracks,
    visual_geometry,
)
from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    diagnose_frame_chain,
)
from minecraft_scientist.exploration.visual_geometry import _cells

REGISTRATION = "first-image-only-safe-support-mask-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(diagnosis):
    tracked = diagnosis.tracks
    pixels = np.asarray(tracked.pixels)
    flows = (
        np.linalg.norm(pixels[-1] - pixels[0], axis=1)
        if tracked.track_ids
        else np.empty(0)
    )
    cells = _cells(pixels[-1], *tracked.image_shape) if len(flows) else 0
    survival = len(flows) / max(tracked.seed_count, 1)
    median = float(np.median(flows)) if len(flows) else None
    support = len(flows) >= 32 and cells >= 4 and survival >= 0.5
    return {
        "diagnosis": diagnosis.model_dump(mode="json"),
        "seed_count": tracked.seed_count,
        "surviving_count": len(flows),
        "cells": cells,
        "survival": survival,
        "median_flow_pixels": median,
        "percentile95_flow_pixels": float(np.percentile(flows, 95))
        if len(flows)
        else None,
        "correspondence_supported": bool(support),
        "novelty_supported": bool(support and median is not None and median >= 2.0),
        "loss_by_reason": dict(
            sorted(
                Counter(
                    event.reason
                    for event in diagnosis.rejections
                    for _ in event.track_ids
                ).items()
            )
        ),
    }


def compare_seeded(frames, seeds):
    if not 2 <= len(frames) <= 128:
        raise ValueError("requires the complete bounded raw interval, not a subset")
    frozen = np.asarray(seeds, dtype=np.float32).copy()
    frozen.setflags(write=False)
    output = {}
    for name, sequence in (("direct", [frames[0], frames[-1]]), ("continuous", frames)):
        diagnosis = diagnose_frame_chain(
            sequence,
            config=TemporalTrackConfig(maximum_frames=128),
            seed_pixels=frozen,
        )
        if diagnosis.tracks.seed_count != len(frozen):
            raise ValueError("tracking changed the frozen first-image seed denominator")
        output[name] = summarize(diagnosis)
    first = output["direct"]["diagnosis"]["tracks"]
    last = output["continuous"]["diagnosis"]["tracks"]
    shared = sorted(set(first["track_ids"]) & set(last["track_ids"]))
    indices = [
        {identity: slot for slot, identity in enumerate(item["track_ids"])}
        for item in (first, last)
    ]
    distances = (
        np.linalg.norm(
            np.asarray(first["pixels"][-1])[
                [indices[0][identity] for identity in shared]
            ]
            - np.asarray(last["pixels"][-1])[
                [indices[1][identity] for identity in shared]
            ],
            axis=1,
        )
        if shared
        else np.empty(0)
    )
    output.update(
        shared_anchor_id_count=len(shared),
        shared_endpoint_disagreement_95_pixels=float(np.percentile(distances, 95))
        if len(distances)
        else None,
        is_ground_truth_error=False,
    )
    return output


def probe(route, output):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a seeded-bridge diagnosis")
    paths = [
        route / name
        for name in (
            "config.json",
            "report.json",
            "keyframe-acquisition.json",
            "audit.json",
            "audit.source.py",
        )
    ]
    before = {path: digest(path) for path in paths}
    config, capture, public, audit = [
        json.loads(path.read_bytes()) for path in paths[:4]
    ]
    if (
        config["schema"] != "safe-seed-stable-keyframe-camera-projection-v1"
        or capture["completed"] is not True
        or capture["failure"] is not None
        or not capture["checks"]
        or not all(value is True for value in capture["checks"].values())
        or capture["config_sha256"] != before[paths[0]]
        or capture["artifact_sha256"][paths[2].name] != before[paths[2]]
        or audit["valid"] is not True
        or not audit["checks"]
        or not all(value is True for value in audit["checks"].values())
        or audit["audit_snapshot_sha256"] != before[paths[4]]
        or public["acquisition"]["seed_detection_registration"] != REGISTRATION
    ):
        raise ValueError("requires an intact independently audited safe-seed capture")

    sources = {
        "probe.source.py": Path(__file__),
        "safe-seed.source.py": Path(safe_track_seeds.__file__),
        "temporal-tracks.source.py": Path(temporal_tracks.__file__),
        "correspondence-refinement.source.py": Path(correspondence_refinement.__file__),
        "visual-geometry.source.py": Path(visual_geometry.__file__),
    }
    source_hashes = {name: digest(path) for name, path in sources.items()}
    inputs = {
        "schema": "known-safe-seed-raw-bridge-diagnostic-v1",
        "capture_inputs_sha256": {path.name: value for path, value in before.items()},
        "sources_sha256": source_hashes,
        "tracking": TemporalTrackConfig(maximum_frames=128).model_dump(mode="json"),
        "detector_registration": REGISTRATION,
        "scope": "All actual original selected-anchor pairs, complete raw intervals, frozen first-image seeds and unchanged gates. No private inputs, new controls, view reselection, fitted map, future validation or navigation claim.",
        "navigation_accepted": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True) + "\n"
    )
    for name, path in sources.items():
        (output / name).write_bytes(path.read_bytes())

    frames = {}

    def frame(slot):
        if slot not in frames:
            name = f"frames/frame-{slot:05d}.png"
            path = route / name
            before[path] = digest(path)
            if before[path] != capture["artifact_sha256"][name]:
                raise ValueError("public RGB differs from capture ledger")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError("requires unchanged public RGB")
                frames[slot] = np.array(image)
            if frames[slot].shape != (448, 448, 3) or frames[slot].dtype != np.uint8:
                raise ValueError("cannot change the public sensor shape")
        return frames[slot]

    records = []
    acquisition = public["acquisition"]
    consumed = []
    for epoch in acquisition["epochs"]:
        selected = epoch["anchor_frame"]
        samples = [
            sample
            for sample in acquisition["samples"]
            if sample.get("epoch_id") == epoch["epoch_id"] and "trial" in sample
        ]
        for sample in samples:
            end = sample["absolute_frame"]
            if type(end) is not int or type(selected) is not int or end <= selected:
                raise ValueError("original sample interval is not chronological")
            first = frame(selected)
            seeds = safe_seeds(first)
            novelty = sample["novelty"]
            if (
                novelty["seed_detection_registration"] != REGISTRATION
                or novelty["seed_image_sha256"]
                != hashlib.sha256(first.tobytes()).hexdigest()
                or novelty["seed_pixels"] != seeds.tolist()
            ):
                raise ValueError(
                    "first-image seed registration does not replay original sample"
                )
            direct = summarize(
                diagnose_frame_chain(
                    [first, frame(end)],
                    config=TemporalTrackConfig(maximum_frames=3),
                    seed_pixels=seeds,
                )
            )
            if (
                direct["diagnosis"]["tracks"] != novelty["tracks"]
                or direct["correspondence_supported"]
                is not sample["correspondence_supported"]
                or direct["novelty_supported"] is not sample["selected"]
                or any(
                    direct[key] != novelty[key]
                    for key in (
                        "cells",
                        "median_flow_pixels",
                        "percentile95_flow_pixels",
                    )
                )
            ):
                raise ValueError("direct diagnosis does not replay original sample")
            record = {
                "epoch_id": epoch["epoch_id"],
                "trial": sample["trial"],
                "namespace": f"safe-seeds-epoch-{epoch['epoch_id']}-anchor-{selected}",
                "interval": [selected, end],
                "seed_pixels": seeds.tolist(),
                "seed_image_sha256": novelty["seed_image_sha256"],
                "original_selected": sample["selected"],
                "original_support": sample["correspondence_supported"],
                "direct_tracks_replayed": True,
                "direct": direct,
                "original_sample_replaced": False,
            }
            if end - selected + 1 > 128:
                record["status"] = "raw_interval_exceeds_registered_tracking_budget"
                record["comparison"] = None
            else:
                comparison = compare_seeded(
                    [frame(slot) for slot in range(selected, end + 1)], seeds
                )
                if comparison["direct"] != direct:
                    raise ValueError("comparison changed original direct tracking")
                record.update(status="compared", comparison=comparison)
            records.append(record)
            consumed.append(sample["trial"])
            if sample["selected"]:
                selected = end
        if epoch["frame_ordinals"] != [epoch["anchor_frame"]] + [
            item["absolute_frame"] for item in samples if item["selected"]
        ]:
            raise ValueError(
                "original epoch selected frames do not match sample history"
            )
    expected = [item["trial"] for item in acquisition["samples"] if "trial" in item]
    if consumed != expected or len(consumed) != len(set(consumed)):
        raise ValueError("diagnosis omitted or reordered original trials")
    if any(digest(path) != value for path, value in before.items()):
        raise ValueError("diagnostic input bytes changed during analysis")
    if any(digest(sources[name]) != value for name, value in source_hashes.items()):
        raise ValueError("diagnostic source bytes changed during analysis")
    result = {
        "config": inputs,
        "records": records,
        "public_inputs_sha256": {
            path.relative_to(route).as_posix(): value for path, value in before.items()
        },
        "checks": {
            "all_public_inputs_unchanged": True,
            "all_sources_unchanged": True,
            "all_original_direct_tracks_replayed": True,
            "every_original_trial_retained": True,
        },
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "checks": result["checks"],
                "records": [
                    {
                        "epoch": record["epoch_id"],
                        "trial": record["trial"],
                        "interval": record["interval"],
                        "status": record["status"],
                        "direct_survival": record["direct"]["survival"],
                        "continuous_survival": record["comparison"]["continuous"][
                            "survival"
                        ]
                        if record["comparison"]
                        else None,
                        "continuous_novelty": record["comparison"]["continuous"][
                            "novelty_supported"
                        ]
                        if record["comparison"]
                        else None,
                    }
                    for record in records
                ],
            },
            sort_keys=True,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.route, args.output)

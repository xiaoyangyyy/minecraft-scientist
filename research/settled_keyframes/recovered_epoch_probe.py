"""Diagnose all public endpoint losses within recovered local epochs.

Known-data diagnostic only: no changed tracker gates, detector, fitted map, extra
controls, private poses or promotion. Original sample identities are replayed.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tracking_gap_probe import compare

from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    diagnose_frame_chain,
)
from minecraft_scientist.exploration.visual_geometry import _cells


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(route, output, *, safe_seed_controls=False):
    if type(safe_seed_controls) is not bool:
        raise ValueError("research seed protocol requires an explicit boolean")
    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite recovered-epoch diagnosis")
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
        config["schema"] != "recovering-stable-keyframe-camera-projection-v1"
        or capture["completed"] is not True
        or capture["failure"] is not None
        or not all(value is True for value in capture["checks"].values())
        or capture["config_sha256"] != before[paths[0]]
        or capture["artifact_sha256"][paths[2].name] != before[paths[2]]
        or audit["valid"] is not True
        or not all(value is True for value in audit["checks"].values())
        or audit["audit_snapshot_sha256"] != before[paths[4]]
    ):
        raise ValueError("diagnosis requires an intact independently audited capture")
    frames = {}

    def frame(slot):
        if slot not in frames:
            name = f"frames/frame-{slot:05d}.png"
            path = route / name
            before[path] = digest(path)
            if before[path] != capture["artifact_sha256"][name]:
                raise ValueError("public RGB differs from its capture ledger")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError("diagnosis requires unchanged public RGB")
                frames[slot] = np.array(image)
            if frames[slot].shape != (448, 448, 3):
                raise ValueError("diagnosis cannot change the sensor shape")
        return frames[slot]

    inputs = {
        "schema": "known-recovered-epoch-loss-diagnostic-v1",
        "source_sha256": digest(Path(__file__)),
        "comparison_source_sha256": digest(
            Path(__file__).with_name("tracking_gap_probe.py")
        ),
        "capture_inputs_sha256": {path.name: value for path, value in before.items()},
        "scope": "Public-only known development images; unchanged direct tracking with complete rejection ledger and bounded continuous comparison; not prediction success or live recovery",
        "navigation_accepted": False,
    }
    if safe_seed_controls:
        from safe_seed_detector import safe_seeds

        inputs.update(
            detector_registration="first-image-only-safe-support-mask-v1",
            detector_source_sha256=digest(
                Path(__file__).with_name("safe_seed_detector.py")
            ),
            detector_promotion=False,
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True) + "\n"
    )
    (output / "probe.source.py").write_bytes(Path(__file__).read_bytes())
    (output / "comparison.source.py").write_bytes(
        Path(__file__).with_name("tracking_gap_probe.py").read_bytes()
    )
    if safe_seed_controls:
        (output / "safe-seed.source.py").write_bytes(
            Path(__file__).with_name("safe_seed_detector.py").read_bytes()
        )
    records = []
    acquisition = public["acquisition"]
    for epoch in acquisition["epochs"][1:]:
        selected = previous = epoch["anchor_frame"]
        samples = [
            item
            for item in acquisition["samples"]
            if item.get("epoch_id") == epoch["epoch_id"] and "trial" in item
        ]
        for sample in samples:
            end = sample["absolute_frame"]
            diagnosis = diagnose_frame_chain(
                [frame(selected), frame(end)],
                config=TemporalTrackConfig(maximum_frames=3),
            )
            direct = diagnosis.tracks.model_dump(mode="json")
            if direct != sample["novelty"]["tracks"]:
                raise ValueError(
                    "diagnostic direct tracks do not replay original sample"
                )
            comparisons = []
            for namespace, start in (
                ("last_selected", selected),
                ("previous_stable", previous),
            ):
                if end - start + 1 > 128:
                    comparisons.append(
                        {
                            "namespace": namespace,
                            "interval": [start, end],
                            "status": "raw_interval_exceeds_registered_tracking_budget",
                        }
                    )
                else:
                    comparisons.append(
                        {
                            "namespace": namespace,
                            "interval": [start, end],
                            "status": "compared",
                            "comparison": compare(
                                [frame(slot) for slot in range(start, end + 1)]
                            ),
                        }
                    )
            records.append(
                {
                    "epoch_id": epoch["epoch_id"],
                    "trial": sample["trial"],
                    "interval": [selected, end],
                    "original_selected": sample["selected"],
                    "original_support": sample["correspondence_supported"],
                    "direct_tracks_replayed": True,
                    "diagnosis": diagnosis.model_dump(mode="json"),
                    "direct_loss_by_reason": dict(
                        sorted(
                            Counter(
                                item.reason
                                for item in diagnosis.rejections
                                for _ in item.track_ids
                            ).items()
                        )
                    ),
                    "comparisons": comparisons,
                }
            )
            if safe_seed_controls:
                # A distinct first-frame detector namespace, never relabelled old IDs.
                seeds = safe_seeds(frame(selected))
                masked = diagnose_frame_chain(
                    [frame(selected), frame(end)],
                    config=TemporalTrackConfig(maximum_frames=3),
                    seed_pixels=seeds,
                )
                tracked = masked.tracks
                pixels = np.asarray(tracked.pixels)
                flows = (
                    np.linalg.norm(pixels[-1] - pixels[0], axis=1)
                    if tracked.track_ids
                    else np.empty(0)
                )
                cells = _cells(pixels[-1], *tracked.image_shape) if len(flows) else 0
                survival = len(tracked.track_ids) / max(tracked.seed_count, 1)
                median = float(np.median(flows)) if len(flows) else None
                support = len(flows) >= 32 and cells >= 4 and survival >= 0.5
                records[-1]["safe_seed_probe"] = {
                    "namespace": f"safe-seeds-epoch-{epoch['epoch_id']}-anchor-{selected}",
                    "seed_image_frame": selected,
                    "seed_pixels": seeds.tolist(),
                    "diagnosis": masked.model_dump(mode="json"),
                    "cells": cells,
                    "survival": survival,
                    "median_flow_pixels": median,
                    "correspondence_supported": bool(support),
                    "novelty_supported": bool(
                        support and median is not None and median >= 2.0
                    ),
                    "original_sample_replaced": False,
                    "map_fitted": False,
                    "navigation_accepted": False,
                }
            if sample["selected"]:
                selected = end
            previous = end
    if any(digest(path) != value for path, value in before.items()):
        raise ValueError("diagnostic input bytes changed during analysis")
    result = {
        "config": inputs,
        "records": records,
        "checks": {
            "all_public_inputs_unchanged": True,
            "all_original_direct_tracks_replayed": True,
        },
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    summary = {
        "checks": result["checks"],
        "records": [
            {
                "interval": item["interval"],
                "direct_loss_by_reason": item["direct_loss_by_reason"],
                "survivors": len(item["diagnosis"]["tracks"]["track_ids"]),
                "seeds": item["diagnosis"]["tracks"]["seed_count"],
                "comparisons": [
                    {
                        "namespace": comparison["namespace"],
                        "status": comparison["status"],
                        "continuous_survival": comparison.get("comparison", {})
                        .get("continuous", {})
                        .get("survival"),
                    }
                    for comparison in item["comparisons"]
                ],
            }
            for item in records
        ],
    }
    if safe_seed_controls:
        summary["safe_seed_probes"] = [
            {
                "interval": item["interval"],
                "seeds": item["safe_seed_probe"]["diagnosis"]["tracks"]["seed_count"],
                "survivors": len(
                    item["safe_seed_probe"]["diagnosis"]["tracks"]["track_ids"]
                ),
                "support": item["safe_seed_probe"]["correspondence_supported"],
                "novelty": item["safe_seed_probe"]["novelty_supported"],
                "survival": item["safe_seed_probe"]["survival"],
                "median": item["safe_seed_probe"]["median_flow_pixels"],
            }
            for item in records
        ]
    print(json.dumps(summary, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-seed-probe", action="store_true")
    args = parser.parse_args()
    probe(args.route, args.output, safe_seed_controls=args.safe_seed_probe)

"""Known-public-data direct/continuous correspondence diagnostic, not a map.

All gates/detector settings stay fixed. Pair namespaces remain independent; this
does not merge different landmark IDs or claim successful live future prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    track_frame_chain,
)
from minecraft_scientist.exploration.visual_geometry import _cells


def compare(frames):
    config = TemporalTrackConfig(maximum_frames=128)
    if not 2 <= len(frames) <= config.maximum_frames:
        raise ValueError(
            "tracking diagnostic requires the complete bounded raw interval"
        )
    output, tracks = {}, {}
    for name, sequence in (("direct", [frames[0], frames[-1]]), ("continuous", frames)):
        tracked = track_frame_chain(sequence, config=config)
        tracks[name] = tracked
        pixels = np.asarray(tracked.pixels)
        flow = (
            np.linalg.norm(pixels[-1] - pixels[0], axis=1)
            if tracked.track_ids
            else np.empty(0)
        )
        output[name] = {
            "observations": tracked.frame_count,
            "seed_count": tracked.seed_count,
            "surviving_count": len(tracked.track_ids),
            "survival": len(tracked.track_ids) / max(tracked.seed_count, 1),
            "cells": _cells(pixels[-1], *tracked.image_shape) if len(flow) else 0,
            "median_flow_pixels": float(np.median(flow)) if len(flow) else None,
            "percentile95_flow_pixels": float(np.percentile(flow, 95))
            if len(flow)
            else None,
            "tracks": tracked.model_dump(mode="json"),
        }
    first, last = tracks["direct"], tracks["continuous"]
    if first.seed_count != last.seed_count:
        raise ValueError("comparison changed the identical anchor detector")
    shared = sorted(set(first.track_ids) & set(last.track_ids))
    indices = [
        {identity: slot for slot, identity in enumerate(item.track_ids)}
        for item in (first, last)
    ]
    distances = (
        np.linalg.norm(
            np.array(first.pixels[-1])[[indices[0][identity] for identity in shared]]
            - np.array(last.pixels[-1])[[indices[1][identity] for identity in shared]],
            axis=1,
        )
        if shared
        else np.empty(0)
    )
    output["shared_anchor_id_count"] = len(shared)
    output["shared_endpoint_disagreement_95_pixels"] = (
        float(np.percentile(distances, 95)) if len(distances) else None
    )
    output["is_ground_truth_error"] = False
    return output


def probe(route, output):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a tracking-gap diagnostic")
    report_path, config_path, public_path = [
        route / name
        for name in ("report.json", "config.json", "keyframe-acquisition.json")
    ]
    report, capture, public = [
        json.loads(path.read_bytes())
        for path in (report_path, config_path, public_path)
    ]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {path: digest(path) for path in (report_path, config_path, public_path)}
    if (
        capture["schema"] != "stable-keyframe-camera-projection-v1"
        or report["completed"] is not True
        or not all(report["checks"].values())
        or report["config_sha256"] != before[config_path]
        or report["artifact_sha256"][public_path.name] != before[public_path]
    ):
        raise ValueError(
            "tracking diagnostic requires an intact completed public capture"
        )
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "schema": "development-tracking-gap-diagnostic-v1",
        "source_sha256": digest(Path(__file__)),
        "capture_inputs_sha256": {path.name: value for path, value in before.items()},
        "tracking": TemporalTrackConfig(maximum_frames=128).model_dump(mode="json"),
        "scope": "known development RGB; same detector/gates; no fitting, new controls, private poses or navigation acceptance",
        "navigation_accepted": False,
    }
    (output / "config.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True) + "\n"
    )
    (output / "probe.source.py").write_bytes(Path(__file__).read_bytes())
    samples = public["acquisition"]["samples"]
    frames = {}

    def frame(slot):
        if slot not in frames:
            relative = f"frames/frame-{slot:05d}.png"
            path = route / relative
            before[path] = digest(path)
            if before[path] != report["artifact_sha256"][relative]:
                raise ValueError("public RGB differs from capture ledger")
            with Image.open(path) as image:
                frames[slot] = np.asarray(image).copy()
            if frames[slot].shape != (448, 448, 3) or frames[slot].dtype != np.uint8:
                raise ValueError("diagnostic requires the unchanged public sensor")
        return frames[slot]

    records = []
    if samples:
        selected = previous = samples[0]["absolute_frame"]
        for sample in samples[1:]:
            end = sample["absolute_frame"]
            for namespace, start in (
                ("last_selected", selected),
                ("previous_stable", previous),
            ):
                if end - start + 1 > 128:
                    records.append(
                        {
                            "namespace": namespace,
                            "interval": [start, end],
                            "status": "raw_interval_exceeds_registered_tracking_budget",
                        }
                    )
                    continue
                records.append(
                    {
                        "namespace": namespace,
                        "interval": [start, end],
                        "status": "compared",
                        "comparison": compare(
                            [frame(slot) for slot in range(start, end + 1)]
                        ),
                    }
                )
            if sample["selected"]:
                selected = end
            previous = end
    result = {
        "config": inputs,
        "records": records,
        "checks": {
            "all_public_inputs_unchanged": all(
                digest(path) == value for path, value in before.items()
            )
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
                        "namespace": r["namespace"],
                        "interval": r["interval"],
                        "status": r["status"],
                        "direct": r.get("comparison", {})
                        .get("direct", {})
                        .get("survival"),
                        "continuous": r.get("comparison", {})
                        .get("continuous", {})
                        .get("survival"),
                    }
                    for r in records
                ],
            },
            sort_keys=True,
        )
    )
    if not all(result["checks"].values()):
        raise RuntimeError("tracking probe changed public inputs")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    probe(arguments.route, arguments.output)

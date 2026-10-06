"""Rank known startup views by a registered first-image-only ambiguity proxy.

This consumes exactly twelve past public RGB images, executes no LOOK, replaces
no original view or outcome, and does not diagnose ground-truth identity/depth.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import public_view_quality as quality

from minecraft_scientist.exploration import (
    correspondence_refinement,
    dense_correspondence,
    safe_track_seeds,
    temporal_tracks,
    visual_geometry,
)

DEPENDENCIES = {
    "safe-seed.source.py": ("safe_track_seeds.py", safe_track_seeds),
    "temporal-tracks.source.py": ("temporal_tracks.py", temporal_tracks),
    "correspondence-refinement.source.py": (
        "correspondence_refinement.py",
        correspondence_refinement,
    ),
    "visual-geometry.source.py": ("visual_geometry.py", visual_geometry),
    "dense-correspondence.source.py": ("dense_correspondence.py", dense_correspondence),
}

ORDINALS = list(range(15, 49, 3))
SCHEDULE = [["wait", None]] * 12 + [["look", None], ["wait", None], ["wait", None]] * 12
SCHEMAS = {
    "long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1",
    "sustained-long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1",
    "support-loss-long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded(route, relative):
    path = route / relative
    if path.is_symlink() or not path.resolve().is_relative_to(route):
        raise ValueError("survey input cannot escape the capture directory")
    return path


def diagnose(route, output):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to replace a public survey diagnosis")
    files = {
        name: bounded(route, name)
        for name in (
            "config.json",
            "report.json",
            "audit.json",
            "audit.source.py",
            "events.jsonl",
        )
    }
    before = {path: digest(path) for path in files.values()}
    config, capture, audit = [
        json.loads(files[name].read_bytes())
        for name in ("config.json", "report.json", "audit.json")
    ]
    if (
        config.get("schema") not in SCHEMAS
        or config.get("reference_indices") != ORDINALS
        or config.get("schedule") != SCHEDULE
        or capture.get("completed") is not True
        or capture.get("failure") is not None
        or not capture.get("checks")
        or not all(value is True for value in capture["checks"].values())
        or capture.get("config_sha256") != before[files["config.json"]]
        or capture.get("artifact_sha256", {}).get("events.jsonl")
        != before[files["events.jsonl"]]
        or audit.get("valid") is not True
        or not audit.get("checks")
        or not all(value is True for value in audit["checks"].values())
        or audit.get("audit_snapshot_sha256") != before[files["audit.source.py"]]
    ):
        raise ValueError("requires an intact independently audited startup survey")
    sources = {
        "probe.source.py": Path(__file__),
        "view-quality.source.py": Path(quality.__file__),
        **{name: Path(module.__file__) for name, (_, module) in DEPENDENCIES.items()},
    }
    hashes = {name: digest(path) for name, path in sources.items()}
    for name, (relative, _) in DEPENDENCIES.items():
        key = "package/exploration/" + relative
        snapshot = bounded(route, "source-snapshot/" + key)
        if (
            config.get("source_sha256", {}).get(key) != hashes[name]
            or digest(snapshot) != hashes[name]
        ):
            raise ValueError(
                "survey diagnostic dependency differs from capture snapshot"
            )
        before[snapshot] = hashes[name]
    images, pixel_hashes = [], []
    for ordinal in ORDINALS:
        relative = f"frames/frame-{ordinal:05d}.png"
        path = bounded(route, relative)
        before[path] = digest(path)
        if capture["artifact_sha256"].get(relative) != before[path]:
            raise ValueError("public survey RGB differs from original capture ledger")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (448, 448):
                raise ValueError("cannot transform the registered public RGB sensor")
            rgb = np.array(image)
        images.append(rgb)
        pixel_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    settings = {
        "schema": "known-first-image-public-survey-diagnostic-v2",
        "capture_schema": config["schema"],
        "profile": asdict(quality.ViewQualityProfile()),
        "view_ordinals": ORDINALS,
        "rgb_sha256": pixel_hashes,
        "sources_sha256": hashes,
        "inputs_sha256": {
            path.relative_to(route).as_posix(): value for path, value in before.items()
        },
        "scope": "Exactly twelve already consumed public startup images; independently audited metadata used only for attestation. No later pixels, actual survivor, private pose or outcome enters first-image ranking. No control execution, original selection replacement, map or navigation proof.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n"
    )
    for name, path in sources.items():
        (output / name).write_bytes(path.read_bytes())
    result = quality.rank_views(images, ORDINALS)
    if any(digest(path) != value for path, value in before.items()):
        raise ValueError("survey inputs changed during ranking")
    if any(digest(sources[name]) != value for name, value in hashes.items()):
        raise ValueError("survey diagnostic sources changed during ranking")
    result.update(
        config=settings,
        checks={
            "all_public_inputs_unchanged": True,
            "all_diagnostic_sources_unchanged": True,
            "used_dependencies_match_capture_snapshots": True,
            "only_twelve_original_startup_images_consumed": True,
            "no_original_record_or_control_replaced": True,
        },
    )
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "checks": result["checks"],
                "selected_frame": result["selected_frame"],
                "views": [
                    {
                        key: view[key]
                        for key in (
                            "absolute_frame",
                            "seed_count",
                            "distinctive_seed_count",
                            "distinctive_cells",
                            "eligible_proxy",
                        )
                    }
                    for view in result["views"]
                ],
                "executed_reorientation": False,
                "navigation_accepted": False,
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
    diagnose(args.route, args.output)

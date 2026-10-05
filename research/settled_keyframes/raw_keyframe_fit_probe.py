"""Past-only known-data test of identity-preserving raw tracks at selected views.

This does not choose new views, replace a live proposal or validate future motion.
Every intermediate raw image is tracked before selecting existing sample slots.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minecraft_scientist.exploration.joint_multiview_map import (
    JointMapConfig,
    _project,
    fit_joint_map,
)
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    TemporalTracks,
    diagnose_frame_chain,
)


def sample_raw_tracks(tracks, slots):
    slots = tuple(slots)
    if (
        not 3 <= len(slots) <= 12
        or any(type(slot) is not int for slot in slots)
        or slots != tuple(sorted(set(slots)))
        or slots[0] != 0
        or slots[-1] != tracks.frame_count - 1
    ):
        raise ValueError("sampling requires ordered exact full-prefix endpoint slots")
    return TemporalTracks(
        image_shape=tracks.image_shape,
        frame_count=len(slots),
        seed_count=tracks.seed_count,
        surviving_counts=tuple(tracks.surviving_counts[slot] for slot in slots),
        track_ids=tracks.track_ids,
        pixels=tuple(tracks.pixels[slot] for slot in slots),
        minimum_patch_correlations=tracks.minimum_patch_correlations,
        chain_cycle_errors=tracks.chain_cycle_errors,
    )


def probe(route, output):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a raw-keyframe diagnostic")
    paths = [
        route / name
        for name in ("config.json", "report.json", "keyframe-acquisition.json")
    ]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {path: digest(path) for path in paths}
    capture, report, saved = [json.loads(path.read_bytes()) for path in paths]
    if (
        capture["schema"]
        not in (
            "stable-keyframe-camera-projection-v1",
            "micro-stable-keyframe-camera-projection-v1",
        )
        or report["completed"] is not True
        or not all(report["checks"].values())
        or report["config_sha256"] != before[paths[0]]
        or report["artifact_sha256"][paths[2].name] != before[paths[2]]
        or saved["acquisition"]["failure"] is not None
        or saved["acquisition"]["future"] is not None
        or saved["acquisition"]["frozen"] is not None
        or len(saved["acquisition"]["attempts"]) != 1
    ):
        raise ValueError("probe requires an intact single rejected past-only proposal")
    proposal = saved["acquisition"]["attempts"][0]["proposal"]
    ordinals = proposal["index"]["frame_ordinals"]
    if not 6 <= len(ordinals) <= 12 or not 3 <= ordinals[-1] - ordinals[0] + 1 <= 128:
        raise ValueError("complete raw prefix exceeds registered correspondence budget")
    sources = {"probe.py": Path(__file__)}
    package = Path(fit_joint_map.__code__.co_filename).parent.parent
    sources.update(
        {
            "package/" + path.relative_to(package).as_posix(): path
            for path in package.rglob("*.py")
        }
    )
    source_hashes = {name: digest(path) for name, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, path in sources.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    config = {
        "schema": "raw-selected-keyframe-past-diagnostic-v1",
        "scope": "Known development past images; no future/private reads, controls, changed gates or live promotion",
        "source_sha256": source_hashes,
        "input_sha256": {path.name: value for path, value in before.items()},
        "frame_ordinals": ordinals,
        "raw_interval": [ordinals[0], ordinals[-1]],
        "tracking": TemporalTrackConfig(maximum_frames=128).model_dump(mode="json"),
        "fitting": JointMapConfig().model_dump(mode="json"),
        "seed": saved["acquisition"]["seed"],
        "navigation_accepted": False,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    frames = []
    for ordinal in range(ordinals[0], ordinals[-1] + 1):
        relative = f"frames/frame-{ordinal:05d}.png"
        path = route / relative
        before[path] = digest(path)
        if before[path] != report["artifact_sha256"][relative]:
            raise ValueError("raw image differs from original capture hash")
        with Image.open(path) as image:
            frame = np.asarray(image).copy()
        if frame.shape != (448, 448, 3) or frame.dtype != np.uint8:
            raise ValueError("raw projection requires the unchanged public sensor")
        frames.append(frame)
    diagnosis = diagnose_frame_chain(
        frames, config=TemporalTrackConfig(maximum_frames=128)
    )
    projected = sample_raw_tracks(
        diagnosis.tracks, [value - ordinals[0] for value in ordinals]
    )
    original = TemporalTracks.model_validate(proposal["past"])
    results = {}
    for name, tracks in (("sparse_replay", original), ("raw_projected", projected)):
        results[name] = [
            fit_joint_map(
                tracks,
                saved["calibration"]["camera_intrinsics"],
                bootstrap_slot=slot,
                seed=config["seed"],
                config=JointMapConfig(),
            ).model_dump(mode="json")
            for slot in range(2, tracks.frame_count)
        ]
    depth_checks = {}
    for name, models in results.items():
        depth_checks[name] = [
            {
                "bootstrap_slot": model["bootstrap_slot"],
                "frames": [
                    {
                        "frame_slot": pose["frame_slot"],
                        "withheld_count": len(model["validation_points"]),
                        "positive_depth_count": int(
                            (
                                _project(
                                    np.asarray(model["validation_points"]),
                                    np.asarray(pose["rotation"]),
                                    np.asarray(pose["translation"]),
                                    np.asarray(model["camera"]),
                                )[1][:, 2]
                                > 0
                            ).sum()
                        ),
                    }
                    for pose in model["poses"]
                ],
            }
            for model in models
        ]
    checks = {
        "sparse_original_models_exactly_replayed": json.dumps(
            results["sparse_replay"], sort_keys=True
        )
        == json.dumps(proposal["maps"], sort_keys=True),
        "all_public_inputs_unchanged": all(
            digest(path) == value for path, value in before.items()
        ),
        "all_sources_unchanged": all(
            digest(path) == source_hashes[name] for name, path in sources.items()
        ),
        "all_source_snapshots_match": all(
            digest(output / "source-snapshot" / name) == value
            for name, value in source_hashes.items()
        ),
    }
    result = {
        "config": config,
        "checks": checks,
        "raw_diagnosis": diagnosis.model_dump(mode="json"),
        "projected": projected.model_dump(mode="json"),
        "models": results,
        "withheld_depth_checks": depth_checks,
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "checks": checks,
                "raw_survivors": len(projected.track_ids),
                "models": {
                    name: [
                        {
                            "bootstrap": model["bootstrap_slot"],
                            "supported": model["supported"],
                            "reason": model["reason"],
                        }
                        for model in models
                    ]
                    for name, models in results.items()
                },
            },
            sort_keys=True,
        )
    )
    if not all(checks.values()):
        raise RuntimeError(
            "raw-keyframe diagnostic failed evidence preservation/replay"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    probe(arguments.route, arguments.output)

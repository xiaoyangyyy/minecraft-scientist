"""Past-only full-proposal geometry comparison for audited safe-seed captures.

Every original bootstrap is replayed; raw tracking retains every intermediate
image and original first-image seed ID. No new controls, view selection, future
images, private poses, changed gates or navigation promotion are involved.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from raw_keyframe_fit_probe import sample_raw_tracks

from minecraft_scientist.exploration.joint_multiview_map import fit_joint_map
from minecraft_scientist.exploration.map_recovery import _digest
from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.stable_keyframe_map import (
    StableSafeSeedKeyframePolicy,
)
from minecraft_scientist.exploration.temporal_tracks import (
    diagnose_frame_chain,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_proposal(proposal, frames, camera, seed, policy):
    ordinals = proposal["index"]["frame_ordinals"]
    if (
        not 6 <= len(ordinals) <= 12
        or any(type(value) is not int for value in ordinals)
        or ordinals != sorted(set(ordinals))
        or len(frames) != ordinals[-1] - ordinals[0] + 1
        or not 3 <= len(frames) <= 128
        or proposal["frozen_at_frame"] != ordinals[-1]
    ):
        raise ValueError("requires the complete original bounded past prefix")
    settings = StableSafeSeedKeyframePolicy.model_validate(policy)
    seeds = safe_seeds(frames[0])
    detector = proposal["detector"]
    if (
        detector["registration"] != settings.seed_detection
        or detector["seed_image_frame"] != ordinals[0]
        or detector["seed_image_sha256"]
        != hashlib.sha256(frames[0].tobytes()).hexdigest()
        or detector["seed_pixels"] != seeds.tolist()
    ):
        raise ValueError("past proposal's frozen first-image detector does not replay")
    seeds.setflags(write=False)
    slots = [value - ordinals[0] for value in ordinals]
    selected = [frames[slot] for slot in slots]
    if proposal["index"]["rgb_sha256"] != [
        hashlib.sha256(frame.tobytes()).hexdigest() for frame in selected
    ]:
        raise ValueError("selected RGB hashes do not replay original proposal")
    sparse = diagnose_frame_chain(selected, config=settings.temporal, seed_pixels=seeds)
    if (
        sparse.tracks.model_dump(mode="json") != proposal["past"]
        or proposal["past"] != proposal["index"]["tracks"]
    ):
        raise ValueError("sparse past tracks do not replay original proposal")
    raw = diagnose_frame_chain(
        frames,
        config=settings.temporal.model_copy(update={"maximum_frames": 128}),
        seed_pixels=seeds,
    )
    if raw.tracks.seed_count != len(seeds) or sparse.tracks.seed_count != len(seeds):
        raise ValueError("tracking changed the frozen seed denominator")
    projected = sample_raw_tracks(raw.tracks, slots)
    models = {}
    for name, tracks in (
        ("sparse_replay", sparse.tracks),
        ("raw_projected", projected),
    ):
        models[name] = [
            fit_joint_map(
                tracks, camera, bootstrap_slot=slot, seed=seed, config=settings.fitting
            ).model_dump(mode="json")
            for slot in range(2, tracks.frame_count)
        ]
    if models["sparse_replay"] != proposal["maps"] or proposal[
        "required_bootstraps"
    ] != [
        model["bootstrap_slot"]
        for model in models["sparse_replay"]
        if model["supported"]
    ]:
        raise ValueError("sparse bootstrap models do not replay original proposal")
    return {
        "frame_ordinals": ordinals,
        "raw_interval": [ordinals[0], ordinals[-1]],
        "detector": detector,
        "raw_diagnosis": raw.model_dump(mode="json"),
        "projected": projected.model_dump(mode="json"),
        "models": models,
        "all_original_models_replayed": True,
        "raw_past_consensus_candidate": sum(
            model["supported"] for model in models["raw_projected"]
        )
        >= settings.consensus.minimum_models,
        "original_proposal_replaced": False,
        "future_validated": False,
        "navigation_accepted": False,
    }


def probe(route, output):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a safe-seed raw-fit diagnostic")
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
    config, capture, saved, audit = [
        json.loads(path.read_bytes()) for path in paths[:4]
    ]
    acquisition = saved["acquisition"]
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
        or acquisition["failure"] is not None
        or acquisition["frozen"] is not None
        or acquisition["future"] is not None
    ):
        raise ValueError(
            "requires an intact audited rejected past-only safe-seed capture"
        )
    settings = StableSafeSeedKeyframePolicy.model_validate(acquisition["policy"])
    if settings.model_dump(mode="json") != config["keyframe_policy"]:
        raise ValueError("capture policies do not match")
    package = Path(fit_joint_map.__code__.co_filename).parent.parent
    sources = {
        "probe.py": Path(__file__),
        "raw-clock-helper.py": Path(__file__).with_name("raw_keyframe_fit_probe.py"),
    }
    sources.update(
        {
            "package/" + path.relative_to(package).as_posix(): path
            for path in package.rglob("*.py")
        }
    )
    hashes = {name: digest(path) for name, path in sources.items()}
    declaration = {
        "schema": "known-safe-seed-raw-past-fit-diagnostic-v1",
        "scope": "All existing rejected past proposals and bootstraps; unchanged first-image seeds, original gates and complete raw prefixes. No new controls, view reselection, private inputs, future validation or navigation promotion.",
        "input_sha256": {path.name: value for path, value in before.items()},
        "source_sha256": hashes,
        "policy": settings.model_dump(mode="json"),
        "seed": acquisition["seed"],
        "navigation_accepted": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True) + "\n"
    )
    for name, path in sources.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    results = []
    for index, attempt in enumerate(acquisition["attempts"]):
        proposal = attempt["proposal"]
        if attempt["sha256"] != _digest(proposal):
            raise ValueError("original proposal bytes differ from sealed hash")
        ordinals = proposal["index"]["frame_ordinals"]
        if not 3 <= ordinals[-1] - ordinals[0] + 1 <= 128:
            results.append(
                {
                    "attempt": index,
                    "status": "raw_interval_exceeds_registered_tracking_budget",
                    "comparison": None,
                }
            )
            continue
        frames = []
        for ordinal in range(ordinals[0], ordinals[-1] + 1):
            name = f"frames/frame-{ordinal:05d}.png"
            path = route / name
            before[path] = digest(path)
            if before[path] != capture["artifact_sha256"][name]:
                raise ValueError("public RGB differs from original capture hash")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError("requires unchanged public RGB")
                frame = np.array(image)
            if frame.shape != (448, 448, 3) or frame.dtype != np.uint8:
                raise ValueError("cannot change the public sensor shape")
            frames.append(frame)
        results.append(
            {
                "attempt": index,
                "status": "compared",
                "comparison": compare_proposal(
                    proposal,
                    frames,
                    saved["calibration"]["camera_intrinsics"],
                    acquisition["seed"],
                    acquisition["policy"],
                ),
            }
        )
    checks = {
        "all_original_proposals_retained": len(results) == len(acquisition["attempts"]),
        "all_public_inputs_unchanged": all(
            digest(path) == value for path, value in before.items()
        ),
        "all_sources_unchanged": all(
            digest(path) == hashes[name] for name, path in sources.items()
        ),
        "all_source_snapshots_match": all(
            digest(output / "source-snapshot" / name) == value
            for name, value in hashes.items()
        ),
    }
    result = {
        "config": declaration,
        "checks": checks,
        "attempts": results,
        "public_inputs_sha256": {
            path.relative_to(route).as_posix(): value for path, value in before.items()
        },
        "future_validated": False,
        "navigation_accepted": False,
    }
    if not all(checks.values()):
        raise ValueError("diagnostic inputs or sources changed during analysis")
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "checks": checks,
                "attempts": [
                    {
                        "attempt": record["attempt"],
                        "status": record["status"],
                        "models": {
                            mode: [
                                {
                                    "bootstrap": model["bootstrap_slot"],
                                    "supported": model["supported"],
                                    "reason": model["reason"],
                                }
                                for model in models
                            ]
                            for mode, models in record["comparison"]["models"].items()
                        }
                        if record["comparison"]
                        else None,
                    }
                    for record in results
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

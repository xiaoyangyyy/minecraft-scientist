"""Explain original landmark losses from public RGB; never score private poses.

The historical acquisition source is replayed explicitly rather than mislabeled
as current. New diagnostic output must equal that frozen tracker on every prefix.
This is a development diagnosis, not a navigation promotion or fresh acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
from PIL import Image

from minecraft_scientist.exploration.keyframe_motion import KeyframeAttempt
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    diagnose_frame_chain,
)
from minecraft_scientist.exploration.visibility_acquisition import (
    VisibilityController,
    VisibilityPolicy,
)

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_inputs(route):
    files = [route / name for name in ("config.json", "report.json", "events.jsonl")]
    files.extend((route / "frames").glob("frame-*.png"))
    files.extend((route / "goal-reference").glob("view-*.png"))
    files.extend((route / "source-snapshot").rglob("*.py"))
    return {path.relative_to(route).as_posix(): path for path in sorted(files)}


def validate_public(route):
    manifest = json.loads((route / "config.json").read_bytes())
    raw = json.loads((route / "report.json").read_bytes())
    if (
        manifest["schema"] != "visibility-rgb-acquisition-v2"
        or raw["failure"] is not None
    ):
        raise ValueError(
            "diagnosis requires a complete historical landmark acquisition"
        )
    if (
        raw["passed"] is not True
        or set(raw["checks"])
        != {
            "execution_completed",
            "action_frame_alignment",
            "bounded_frames",
            "telemetry_alignment",
            "preregistration_unchanged",
            "input_sources_unchanged",
            "public_log_clean",
        }
        or not all(value is True for value in raw["checks"].values())
        or digest(route / "config.json") != raw["config_sha256"]
    ):
        raise ValueError("historical acquisition integrity failed")
    snapshots = {
        path.relative_to(route / "source-snapshot").as_posix(): path
        for path in (route / "source-snapshot").rglob("*.py")
    }
    if {name: digest(path) for name, path in snapshots.items()} != manifest[
        "source_sha256"
    ]:
        raise ValueError("historical source snapshot mismatch")
    expected = {f"frame-{index:05d}.png" for index in range(raw["frame_count"])}
    if {path.name for path in (route / "frames").iterdir()} != expected or len(
        raw["trials"]
    ) != 8:
        raise ValueError("retain every frame and all eight original trials")
    if {name: digest(route / "frames" / name) for name in sorted(expected)} != raw[
        "frame_sha256"
    ]:
        raise ValueError("historical PNG bytes mismatch")
    previous_end = -1
    for number, trial in enumerate(raw["trials"]):
        ticks = trial["end_frame"] - trial["anchor_frame"]
        if (
            trial["repetition"] != number // len(manifest["handle_pairs"])
            or trial["handles"]
            != manifest["handle_pairs"][number % len(manifest["handle_pairs"])]
            or not previous_end
            < trial["anchor_frame"]
            < trial["end_frame"]
            < raw["frame_count"]
            or len(trial["decisions"]) != ticks
            or len(trial["prefix_tracks"]) != ticks
            or [decision["tick"] for decision in trial["decisions"]]
            != list(range(1, ticks + 1))
            or trial["decisions"][-1]["instruction"] not in ("finish", "abort")
        ):
            raise ValueError(
                "historical trial ordering, population or prefix schedule mismatch"
            )
        previous_end = trial["end_frame"]
    path = snapshots["package/exploration/temporal_tracks.py"]
    name = "minecraft_scientist.exploration._historical_track_diagnosis"
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "minecraft_scientist.exploration"
    sys.modules[name] = module
    # A normal file loader writes __pycache__ inside the historical snapshot.
    # Compile verified bytes in memory instead; the old input tree stays read-only.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102 -- replay verified project source without cache writes
    return manifest, raw, module


def diagnose(route):
    manifest, raw, historical = validate_public(route)
    policy = VisibilityPolicy.model_validate(manifest["policy"])
    config = TemporalTrackConfig.model_validate(policy.temporal.model_dump())
    images = []
    for index in range(raw["frame_count"]):
        with Image.open(route / "frames" / f"frame-{index:05d}.png") as image:
            images.append(np.asarray(image.convert("RGB")))
    if [hashlib.sha256(image.tobytes()).hexdigest() for image in images] != raw[
        "rgb_sha256"
    ]:
        raise ValueError("historical RGB bytes mismatch")
    trials = []
    for trial in raw["trials"]:
        anchor = trial["anchor_frame"]
        controller = VisibilityController(tuple(trial["handles"]), policy=policy)
        prior_ids = None
        protected = None
        steps = []
        for recorded in trial["decisions"]:
            tick = recorded["tick"]
            prefix = images[anchor : anchor + tick + 1]
            diagnosis = diagnose_frame_chain(prefix, config=config)
            old = historical.track_frame_chain(
                prefix,
                config=historical.TemporalTrackConfig.model_validate(
                    config.model_dump()
                ),
            )
            if old.model_dump(mode="json") != diagnosis.tracks.model_dump(mode="json"):
                raise ValueError("diagnostic changed the historical tracking output")
            if (
                diagnosis.tracks.model_dump(mode="json")
                != trial["prefix_tracks"][tick - 1]
            ):
                raise ValueError("diagnostic failed recorded public track replay")
            landmark = next(
                (
                    KeyframeAttempt.model_validate(item)
                    for item in trial["landmark_attempts"]
                    if item["end_offset"] == tick and item["motion"]["supported"]
                ),
                None,
            )
            decision = controller.observe(diagnosis.tracks, landmark=landmark)
            if decision.model_dump(mode="json") != recorded:
                raise ValueError(
                    "diagnostic prefix failed historical controller replay"
                )
            if landmark is not None:
                protected = {
                    "training": set(landmark.motion.training_track_ids),
                    "validation": set(landmark.motion.validation_track_ids),
                }
            available = set(decision.retained_ids)
            loss_lookup = {
                identity: event
                for event in diagnosis.rejections
                for identity in event.track_ids
            }
            newly_lost = set() if prior_ids is None else prior_ids - available
            if not newly_lost <= loss_lookup.keys():
                raise ValueError("newly lost identity lacks an actual pixel rejection")
            group_losses = {}
            for group, identities in (protected or {}).items():
                group_losses[group] = dict(
                    Counter(
                        loss_lookup[identity].reason
                        for identity in identities & newly_lost
                    )
                )
            steps.append(
                {
                    "tick": tick,
                    "instruction": decision.instruction,
                    "diagnosis": diagnosis.model_dump(mode="json"),
                    "newly_lost": {
                        str(identity): loss_lookup[identity].model_dump(mode="json")
                        for identity in sorted(newly_lost)
                    },
                    "protected_new_losses": group_losses,
                    "protected_remaining": {
                        group: len(identities & available)
                        for group, identities in (protected or {}).items()
                    },
                }
            )
            prior_ids = available
        trials.append({"anchor_frame": anchor, "steps": steps})
    return {
        "trials": trials,
        "prefix_count": sum(len(trial["steps"]) for trial in trials),
        "historical_tracks_unchanged": True,
        "historical_decisions_replayed": True,
        "navigation_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route = args.route.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = public_inputs(route)
    package = ROOT / "src/minecraft_scientist"
    sources = {
        f"package/{path.relative_to(package).as_posix()}": path
        for path in sorted(package.rglob("*.py"))
    } | {Path(__file__).name: Path(__file__)}
    config = {
        "schema": "public-track-loss-diagnosis-v1",
        "route": str(route),
        "opencv_version": cv2.__version__,
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "scope": "historical public-pixel replay; not current acquisition accuracy or navigation",
    }
    config_path = args.output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_hash = digest(config_path)
    for name, source in sources.items():
        target = args.output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    result = {"completed": False, "failure": None, "config_sha256": config_hash}
    try:
        result.update(diagnose(route))
        result["completed"] = True
    except Exception as error:  # noqa: BLE001 -- retain failed development evidence
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["inputs_unchanged"] = config["input_sha256"] == {
        name: digest(path) for name, path in public_inputs(route).items()
    }
    result["sources_unchanged"] = all(
        digest(path) == config["source_sha256"][name] for name, path in sources.items()
    )
    result["preregistration_unchanged"] = digest(config_path) == config_hash
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}))
    if not result["completed"] or not all(
        result[key]
        for key in (
            "inputs_unchanged",
            "sources_unchanged",
            "preregistration_unchanged",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

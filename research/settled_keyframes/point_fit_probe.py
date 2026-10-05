"""Past-only fixed-pose point diagnostic, never a failed-map rescue/promotion.

Already observed public endpoints are development counterexamples, not new
holdouts. Camera poses, partitions, training points and acceptance gates remain
unchanged. No private camera/player files are inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minecraft_scientist.exploration.joint_multiview_map import (
    JointMapEvidence,
    _camera,
    _check,
    _linearize,
    _project,
    _triangulate_many,
    validate_joint_future,
)
from minecraft_scientist.exploration.map_recovery import _digest
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.shared_anchor_map import original_shared_root
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    TemporalTracks,
    track_frame_chain,
)


def _point_costs(points, pixels, rotations, translations, camera, delta):
    costs = np.zeros(len(points))
    for observed, rotation, translation in zip(
        pixels, rotations, translations, strict=True
    ):
        predicted, transformed = _project(points, rotation, translation, camera)
        norms = np.linalg.norm(predicted - observed, axis=1)
        costs += np.where(norms <= delta, 0.5 * norms**2, delta * (norms - 0.5 * delta))
        costs[(transformed[:, 2] <= 1e-9) | ~np.isfinite(norms)] = np.inf
    return costs


def refine_past_points(pixels, rotations, translations, camera, *, config, scale):
    """DLT + bounded independent point LM; all poses/intrinsics are read-only."""
    pixels, rotations, translations, camera = [
        np.array(value, dtype=float, copy=True)
        for value in (pixels, rotations, translations, camera)
    ]
    if (
        pixels.ndim != 3
        or pixels.shape[2] != 2
        or len(pixels) < 2
        or not pixels.shape[1]
        or rotations.shape != (len(pixels), 3, 3)
        or translations.shape != (len(pixels), 3)
        or camera.shape != (3, 3)
        or not np.isfinite(scale)
        or scale <= 0
        or not all(
            np.isfinite(item).all()
            for item in (pixels, rotations, translations, camera)
        )
        or not np.allclose(camera[2], (0, 0, 1))
        or camera[0, 0] <= 0
        or camera[1, 1] <= 0
        or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3))
        or not np.allclose(np.linalg.det(rotations), 1)
    ):
        raise ValueError("point diagnostic needs finite fixed-pose pinhole rays")
    points = _triangulate_many(pixels, rotations, translations, camera)
    if points is None or not np.isfinite(points).all():
        return None
    delta = config.reference_huber_pixels * scale
    costs = _point_costs(points, pixels, rotations, translations, camera, delta)
    if not np.isfinite(costs).all():
        return None
    initial_points, initial_costs = points.copy(), costs.copy()
    history = [float(costs.sum())]
    damping = np.full(len(points), config.initial_damping)
    for _ in range(config.iterations):
        normal = np.zeros((len(points), 3, 3))
        gradient = np.zeros((len(points), 3))
        for observed, rotation, translation in zip(
            pixels, rotations, translations, strict=True
        ):
            _, jacobian, residual = _linearize(
                points, rotation, translation, observed, camera, delta
            )
            normal += np.einsum("nki,nkj->nij", jacobian, jacobian)
            gradient += np.einsum("nki,nk->ni", jacobian, residual)
        regularized = normal + (
            damping[:, None, None]
            * np.eye(3)[None]
            * np.maximum(np.diagonal(normal, axis1=1, axis2=2), 1e-6)[:, None]
        )
        try:
            step = np.linalg.solve(regularized, -gradient[..., None])[..., 0]
        except np.linalg.LinAlgError:
            damping = np.minimum(damping * 10, 1e12)
            continue
        trial = points + step
        trial_costs = _point_costs(
            trial, pixels, rotations, translations, camera, delta
        )
        improving = np.isfinite(trial_costs) & (trial_costs < costs)
        points[improving], costs[improving] = trial[improving], trial_costs[improving]
        damping[improving] = np.maximum(damping[improving] / 3, 1e-9)
        damping[~improving] = np.minimum(damping[~improving] * 10, 1e12)
        history.append(float(costs.sum()))
    return points, {
        "initial_cost": float(initial_costs.sum()),
        "final_cost": float(costs.sum()),
        "cost_history": history,
        "maximum_point_change": float(
            np.linalg.norm(points - initial_points, axis=1).max()
        ),
        "every_point_cost_nonincreasing": bool(np.all(costs <= initial_costs)),
        "fitting_frames": len(pixels),
        "fitting_points": len(points),
        "poses_refined": False,
    }


def revise_points(model, original, consumed):
    """Recompute every consumed-frame check, not just replacement point bytes."""
    if not model.poses:
        return model, {"status": "retained_without_geometry"}
    original_indices = {
        identity: slot for slot, identity in enumerate(original.track_ids)
    }
    consumed_indices = {
        identity: slot for slot, identity in enumerate(consumed.track_ids)
    }
    if not set(model.validation_ids) <= original_indices.keys():
        raise ValueError("cannot replenish the original withheld partition")
    rotations = np.asarray([pose.rotation for pose in model.poses])
    translations = np.asarray([pose.translation for pose in model.poses])
    camera = _camera(model.camera)
    result = refine_past_points(
        np.asarray(original.pixels)[
            :, [original_indices[identity] for identity in model.validation_ids]
        ],
        rotations[: original.frame_count],
        translations[: original.frame_count],
        camera,
        config=model.config,
        scale=original.image_shape[1] / 224,
    )
    if result is None:
        return model, {"status": "invalid_past_triangulation_retained"}
    points, diagnostic = result
    pixels = np.asarray(consumed.pixels)
    train_slots = [consumed_indices[identity] for identity in model.training_ids]
    valid_slots = [consumed_indices[identity] for identity in model.validation_ids]
    checks = tuple(
        _check(
            slot,
            np.asarray(model.training_points),
            points,
            pixels[slot, train_slots],
            pixels[slot, valid_slots],
            rotations[slot],
            translations[slot],
            camera,
            consumed.image_shape,
            model.config.geometry,
        )
        for slot in range(consumed.frame_count)
    )
    return JointMapEvidence.model_validate(
        model.model_dump()
        | {
            "validation_points": tuple(map(tuple, points)),
            "frame_checks": checks,
            "supported": all(check.supported for check in checks),
            "reason": "development-only original-past fixed-pose point refinement",
        }
    ), diagnostic | {"status": "refined_without_pose_updates"}


def probe(route, output):
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a development diagnostic")
    config_path, report_path = route / "config.json", route / "report.json"
    capture = json.loads(report_path.read_bytes())
    capture_config = json.loads(config_path.read_bytes())
    active_path = route / "active-acquisition.json"
    active_sha = hashlib.sha256(active_path.read_bytes()).hexdigest()
    public = json.loads(active_path.read_bytes())
    trial = public["shared_endpoint"]
    if (
        capture_config["schema"] != "shared-endpoint-camera-projection-v1"
        or not all(capture["checks"].values())
        or capture["artifact_sha256"]["active-acquisition.json"] != active_sha
        or trial["initial_endpoint"] is None
        or trial["final_endpoint"] is None
        or _digest(trial["frozen"]) != trial["frozen_sha256"]
        or _digest(trial["initial_endpoint"]) != trial["initial_endpoint_sha256"]
    ):
        raise ValueError("requires intact public capture and frozen endpoints")
    parent = public["maintenance"]
    version = parent["shared_anchor_versions"][0]
    root = parent["epochs"][version["parent_epoch"]]["frozen"]
    if _digest(root["snapshot"]) != version["original_snapshot_sha256"]:
        raise ValueError("original root lineage changed")
    _, original, original_models = original_shared_root(root["snapshot"])
    consumed = TemporalTracks.model_validate(trial["frozen"]["past"])
    models = [JointMapEvidence.model_validate(item) for item in trial["frozen"]["maps"]]
    if tuple(item.bootstrap_slot for item in models) != tuple(
        item.bootstrap_slot for item in original_models
    ):
        raise ValueError("must retain every original candidate including failures")
    package_root = Path(__file__).resolve().parents[2] / "src/minecraft_scientist"
    sources = {
        name: value
        for name, value in capture_config["source_sha256"].items()
        if name.startswith("package/")
    }

    def sources_unchanged():
        return all(
            hashlib.sha256((package_root / name[8:]).read_bytes()).hexdigest() == value
            for name, value in sources.items()
        )

    if not sources_unchanged():
        raise ValueError("diagnostic requires unchanged captured package sources")
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "schema": "development-past-only-point-fit-v1",
        "public_parent_sha256": active_sha,
        "capture_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "capture_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "old_proposal_sha256": trial["frozen_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_source_sha256": sources,
        "scope": "known same-evidence development diagnostic, not fresh validation",
        "point_fitting_frames": original.frame_count,
        "poses_refined": False,
        "new_controls": 0,
        "metric_scale_known": False,
        "navigation_accepted": False,
    }
    (output / "config.json").write_text(json.dumps(inputs, indent=2, sort_keys=True))
    (output / "probe.source.py").write_bytes(Path(__file__).read_bytes())
    proposals, diagnostics = zip(
        *(revise_points(model, original, consumed) for model in models), strict=True
    )
    endpoints, retained = [], set(consumed.track_ids)
    for name in ("initial_endpoint", "final_endpoint"):
        end = trial[name]["absolute_frame"]
        frames = []
        for slot in range(trial["frozen"]["anchor_frame"], end + 1):
            relative = f"frames/frame-{slot:05d}.png"
            path = route / relative
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != capture["artifact_sha256"][relative]
            ):
                raise ValueError("public image differs from captured hash")
            with Image.open(path) as image:
                rgb = np.asarray(image).copy()
            if rgb.shape != (448, 448, 3) or rgb.dtype != np.uint8:
                raise ValueError("diagnostic requires registered public sensor")
            frames.append(rgb)
        tracked = track_frame_chain(
            frames, config=TemporalTrackConfig(maximum_frames=64)
        )
        retained &= set(tracked.track_ids)
        tracked = retain_tracks(tracked, retained)
        endpoints.append(
            {
                "name": name,
                "absolute_frame": end,
                "candidates": [
                    {
                        "bootstrap_slot": new.bootstrap_slot,
                        "original": validate_joint_future(
                            old,
                            consumed,
                            tracked,
                            target_slots=(tracked.frame_count - 1,),
                        ),
                        "refined": validate_joint_future(
                            new,
                            consumed,
                            tracked,
                            target_slots=(tracked.frame_count - 1,),
                        ),
                    }
                    for old, new in zip(models, proposals, strict=True)
                ],
            }
        )
    result = {
        "config": inputs,
        "models": [model.model_dump(mode="json") for model in proposals],
        "diagnostics": list(diagnostics),
        "endpoints": endpoints,
        "checks": {
            "public_parent_unchanged": hashlib.sha256(
                active_path.read_bytes()
            ).hexdigest()
            == active_sha,
            "old_proposal_unchanged": _digest(trial["frozen"])
            == inputs["old_proposal_sha256"],
            "package_sources_unchanged": sources_unchanged(),
            "poses_training_and_partitions_unchanged": all(
                all(
                    getattr(old, field) == getattr(new, field)
                    for field in (
                        "poses",
                        "camera",
                        "training_points",
                        "training_ids",
                        "reserved_training_ids",
                        "validation_ids",
                        "config",
                        "training_cost_history",
                        "input_sha256",
                        "bootstrap_slot",
                    )
                )
                for old, new in zip(models, proposals, strict=True)
            ),
            "all_original_candidates_retained": len(proposals) == len(original_models),
        },
        "metric_scale_known": False,
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"checks": result["checks"], "output": str(output)}, sort_keys=True)
    )
    if not all(result["checks"].values()):
        raise RuntimeError("point diagnostic changed protected input evidence")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    probe(arguments.route, arguments.output)

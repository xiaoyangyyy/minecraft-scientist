"""Development-only training optimizer comparison; never navigation promotion.

This reuses already observed public evidence. It is not fresh-world validation,
new-control evidence, a released backend, or a rescue of an earlier failed map.
Private camera/player files are deliberately absent from the input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from minecraft_scientist.exploration import shared_anchor_map as shared
from minecraft_scientist.exploration.joint_multiview_map import (
    _cost,
    validate_joint_future,
)
from minecraft_scientist.exploration.map_recovery import _digest
from minecraft_scientist.exploration.second_leg_shadow import retain_tracks
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    TemporalTracks,
    track_frame_chain,
)


def ceres_bundle(
    points,
    rotations,
    translations,
    pixels,
    camera,
    bootstrap,
    config,
    cv2,
    *,
    pixel_scale=1.0,
):
    """Same public training residuals/loss/budget, fixed origin and unit gauge."""
    import pycolmap

    if pycolmap.__version__ != "4.0.1":
        raise ValueError("probe requires the separately pinned PyCOLMAP 4.0.1")
    points, rotations, translations, pixels, camera = [
        np.array(item, dtype=float, copy=True)
        for item in (points, rotations, translations, pixels, camera)
    ]
    if (
        not np.allclose(camera[:, 2], (224, 224, 1), atol=1e-6)
        or not np.allclose(rotations[0], np.eye(3), atol=1e-6)
        or not np.allclose(translations[0], 0, atol=1e-6)
    ):
        raise ValueError(
            "probe requires the registered 448-pixel sensor and original origin"
        )
    frames, count = len(rotations), len(points)
    delta = config.reference_huber_pixels * pixel_scale
    initial = _cost(points, rotations, translations, pixels, camera, delta)
    if not np.isfinite(initial):
        return None
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(
        pycolmap.Camera(
            model="PINHOLE",
            width=448,
            height=448,
            camera_id=1,
            params=[camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2]],
        )
    )
    adjustment = pycolmap.BundleAdjustmentConfig()
    for slot in range(frames):
        image = pycolmap.Image(
            name=f"public-training-{slot:03d}",
            image_id=slot + 1,
            camera_id=1,
            keypoints=np.ascontiguousarray(pixels[slot]),
        )
        pose = pycolmap.Rigid3d(np.column_stack((rotations[slot], translations[slot])))
        reconstruction.add_image_with_trivial_frame(image, pose)
        adjustment.add_image(slot + 1)
    point_ids = [
        reconstruction.add_point3D(
            points[index],
            pycolmap.Track(
                [pycolmap.TrackElement(slot + 1, index) for slot in range(frames)]
            ),
        )
        for index in range(count)
    ]
    adjustment.set_constant_rig_from_world_pose(1)
    adjustment.set_constant_cam_intrinsics(1)
    adjustment.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.print_summary = False
    options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    options.ceres.loss_function_scale = delta
    options.ceres.solver_options.max_num_iterations = config.iterations
    options.ceres.solver_options.max_solver_time_in_seconds = 30
    options.ceres.solver_options.num_threads = 1
    adjuster = pycolmap.create_default_bundle_adjuster(
        options, adjustment, reconstruction
    )
    adjuster.solve()
    points = np.array([reconstruction.point3D(identity).xyz for identity in point_ids])
    poses = [reconstruction.image(slot + 1).cam_from_world() for slot in range(frames)]
    rotations = np.array([pose.rotation.matrix() for pose in poses])
    translations = np.array([pose.translation for pose in poses])
    scale = np.linalg.norm(translations[bootstrap])
    if not np.isfinite(scale) or scale <= 1e-9:
        return None
    points /= scale
    translations /= scale
    final = _cost(points, rotations, translations, pixels, camera, delta)
    if not np.isfinite(final) or final > initial + 1e-8:
        return None
    return points, rotations, translations, [initial, min(initial, final)]


@contextmanager
def isolated_optimizer():
    original = shared._bundle
    shared._bundle = ceres_bundle
    try:
        yield
    finally:
        shared._bundle = original
        if shared._bundle is not original:
            raise RuntimeError("probe did not restore its process-local optimizer")


def probe(route, output):
    import pycolmap
    from PIL import Image

    route, output = Path(route).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a development probe")
    active_path = route / "active-acquisition.json"
    before = hashlib.sha256(active_path.read_bytes()).hexdigest()
    captured = json.loads((route / "report.json").read_bytes())
    capture_config = json.loads((route / "config.json").read_bytes())
    if (
        captured["artifact_sha256"]["active-acquisition.json"] != before
        or not all(captured["checks"].values())
        or capture_config["schema"] != "shared-endpoint-camera-projection-v1"
    ):
        raise ValueError("probe requires intact public endpoint capture evidence")
    public = json.loads(active_path.read_bytes())
    trial = public["shared_endpoint"]
    if (
        trial["frozen"] is None
        or trial["initial_endpoint"] is None
        or trial["final_endpoint"] is None
        or _digest(trial["frozen"]) != trial["frozen_sha256"]
        or _digest(trial["initial_endpoint"]) != trial["initial_endpoint_sha256"]
    ):
        raise ValueError("probe needs intact frozen map and both actual endpoints")
    output.mkdir(parents=True, exist_ok=False)
    original_optimizer = shared._bundle
    inputs = {
        "schema": "development-training-ceres-probe-v2",
        "capture_config_sha256": hashlib.sha256(
            (route / "config.json").read_bytes()
        ).hexdigest(),
        "capture_report_sha256": hashlib.sha256(
            (route / "report.json").read_bytes()
        ).hexdigest(),
        "package_source_sha256": {
            name: value
            for name, value in capture_config["source_sha256"].items()
            if name.startswith("package/")
        },
        "public_parent_sha256": before,
        "old_proposal_sha256": trial["frozen_sha256"],
        "backend_version": pycolmap.__version__,
        "backend_extension_sha256": hashlib.sha256(
            Path(pycolmap._core.__file__).read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "holdouts": "original partitions and original past triangulation only",
        "new_controls": 0,
        "metric_scale_known": False,
        "navigation_accepted": False,
        "scope": "same-evidence development diagnostic; not fresh validation or release",
    }
    (output / "config.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True) + "\n"
    )
    (output / "probe.source.py").write_bytes(Path(__file__).read_bytes())
    parent = public["maintenance"]
    version = parent["shared_anchor_versions"][0]
    root = parent["epochs"][version["parent_epoch"]]["frozen"]
    if _digest(root["snapshot"]) != version["original_snapshot_sha256"]:
        raise ValueError("probe cannot revise original map lineage")
    _, original, models = shared.original_shared_root(root["snapshot"])
    consumed = TemporalTracks.model_validate(trial["frozen"]["past"])
    with isolated_optimizer():
        proposals = [
            shared.refine_shared_crisis_map(model, original, consumed)
            for model in models
        ]
    endpoint_records = []
    retained = set(consumed.track_ids)
    anchor = trial["frozen"]["anchor_frame"]
    for name in ("initial_endpoint", "final_endpoint"):
        end = trial[name]["absolute_frame"]
        frames = []
        for slot in range(anchor, end + 1):
            relative = f"frames/frame-{slot:05d}.png"
            path = route / relative
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != captured["artifact_sha256"][relative]
            ):
                raise ValueError("public RGB differs from the capture ledger")
            with Image.open(path) as image:
                rgb = np.asarray(image).copy()
            if rgb.shape != (448, 448, 3) or rgb.dtype != np.uint8:
                raise ValueError("probe requires unchanged public RGB")
            frames.append(rgb)
        tracks = track_frame_chain(
            frames, config=TemporalTrackConfig(maximum_frames=64)
        )
        retained &= set(tracks.track_ids)
        tracks = retain_tracks(tracks, retained)
        endpoint_records.append(
            {
                "name": name,
                "absolute_frame": end,
                "candidates": [
                    {
                        "bootstrap_slot": model.bootstrap_slot,
                        "future": validate_joint_future(
                            model,
                            consumed,
                            tracks,
                            target_slots=(tracks.frame_count - 1,),
                        ),
                    }
                    for model in proposals
                ],
            }
        )
    result = {
        "config": inputs,
        "models": [model.model_dump(mode="json") for model in proposals],
        "endpoints": endpoint_records,
        "checks": {
            "public_parent_unchanged": hashlib.sha256(
                active_path.read_bytes()
            ).hexdigest()
            == before,
            "original_optimizer_restored": shared._bundle is original_optimizer,
            "old_proposal_unchanged": _digest(public["shared_endpoint"]["frozen"])
            == trial["frozen_sha256"],
            "package_sources_unchanged": all(
                hashlib.sha256(
                    (
                        Path(shared.__file__).resolve().parent.parent / name[8:]
                    ).read_bytes()
                ).hexdigest()
                == value
                for name, value in inputs["package_source_sha256"].items()
            ),
        },
        "navigation_accepted": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "checks": result["checks"],
                "models": [
                    {
                        "bootstrap": model.bootstrap_slot,
                        "supported": model.supported,
                        "reason": model.reason,
                        "cost": model.training_cost_history,
                    }
                    for model in proposals
                    if model.bootstrap_slot in trial["frozen"]["required_bootstraps"]
                ],
                "endpoints": [
                    {
                        "name": record["name"],
                        "candidates": [
                            {
                                "bootstrap": item["bootstrap_slot"],
                                "status": item["future"]["status"],
                                "checks": item["future"]["frame_checks"],
                            }
                            for item in record["candidates"]
                            if item["future"]["frame_checks"]
                        ],
                    }
                    for record in endpoint_records
                ],
            },
            sort_keys=True,
        )
    )
    if not all(result["checks"].values()):
        raise RuntimeError("development probe changed its input or optimizer")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    probe(arguments.route, arguments.output)

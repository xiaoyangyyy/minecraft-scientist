import copy
import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration import map_gauge_bridge as bridge
from minecraft_scientist.exploration.joint_multiview_map import (
    JointMapConfig,
    JointMapEvidence,
    JointPose,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks

CAMERA = np.array(((90.0, 0, 112), (0, 120.0, 112), (0, 0, 1)))


def rotation(slot):
    angle = slot * 0.015
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


@pytest.fixture
def roots():
    points = np.array(
        [
            (-1.8 + 0.5 * (i % 8), -1 + 0.6 * (i // 8), 4 + 0.4 * (i % 3))
            for i in range(32)
        ]
    )
    scale = 1.7
    matrix = rotation(2).T
    center = np.array((0.2, 0, 0))
    similarity = bridge.GaugeSimilarity(
        scale=scale, rotation=tuple(map(tuple, matrix)), translation=tuple(center)
    )
    child_points = similarity.inverse().points(points)

    def root(anchor):
        identities = tuple(range(32)) if anchor == 0 else tuple(reversed(range(32)))
        poses, pixels = [], []
        for absolute in range(anchor, anchor + 6):
            r = rotation(absolute)
            t = -r @ np.array((0.1 * absolute, 0, 0))
            if anchor:
                r, t = r @ matrix, (r @ center + t) / scale
            xyz = points if anchor == 0 else child_points
            projected, _ = bridge._project(xyz, r, t, CAMERA)
            pixels.append(tuple(map(tuple, projected)))
            poses.append(
                JointPose(
                    frame_slot=absolute - anchor,
                    rotation=tuple(map(tuple, r)),
                    translation=tuple(t),
                )
            )
        tracks = TemporalTracks(
            image_shape=(224, 224),
            frame_count=6,
            seed_count=32,
            surviving_counts=(32,) * 6,
            track_ids=identities,
            pixels=tuple(pixels),
            minimum_patch_correlations=(1.0,) * 32,
            chain_cycle_errors=(0.0,) * 32,
        )
        xyz = points if anchor == 0 else child_points
        maps = [
            JointMapEvidence(
                supported=True,
                reason="synthetic calibrated unit fixture; not runtime proof",
                config=JointMapConfig(),
                seed=2000,
                camera=tuple(map(tuple, CAMERA)),
                bootstrap_slot=slot,
                fitting_frames=6,
                input_sha256=hashlib.sha256(
                    tracks.model_dump_json().encode()
                ).hexdigest(),
                reserved_training_ids=identities[:24],
                training_ids=identities[:24],
                validation_ids=identities[24:],
                training_points=tuple(map(tuple, xyz[:24])),
                validation_points=tuple(map(tuple, xyz[24:])),
                poses=tuple(poses),
                training_cost_history=(0.0,),
                frame_checks=tuple(
                    {
                        "frame_slot": f,
                        "supported": True,
                        "reason": "unit fixture",
                        "training_inliers": 24,
                        "validation_tracks": 8,
                    }
                    for f in range(6)
                ),
            ).model_dump(mode="json")
            for slot in range(2, 6)
        ]
        return {
            "anchor_frame": anchor,
            "past": tracks.model_dump(mode="json"),
            "maps": maps,
            "prefixes": [],
            "decisions": [],
            "navigation_accepted": False,
            "metric_scale_known": False,
        }

    first, second = root(0), root(2)
    images = [
        [hashlib.sha256(str(i).encode()).hexdigest() for i in range(anchor, anchor + 6)]
        for anchor in (0, 2)
    ]
    return first, second, images, similarity


def register(roots):
    parent, child, images, _ = roots
    return bridge.register_map_gauges(
        parent,
        child,
        parent_images=images[0],
        child_images=images[1],
        parent_namespace="detector-parent",
        child_namespace="detector-child",
    )


def test_known_similarity_uses_pixels_not_namespace_id_equality_and_all_map_pairs(
    roots,
):
    first, second, _, truth = roots
    before = copy.deepcopy((first, second))
    result = register(roots)
    assert result["supported"]
    assert result["track_pairs"] == [[i, 31 - i] for i in range(32)]
    assert result["reference_pair"] == [2, 2]
    assert len(result["map_pairs"]) == 16
    assert all(
        item["supported"] and len(item["validation_pairs"]) == 8
        for item in result["map_pairs"]
    )
    estimated = bridge.GaugeSimilarity.model_validate(result["similarity"])
    assert estimated.scale == pytest.approx(truth.scale)
    assert np.allclose(estimated.rotation, truth.rotation)
    assert np.allclose(estimated.translation, truth.translation)
    assert (first, second) == before
    assert not result["global_gauge_aligned"] and not result["metric_scale_known"]
    assert not result["navigation_accepted"]


def test_similarity_executes_points_inverse_and_world_to_camera_pose(roots):
    first, second, _, truth = roots
    child_model = second["maps"][0]
    assert np.allclose(
        truth.points(child_model["training_points"]),
        first["maps"][0]["training_points"],
    )
    projected_pose = truth.pose(child_model["poses"][0])
    assert np.allclose(
        projected_pose.rotation, first["maps"][0]["poses"][2]["rotation"]
    )
    assert np.allclose(
        projected_pose.translation, first["maps"][0]["poses"][2]["translation"]
    )
    values = np.array(((1, 2, 3), (-3, 4, 8)))
    assert np.allclose(truth.inverse().points(truth.points(values)), values)


def test_withheld_points_cannot_change_fitted_transform_or_choose_runner_up(roots):
    original = register(roots)
    roots[1]["maps"][0]["validation_points"][0][0] += 2
    result = register(roots)
    assert not result["supported"] and result["similarity"] is None
    assert result["reference_pair"] == [2, 2]
    assert len(result["map_pairs"]) == 16
    assert (
        result["map_pairs"][0]["similarity"] == original["map_pairs"][0]["similarity"]
    )
    assert any(pair["supported"] for pair in result["map_pairs"])
    assert any(not pair["supported"] for pair in result["map_pairs"])


def test_insufficient_original_validation_correspondences_cannot_be_replaced(roots):
    roots[1]["maps"][0]["validation_ids"] = list(range(24, 32))
    roots[1]["maps"][0]["reserved_training_ids"] = list(range(24))
    roots[1]["maps"][0]["training_ids"] = list(range(24))
    result = register(roots)
    assert not result["supported"]
    assert len(result["map_pairs"][0]["validation_pairs"]) == 0
    assert result["map_pairs"][0]["similarity"] is None


def test_insufficient_parent_models_retains_remaining_candidate_without_promotion(
    roots,
):
    for item in roots[0]["maps"][1:]:
        item["supported"] = False
    result = register(roots)
    assert not result["supported"] and len(result["map_pairs"]) == 4
    assert result["reference_pair"] is None and result["similarity"] is None


def test_no_common_images_abstains_before_geometric_fit(roots):
    roots[1]["anchor_frame"] = 6
    result = register(roots)
    assert not result["supported"] and not result["overlap_frames"]
    assert result["map_pairs"] == []


def test_duplicate_pixel_paths_are_ambiguous_not_correspondences(roots):
    child = roots[1]
    for frame in child["past"]["pixels"]:
        frame[1] = frame[0][:]
    tracks = TemporalTracks.model_validate(child["past"])
    for model in child["maps"]:
        model["input_sha256"] = hashlib.sha256(
            tracks.model_dump_json().encode()
        ).hexdigest()
    result = register(roots)
    assert not result["supported"]
    assert not any(pair[0] == 0 for pair in result["track_pairs"])


def test_later_overlap_pose_failure_keeps_fitted_transform_and_rejects_entire_link(
    roots,
):
    original = register(roots)
    roots[1]["maps"][0]["poses"][2]["translation"][0] += 1
    result = register(roots)
    assert not result["supported"]
    assert (
        result["map_pairs"][0]["similarity"] == original["map_pairs"][0]["similarity"]
    )
    assert any(not check["supported"] for check in result["map_pairs"][0]["checks"])
    assert len(result["map_pairs"]) == 16


@pytest.mark.parametrize("case", ["coincident", "collinear", "reflection"])
def test_degenerate_or_reflected_training_geometry_cannot_be_promoted(roots, case):
    for model in roots[1]["maps"]:
        for index, point in enumerate(model["training_points"]):
            model["training_points"][index] = (
                [0.0, 0.0, 1.0]
                if case == "coincident"
                else [float(index), 2.0 * index, 3.0 * index]
                if case == "collinear"
                else [-point[0], point[1], point[2]]
            )
    result = register(roots)
    assert not result["supported"] and result["similarity"] is None


def test_gauge_namespace_identity_cannot_be_reused(roots):
    with pytest.raises(ValueError, match="namespaces"):
        bridge.register_map_gauges(
            roots[0],
            roots[1],
            parent_images=roots[2][0],
            child_images=roots[2][1],
            parent_namespace="same",
            child_namespace="same",
        )


@pytest.mark.parametrize(
    "case",
    [
        "pixels",
        "future",
        "lineage",
        "bootstrap",
        "camera",
        "image_hash",
        "hash_length",
        "past_hash",
    ],
)
def test_invalid_gauge_inputs_refuse_without_mutating_parent(roots, case):
    if case == "pixels":
        roots[1]["past"]["pixels"][0][0][0] = -1
    elif case == "future":
        roots[1]["prefixes"] = [{}]
    elif case == "lineage":
        roots[1]["anchor_frame"] = 0
    elif case == "bootstrap":
        roots[1]["maps"].pop()
    elif case == "camera":
        roots[1]["maps"][1]["camera"][0][0] += 1
    elif case == "image_hash":
        roots[2][1][0] = "0" * 64
    elif case == "hash_length":
        roots[2][1][0] = "wrong"
    else:
        roots[1]["maps"][0]["input_sha256"] = "0" * 64
    before = copy.deepcopy(roots[0])
    with pytest.raises(ValueError):
        register(roots)
    assert roots[0] == before


@pytest.mark.parametrize(
    "policy",
    [
        {"minimum_validation_pairs": 7},
        {"reference_validation_pixels": 2},
        {"minimum_overlap_frames": 2},
        {"maximum_rotation_error_degrees": 3},
        {"minimum_spatial_ratio": 0},
    ],
)
def test_policy_cannot_relax_preregistered_support(policy):
    with pytest.raises(ValidationError):
        bridge.GaugeBridgePolicy.model_validate(policy)


@pytest.mark.parametrize("case", ["scale", "reflection", "point", "pose"])
def test_executable_similarity_rejects_invalid_geometry(case):
    if case == "scale":
        with pytest.raises(ValidationError):
            bridge.GaugeSimilarity(
                scale=0, rotation=np.eye(3).tolist(), translation=(0, 0, 0)
            )
    elif case == "reflection":
        with pytest.raises(ValidationError):
            bridge.GaugeSimilarity(
                scale=1, rotation=np.diag((1, 1, -1)).tolist(), translation=(0, 0, 0)
            )
    else:
        similarity = bridge.GaugeSimilarity(
            scale=1, rotation=np.eye(3).tolist(), translation=(0, 0, 0)
        )
        with pytest.raises(ValueError):
            similarity.points(
                [[1, 2, float("nan")]]
            ) if case == "point" else similarity.pose(
                {
                    "frame_slot": 0,
                    "rotation": np.zeros((3, 3)).tolist(),
                    "translation": (0, 0, 0),
                }
            )

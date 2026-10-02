# ruff: noqa: F401 -- pytest fixture re-export
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")
from test_shadow_motion import CAMERA, motion  # noqa: F401, F811 -- fixture re-export

from minecraft_scientist.exploration.active_parallax import (
    homogeneous_direction_samples,
)
from minecraft_scientist.exploration.parallax_observability import measure_parallax
from minecraft_scientist.exploration.shadow_motion import validate_shadow_motion
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


def test_near_collinear_continuation_matches_known_camera_geometry(motion):
    candidate, tracks = motion
    shadow = validate_shadow_motion(candidate, tracks, CAMERA)
    result = measure_parallax(shadow, tracks, CAMERA)
    assert not result.conditionally_diverse_view and len(result.views) == 2
    # The fixture has linear camera-frame translations AND changing rotations;
    # its world-frame camera centers therefore trace a slightly bent path.
    center = -cv2.Rodrigues(np.array((0.01, 0.04, -0.02)))[0].T @ np.array(
        (-0.7, 0.05, -0.15)
    )
    for item, alpha in zip(result.views, (1.25, 1.5), strict=True):
        future = -cv2.Rodrigues(np.array((0.01, 0.04, -0.02)) * alpha)[0].T @ (
            np.array((-0.7, 0.05, -0.15)) * alpha
        )
        delta = future - center
        expected = np.degrees(
            np.arccos(
                np.clip(
                    center @ delta / (np.linalg.norm(center) * np.linalg.norm(delta)),
                    -1,
                    1,
                )
            )
        )
        assert item.turn_degrees == pytest.approx(expected, abs=0.1)
        assert item.turn_degrees < 45


def test_bent_synthetic_view_has_conditional_diversity_but_no_executable_pose(motion):
    candidate, tracks = motion
    rng = np.random.default_rng(123)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 9, 200))
    )
    rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)))[0]
    pixels = list(tracks.pixels[:5])
    for delta in (0.4, 0.8):
        projected = (
            xyz @ rotation.T + np.array((-0.7, 0.05 + delta, -0.15))
        ) @ CAMERA.T
        pixels.append(tuple(map(tuple, projected[:, :2] / projected[:, 2:])))
    bent = TemporalTracks.model_validate(tracks.model_dump() | {"pixels": pixels})
    shadow = validate_shadow_motion(candidate, bent, CAMERA)
    assert shadow.supported
    evidence = measure_parallax(shadow, bent, CAMERA)
    assert evidence.conditionally_diverse_view
    assert "unit_translation" not in evidence.model_dump()


def test_direction_learning_ignores_mixed_controls_and_shared_image_duplicates(motion):
    candidate, _ = motion
    result = {
        "pair": {"first": 0},
        "selection": {
            "terminal_offset": 4,
            "selected_offset": 4,
            "attempts": [candidate.model_dump(mode="json")],
        },
    }
    single = [
        {
            "start_frame": 0,
            "end_frame": 4,
            "scheduled": {"kind": "move", "motion_handle_index": 0},
        }
    ]
    assert len(homogeneous_direction_samples([result, result], single)) == 1
    mixed = [
        {
            "start_frame": 0,
            "end_frame": 2,
            "scheduled": {"kind": "move", "motion_handle_index": 0},
        },
        {
            "start_frame": 2,
            "end_frame": 4,
            "scheduled": {"kind": "move", "motion_handle_index": 3},
        },
    ]
    assert not homogeneous_direction_samples([result], mixed)
    assert not homogeneous_direction_samples(
        [result],
        [
            {
                "start_frame": 0,
                "end_frame": 4,
                "scheduled": {"kind": "wait", "motion_handle_index": None},
            }
        ],
    )

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig


@pytest.fixture
def comparison(monkeypatch):
    pytest.importorskip("PIL.Image", reason="requires optional visual extra")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("audit_camera_profile_comparison")


@pytest.fixture
def cells():
    shared_profile = {
        "original_runtime_sha256": {},
        "original_options_sha256": "empty-options",
        "source_sha256": {"camera": "same-source"},
    }
    shared_inference = {
        "focal_pixels": 180.0,
        "evaluation_thresholds": {"minimum_translation_blocks": 0.15},
        "source_sha256": {"inference": "same-source"},
    }
    result = []
    for enabled in (True, False):
        result.append(
            {
                "config": MotionProbeConfig().model_dump(mode="json"),
                "profile_config": shared_profile | {"enabled": enabled},
                "profile_audit": {"valid": True, "checks": {"replayed": True}},
                "motion_audit": {
                    "valid": True,
                    "checks": {"replayed": True},
                    "evaluation": {"groups": {"settled": {"trials": 8}}},
                },
                "inference_config": copy.deepcopy(shared_inference),
                "positions": [
                    {"xpos": 0.0, "ypos": 70.0, "zpos": 0.0, "yaw": 30.0, "pitch": 0.0}
                ]
                * 201,
                "predictions": [
                    {
                        "pair": {"group": "settled"},
                        "tracks": {"track_ids": list(range(index))},
                    }
                    for index in range(8)
                ],
            }
        )
    return result


def test_comparison_is_order_independent_and_not_navigation(comparison, cells):
    result = comparison.summarize(cells)
    assert comparison.summarize(cells[::-1]) == result
    assert result["matched_trajectories"] and result["sessions_per_setting"] == 1
    assert result["cells"][0]["bobbing_enabled"] is True
    assert result["cells"][0]["settled_track_median"] == 3.5
    assert result["navigation_accepted"] is False


@pytest.mark.parametrize(
    "change",
    [
        "same-setting",
        "invalid-component",
        "world",
        "source",
        "threshold",
        "variant",
        "missing-trial",
    ],
)
def test_incomparable_or_incomplete_cases_are_not_valid_comparisons(
    comparison, cells, change
):
    if change == "same-setting":
        cells[1]["profile_config"]["enabled"] = True
    elif change == "invalid-component":
        cells[1]["motion_audit"]["checks"]["replayed"] = False
    elif change == "world":
        cells[1]["config"]["seed"] += 1
    elif change == "source":
        cells[1]["profile_config"]["source_sha256"] = {"camera": "different"}
    elif change == "threshold":
        cells[1]["inference_config"]["evaluation_thresholds"][
            "minimum_translation_blocks"
        ] = 0.01
    elif change == "variant":
        cells[0]["inference_config"]["variant"] = "consensus"
    else:
        cells[1]["predictions"].pop()
    with pytest.raises(ValueError):
        comparison.summarize(cells)


def test_nonfinite_and_shifted_trajectories_are_distinguished(comparison, cells):
    cells[1]["positions"] = [dict(point, xpos=0.5) for point in cells[1]["positions"]]
    result = comparison.summarize(cells)
    assert not result["matched_trajectories"]
    assert result["maximum_trajectory_component_difference"] == 0.5
    cells[1]["positions"][10]["xpos"] = float("nan")
    with pytest.raises(ValueError, match="finite aligned"):
        comparison.summarize(cells)


def test_pixel_diagnostic_excludes_hud_and_retains_every_frame(comparison, monkeypatch):
    first = np.zeros((40, 40, 3), dtype=np.uint8)
    second = first.copy()
    second[:3] = 255
    second[30:] = 255
    monkeypatch.setattr(
        comparison, "rgb", lambda path: first if "first" in path.parts else second
    )
    result = comparison.compare_pixels(Path("first"), Path("second"))
    assert len(result["frame_mean_absolute_rgb_difference"]) == 201
    assert result["nonidentical_frames"] == 0
    second[10:20] = 100
    result = comparison.compare_pixels(Path("first"), Path("second"))
    assert result["nonidentical_frames"] == 201
    assert result["median_mean_absolute_rgb_difference"] > 0

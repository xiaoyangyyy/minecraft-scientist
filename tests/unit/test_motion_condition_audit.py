import copy
import importlib
from pathlib import Path

import pytest

pytest.importorskip("PIL", reason="requires optional visual extra")

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig


@pytest.fixture
def summary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("audit_controlled_motion_conditions").summarize


def _cells():
    return [
        {
            "config": MotionProbeConfig(time_of_day=time, weather=weather).model_dump(),
            "audit": {
                "valid": True,
                "checks": {"replayed": True},
                "evaluation": {"groups": {}},
            },
            "positions": [{"xpos": 1, "ypos": 2, "zpos": 3, "yaw": 0, "pitch": 0}] * 3,
            "predictions": [
                {
                    "pair": {"group": "settled"},
                    "tracks": {"track_ids": list(range(10 + index))},
                }
                for index in range(8)
            ],
            "inference_config": {"focal_pixels": 180},
        }
        for time in ("noon", "midnight")
        for weather in ("clear", "thunder")
    ]


def test_factorial_keeps_one_session_per_cell_and_all_tracks(summary):
    result = summary(_cells())
    assert result["matched_trajectories"] is True
    assert result["sessions_per_cell"] == 1 and result["navigation_accepted"] is False
    assert all(item["settled_track_median"] == 13.5 for item in result["cells"])


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "budget",
        "failed",
        "nonfinite",
        "unaligned",
        "missing_trial",
    ],
)
def test_incomplete_or_incomparable_conditions_cannot_be_promoted(summary, defect):
    cells = copy.deepcopy(_cells())
    if defect == "missing":
        cells.pop()
    elif defect == "duplicate":
        cells[0] = cells[1]
    elif defect == "budget":
        cells[0]["config"]["seed"] = 9
    elif defect == "failed":
        cells[0]["audit"]["checks"]["replayed"] = False
    elif defect == "nonfinite":
        cells[0]["positions"][0]["xpos"] = float("nan")
    elif defect == "unaligned":
        cells[0]["positions"].pop()
    else:
        cells[0]["predictions"].pop()
    with pytest.raises(ValueError):
        summary(cells)


def test_physical_mismatch_is_reported_not_silently_normalized(summary):
    cells = copy.deepcopy(_cells())
    cells[0]["positions"][0]["xpos"] += 1
    result = summary(cells)
    assert result["matched_trajectories"] is False
    assert result["maximum_trajectory_component_difference"] == 1

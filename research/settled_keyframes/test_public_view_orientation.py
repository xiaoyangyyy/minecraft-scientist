import hashlib

import numpy as np
import pytest

pytest.importorskip("cv2")

import public_view_orientation as orientation
from test_public_view_quality import texture


@pytest.fixture
def scene(monkeypatch):
    """Mock ranking isolates timing/receipt contract, not native heading evidence."""
    images = [texture(448) for _ in range(12)]
    picked = {"selected_frame": 33}
    monkeypatch.setattr(orientation, "rank_views", lambda *args: picked.copy())
    calls = []

    def execute(kind):
        calls.append(kind)
        return orientation.SurveyReceipt(
            images[-1].copy(), kind, kind == "look", True, 1
        )

    return images, picked, calls, execute


@pytest.mark.parametrize("selected,turns", [(15, 1), (33, 7), (45, 11), (48, 0)])
def test_exact_program_has_real_contiguous_clock_and_cannot_claim_heading(
    scene, selected, turns
):
    images, picked, calls, execute = scene
    before = [image.copy() for image in images]
    picked["selected_frame"] = selected
    outcome = orientation.orient(images, execute)
    result = outcome.journal
    assert calls == ["look", "wait", "wait"] * turns
    assert result["requested_look_ticks"] == turns
    assert result["end_frame"] == 48 + 3 * turns <= 81
    assert [step["absolute_frame"] for step in result["steps"]] == list(
        range(49, 49 + 3 * turns)
    )
    assert result["terminal"] == "completed"
    assert not result["actual_heading_known"] and not result["navigation_accepted"]
    assert all(np.array_equal(old, new) for old, new in zip(before, images))
    assert result["reference_rgb_sha256"] == [
        hashlib.sha256(image.tobytes()).hexdigest() for image in images
    ]


def test_no_eligible_view_abstains_before_any_look_or_move(scene):
    images, picked, calls, execute = scene
    picked["selected_frame"] = None
    outcome = orientation.orient(images, execute)
    assert outcome.journal["terminal"] == "no_eligible_view"
    assert calls == [] and outcome.journal["steps"] == []
    assert np.array_equal(outcome.frame, images[-1])


@pytest.mark.parametrize("failed_slot", [1, 2, 3, 7, 21])
def test_failed_receipt_preserves_exact_partial_prefix_and_stops(scene, failed_slot):
    images, _, calls, execute = scene

    def fail(kind):
        good = execute(kind)
        return orientation.SurveyReceipt(
            good.frame,
            good.executed_kind,
            good.executed_right_cell,
            len(calls) != failed_slot,
            1,
        )

    outcome = orientation.orient(images, fail)
    result = outcome.journal
    assert result["terminal"] == "execution_failed"
    assert len(calls) == len(result["steps"]) == failed_slot
    assert result["end_frame"] == 48 + failed_slot
    assert result["steps"][-1]["compliant"] is False
    assert not result["actual_heading_known"]


@pytest.mark.parametrize(
    "change", ["wrong_target", "wrong_kind", "ticks", "tick_bool", "exception", "shape"]
)
def test_invalid_dispatch_does_not_continue_or_invent_success(scene, change):
    images, _, calls, execute = scene

    def invalid(kind):
        good = execute(kind)
        if change == "exception":
            raise RuntimeError("injected dispatch failure")
        if change == "shape":
            return orientation.SurveyReceipt(
                good.frame[:224, :224].copy(), kind, True, True, 1
            )
        return orientation.SurveyReceipt(
            good.frame,
            "wait" if change == "wrong_kind" else kind,
            change != "wrong_target",
            True,
            True if change == "tick_bool" else 2 if change == "ticks" else 1,
        )

    outcome = orientation.orient(images, invalid)
    assert calls == ["look"] and outcome.journal["terminal"] == "execution_failed"
    assert outcome.journal["end_frame"] == (48 if change == "exception" else 49)
    assert not outcome.journal["navigation_accepted"]


@pytest.mark.parametrize("selected", [True, 33.0, 51, -3])
def test_selection_must_be_an_actual_original_ordinal_before_dispatch(scene, selected):
    images, picked, calls, execute = scene
    picked["selected_frame"] = selected
    with pytest.raises(ValueError, match="future or unregistered"):
        orientation.orient(images, execute)
    assert calls == []


@pytest.mark.parametrize(
    "values",
    [
        {"start_frame": 49},
        {"maximum_look_ticks": 12},
        {"waits_after_each_look": True},
        {"receipt_ticks": 1.0},
    ],
)
def test_orientation_profile_cannot_retime_or_extend_the_program(values):
    with pytest.raises(ValueError, match="exact registered clocks"):
        orientation.OrientationProfile(**values)

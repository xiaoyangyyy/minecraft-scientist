from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="optional public keyframe geometry")

from minecraft_scientist.exploration import stable_keyframe_map as stable
from minecraft_scientist.exploration.public_wait_profile import PostMotorWaitWindow
from minecraft_scientist.exploration.settled_map_control import PublicControlStep


def acquire(first, execute, *, long_wait=True, anchor=48, **kwargs):
    return stable.acquire_stable_keyframe_map(
        first,
        execute,
        ((100, 0, 112), (0, 100, 112), (0, 0, 1)),
        anchor_frame=anchor,
        micro_controls=True,
        responsive_controls=True,
        recovery_controls=True,
        safe_seed_controls=True,
        jump_controls=True,
        long_wait_controls=long_wait,
        **kwargs,
    )


def textured():
    gray = cv2.GaussianBlur(
        np.random.default_rng(29).integers(0, 256, (224, 224), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    return np.repeat(gray[:, :, None], 3, axis=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_wait_ticks": 25},
        {"maximum_wait_ticks": 24.0},
        {"minimum_wait_ticks": True},
        {"stable_prefixes": 2},
    ],
)
def test_post_motor_window_is_exact_frozen_and_not_a_geometry_policy(kwargs):
    with pytest.raises(ValueError, match="exact"):
        PostMotorWaitWindow(**kwargs)
    with pytest.raises(FrozenInstanceError):
        PostMotorWaitWindow().maximum_wait_ticks = 25
    assert not hasattr(PostMotorWaitWindow(), "maximum_p95_flow")


def test_untextured_initial_phase_still_uses_three_sixteen_tick_windows():
    first = np.zeros((224, 224, 3), dtype=np.uint8)
    calls = []

    def execute(kind, handle):
        calls.append((kind, handle))
        return PublicControlStep(first.copy(), kind, handle, True, 1)

    result = acquire(first, execute)
    assert result["terminal"] == "initial_wait_exhausted"
    assert calls == [("wait", None)] * 48
    assert len(result["failed_wait_windows"]) == 3
    assert len(result["recovery_wait_windows"]) == 2
    assert all(
        entry["registered_window_ticks"] == 16
        for name in ("failed_wait_windows", "recovery_wait_windows")
        for entry in result[name]
    )


def test_real_static_pixels_stop_at_eight_and_do_not_invent_motion_or_a_map():
    first, calls = textured(), []

    def execute(kind, handle):
        calls.append((kind, handle))
        return PublicControlStep(first.copy(), kind, handle, True, 1)

    result = acquire(first, execute)
    assert result["settling_window_registration"] == (
        "compound-post-motor-24-frame-windows-v1"
    )
    assert result["terminal"] == "map_unavailable"
    assert result["frozen"] is None and not result["future"] and not result["motion"]
    assert not result["navigation_accepted"]
    assert len(result["settles"]) == 12
    for entry in result["settles"]:
        assert entry["interval"][1] - entry["interval"][0] == 8
        assert entry["registered_window_ticks"] == (
            16 if entry["phase"] == "initial" else 24
        )
    assert not result["failed_wait_windows"] and not result["recovery_wait_windows"]
    moves = [handle for kind, handle in calls if kind == "move"]
    assert len(moves) == 11 and set(moves) == set(range(4))
    assert len(calls) + 48 <= 288


@pytest.mark.parametrize("long_wait", [False, True])
def test_real_moving_pixels_reach_same_first_stable_tail_with_different_cost_ledger(
    long_wait,
):
    first = textured()
    moved, waits = False, 0

    def execute(kind, handle):
        nonlocal moved, waits
        if kind == "move":
            if moved:
                raise RuntimeError("test stops after one completed moving trial")
            moved = True
        elif moved:
            waits += 1
        shift = min(waits, 17)
        current = cv2.warpAffine(
            first,
            np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32),
            (224, 224),
        )
        return PublicControlStep(current, kind, handle, True, 1)

    result = acquire(first, execute, long_wait=long_wait)
    assert result["terminal"] == "execution_failed"
    initial, post_motor = result["settles"]
    assert initial["interval"] == [48, 56]
    assert post_motor["interval"] == [57, 77]
    assert post_motor["stable_tail"] == [[74, 75], [75, 76], [76, 77]]
    assert len(result["recovery_wait_windows"]) == (0 if long_wait else 1)
    assert len(result["failed_wait_windows"]) == (0 if long_wait else 1)
    if long_wait:
        assert post_motor["registered_window_ticks"] == 24
    else:
        assert "settling_window_registration" not in result
        assert "registered_window_ticks" not in post_motor


def test_post_motor_exhaustion_keeps_all_three_twenty_four_tick_windows(monkeypatch):
    first, calls = textured(), []

    def execute(kind, handle):
        calls.append((kind, handle))
        return PublicControlStep(first.copy(), kind, handle, True, 1)

    original = stable._visible_flow

    def flow(*args, **kwargs):
        observed = original(*args, **kwargs)
        if any(kind == "move" for kind, _ in calls):
            observed["stable"] = False  # budget test, not native stability evidence
        return observed

    monkeypatch.setattr(stable, "_visible_flow", flow)
    result = acquire(first, execute)
    assert result["terminal"] == "acquisition_wait_exhausted"
    assert len(calls) == 8 + 1 + 72
    assert [entry["interval"] for entry in result["failed_wait_windows"]] == [
        [57, 81],
        [81, 105],
        [105, 129],
    ]
    assert len(result["recovery_wait_windows"]) == 2
    assert all(
        entry["registered_window_ticks"] == 24
        for name in ("failed_wait_windows", "recovery_wait_windows")
        for entry in result[name]
    )


def test_global_budget_stops_a_partial_post_motor_window_without_an_extra_action(
    monkeypatch,
):
    first, calls, post_motor_waits, moved = textured(), [], 0, False

    def execute(kind, handle):
        nonlocal post_motor_waits, moved
        calls.append((kind, handle))
        if kind == "move":
            moved, post_motor_waits = True, 0
        elif moved:
            post_motor_waits += 1
        return PublicControlStep(first.copy(), kind, handle, True, 1)

    original = stable._visible_flow

    def flow(*args, **kwargs):
        observed = original(*args, **kwargs)
        if moved:
            observed["stable"] = post_motor_waits >= 22
        return observed

    monkeypatch.setattr(stable, "_visible_flow", flow)
    result = acquire(first, execute)
    assert result["terminal"] == "budget_exhausted"
    assert len(calls) + 48 == 288
    assert len([kind for kind, _ in calls if kind == "move"]) == 10
    assert post_motor_waits == 6
    assert not result["recovery_wait_windows"]


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_untyped_long_wait_flag_rejects_before_dispatch(flag):
    def forbidden(*args):
        raise AssertionError("invalid protocol dispatched a control")

    with pytest.raises(ValueError, match="explicit boolean"):
        stable.acquire_stable_keyframe_map(
            textured(), forbidden, np.eye(3), anchor_frame=48, long_wait_controls=flag
        )


def test_long_wait_registration_requires_compound_protocol_before_dispatch():
    def forbidden(*args):
        raise AssertionError("invalid protocol dispatched a control")

    with pytest.raises(ValueError, match="explicit compound"):
        stable.acquire_stable_keyframe_map(
            textured(), forbidden, np.eye(3), anchor_frame=48, long_wait_controls=True
        )


@pytest.mark.parametrize(
    "field,value",
    [("post_motor_wait_ticks", 25), ("settling_profile", "private-pose-stop")],
)
def test_new_policy_cannot_relax_budgets_or_disguise_itself_as_parent(field, value):
    settings = stable.StableLongWaitJumpKeyframePolicy().model_dump(mode="json")
    with pytest.raises(ValidationError):
        stable.StableJumpSafeSeedKeyframePolicy.model_validate(settings)
    with pytest.raises(ValidationError):
        stable.StableLongWaitJumpKeyframePolicy.model_validate(
            settings | {field: value}
        )
    new, old = (
        stable.StableLongWaitJumpKeyframePolicy(),
        stable.StableJumpSafeSeedKeyframePolicy(),
    )
    for name in ("stability", "fitting", "consensus", "temporal"):
        assert getattr(new, name) == getattr(old, name)

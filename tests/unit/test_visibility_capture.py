import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PIL.Image", reason="optional RGB acquisition wrapper")

from minecraft_scientist.exploration.compound_motion import CompoundProbeConfig
from minecraft_scientist.exploration.visibility_acquisition import VisibilityPolicy
from minecraft_scientist.protocols import ActionKind


@pytest.fixture
def capture(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "environments" / "minestudio"))
    module = importlib.import_module("capture_visibility_rgb")
    monkeypatch.setattr(
        module,
        "calibrate_reference_camera",
        lambda images, size: {
            "camera_intrinsics": [[180, 0, 112], [0, 180, 112], [0, 0, 1]],
        },
    )
    return module


class PublicWorld:
    def __init__(self, frames):
        self.frames = frames
        self.actions = []

    def reset(self, seed):
        self.seed = seed
        self.frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    def observe(self):
        objects = [
            SimpleNamespace(object_id=f"opaque-{index}", affordances=(ActionKind.MOVE,))
            for index in range(4)
        ]
        objects.append(
            SimpleNamespace(
                object_id="anonymous-view",
                affordances=(ActionKind.LOOK,),
                observable_features={"screen_row": 1, "screen_column": 2},
            )
        )
        return SimpleNamespace(objects=objects)

    def step(self, action):
        self.actions.append(action)
        self.frames.extend(
            np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(action.ticks)
        )
        return SimpleNamespace(
            executed_action=action,
            intervention_compliance=True,
            event=SimpleNamespace(duration=action.ticks, success=True),
        )

    @property
    def positions(self):
        pytest.fail("capture policy inspected evaluator positions")


def test_acquisition_orchestration_exposes_only_rgb_and_receipts(capture, monkeypatch):
    frames, records, trials, receipts = [], [], [], []
    world = PublicWorld(frames)

    def acquire(initial, execute, pair, *, policy, landmark_observer):
        assert isinstance(initial, np.ndarray)
        assert policy.landmark_protection == "first_supported_map_v1"
        assert callable(landmark_observer)
        for handle in pair:
            receipt = execute(handle)
            assert (
                receipt.executed_handle == handle
                and receipt.compliant
                and receipt.ticks == 1
            )
            receipts.append(receipt)
        return SimpleNamespace(
            terminal="finished",
            failure=None,
            steps=(),
            decisions=(),
            tracks=(),
            landmark_attempts=(),
        )

    monkeypatch.setattr(capture, "acquire_visibility_probe", acquire)
    config = CompoundProbeConfig(seed=201)
    references, calibration = capture.collect(
        world, frames, config, capture.online_policy(), records, trials
    )
    assert world.seed == 201 and len(trials) == 8 and len(receipts) == 16
    assert len(references) == 12 and references[-1] == 56
    assert calibration["camera_intrinsics"][0][0] == 180
    assert [trial["landmark_seed"] for trial in trials] == list(range(2, 38, 5))
    moves = [action for action in world.actions if action.kind == ActionKind.MOVE]
    assert len(moves) == 16 and all(action.ticks == 1 for action in moves)
    assert len(frames) == sum(item.event.duration for item in records) + 1


def test_probe_failure_retains_trial_and_stops_before_wait_or_retry(
    capture, monkeypatch
):
    frames, records, trials = [], [], []
    world = PublicWorld(frames)
    monkeypatch.setattr(
        capture,
        "acquire_visibility_probe",
        lambda *a, **k: SimpleNamespace(
            terminal="control_failed",
            failure={"message": "lost control"},
            steps=(),
            decisions=(),
            tracks=(),
            landmark_attempts=(),
        ),
    )
    with pytest.raises(RuntimeError, match="visibility probe failed"):
        capture.collect(
            world, frames, CompoundProbeConfig(), VisibilityPolicy(), records, trials
        )
    assert len(trials) == 1 and trials[0]["terminal"] == "control_failed"
    assert len(records) == 25 and not any(
        action.kind == ActionKind.MOVE for action in world.actions
    )


def test_second_leg_shadow_fits_only_completed_leg_before_new_controls(
    capture, monkeypatch
):
    frames, records, trials, seeds = [], [], [], []
    world = PublicWorld(frames)

    def acquire(initial, execute, pair, **kwargs):
        for handle in pair:
            execute(handle)
            execute(handle)
        return SimpleNamespace(
            terminal="finished",
            failure=None,
            steps=(),
            tracks=(),
            landmark_attempts=(),
            decisions=(
                SimpleNamespace(
                    instruction="switch",
                    tick=2,
                    model_dump=lambda **kwargs: {"instruction": "switch", "tick": 2},
                ),
                SimpleNamespace(
                    instruction="finish",
                    tick=4,
                    model_dump=lambda **kwargs: {"instruction": "finish", "tick": 4},
                ),
            ),
        )

    def shadow(leg, execute, handles, camera, *, seed, handle_count, policy):
        assert len(leg) == 3 and handle_count == 4
        assert all(isinstance(image, np.ndarray) for image in leg)
        seeds.append(seed)
        handle = next(index for index in range(handle_count) if index not in handles)
        for _ in range(policy.ticks):
            assert execute(handle).compliant
        return {"terminal": "completed", "failure": None, "handle": handle}

    monkeypatch.setattr(capture, "acquire_visibility_probe", acquire)
    monkeypatch.setattr(capture, "acquire_second_leg_shadow", shadow)
    capture.collect(
        world,
        frames,
        CompoundProbeConfig(seed=201),
        capture.online_policy(),
        records,
        trials,
        shadow_policy=capture.SecondLegShadowPolicy(),
    )
    assert len(trials) == 8 and seeds == list(range(1002, 1038, 5))
    assert all(
        trial["second_leg_shadow"]["anchor_frame"] == trial["anchor_frame"] + 2
        for trial in trials
    )
    assert all(
        trial["second_leg_shadow"]["freeze_frame"] == trial["end_frame"]
        for trial in trials
    )
    assert all(
        trial["second_leg_shadow"]["end_frame"] == trial["end_frame"] + 4
        for trial in trials
    )
    assert len(frames) == 193

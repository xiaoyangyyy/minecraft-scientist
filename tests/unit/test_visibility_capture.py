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
    return importlib.import_module("capture_visibility_rgb")


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

    def acquire(initial, execute, pair, *, policy):
        assert isinstance(initial, np.ndarray)
        for handle in pair:
            receipt = execute(handle)
            assert (
                receipt.executed_handle == handle
                and receipt.compliant
                and receipt.ticks == 1
            )
            receipts.append(receipt)
        return SimpleNamespace(
            terminal="finished", failure=None, steps=(), decisions=(), tracks=()
        )

    monkeypatch.setattr(capture, "acquire_visibility_probe", acquire)
    config = CompoundProbeConfig(seed=201)
    references = capture.collect(
        world, frames, config, VisibilityPolicy(), records, trials
    )
    assert world.seed == 201 and len(trials) == 8 and len(receipts) == 16
    assert len(references) == 12 and references[-1] == 56
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

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.protocols import RunConfig
from minecraft_scientist.training import (
    ToolDiscoveryEnv,
    TrainingConfig,
    evaluate_checkpoint,
    train_policy,
)
from minecraft_scientist.training.trainer import generate_curriculum


def test_gym_environment_contract() -> None:
    check_env(ToolDiscoveryEnv(), skip_render_check=True)


def test_training_checkpoint_and_learned_planner_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "planner.pt"
    metrics = train_policy(
        TrainingConfig(
            seed=17,
            worlds=64,
            validation_worlds=32,
            epochs=4,
            batch_size=32,
            device="cpu",
        ),
        checkpoint,
    )
    assert checkpoint.exists()
    assert checkpoint.with_suffix(".json").exists()
    assert metrics["success_rate"] == 1.0
    assert metrics["fall"]["success_rate"] == 1.0
    assert metrics["machine"]["success_rate"] == 1.0
    assert metrics["transformation"]["success_rate"] == 1.0
    assert metrics["risk_aware"]["optimal_action_rate"] > 0.9
    assert (
        metrics["risk_aware"]["optimal_action_rate"]
        > metrics["risk_aware"]["entropy_only_optimal_action_rate"]
    )
    evaluated = evaluate_checkpoint(
        checkpoint,
        worlds=16,
        seed=99,
        device="cpu",
    )
    assert evaluated["success_rate"] == 1.0
    assert evaluated["fall"]["success_rate"] == 1.0
    assert evaluated["machine"]["success_rate"] == 1.0
    assert evaluated["transformation"]["success_rate"] == 1.0

    run = ScientistAgent(
        RunConfig(
            seed=3,
            planner="learned",
            planner_checkpoint=str(checkpoint),
        )
    ).run()
    assert run.report.hidden_intervention_accuracy == 1.0


def test_four_form_curriculum_is_seed_deterministic() -> None:
    first = generate_curriculum(
        24,
        seed=1234,
        exploration_rate=0.2,
        fall_fraction=0.25,
        machine_fraction=0.25,
        transformation_fraction=0.25,
    )
    second = generate_curriculum(
        24,
        seed=1234,
        exploration_rate=0.2,
        fall_fraction=0.25,
        machine_fraction=0.25,
        transformation_fraction=0.25,
    )
    assert all(np.array_equal(left, right) for left, right in zip(first, second))

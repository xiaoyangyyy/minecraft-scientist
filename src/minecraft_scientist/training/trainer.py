"""Supervised curriculum that distills an information-gain oracle."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minecraft_scientist.discovery import (
    MachineVersionSpace,
    ToolVersionSpace,
    TransformationLearner,
)
from minecraft_scientist.planning import experiment_utility
from minecraft_scientist.symbolic import FallEquationVersionSpace

from .policy import ExperimentPolicy, load_policy_checkpoint, save_policy_checkpoint

TOOLS = ("tool_0", "tool_1", "tool_2")
TARGETS = ("target_0", "target_1", "target_2")
ARMOR_IDS = ("armor_0", "armor_1", "armor_2")
HEIGHTS = tuple(range(1, 11))
MACHINE_RESOURCES = ("resource_0", "resource_1", "resource_2")
PROCESSORS = ("processor_0", "processor_1", "processor_2")
TRANSFORM_RESOURCES = ("resource_0", "resource_1", "resource_2", "resource_3")
MAX_ACTIONS = 30


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 0
    worlds: int = 5_000
    validation_worlds: int = 500
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3e-4
    hidden_size: int = 64
    exploration_rate: float = 0.2
    fall_fraction: float = 0.25
    machine_fraction: float = 0.20
    transformation_fraction: float = 0.20
    device: str = "auto"

    def __post_init__(self) -> None:
        fractions = (
            self.fall_fraction,
            self.machine_fraction,
            self.transformation_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("curriculum fractions must be between zero and one")
        if sum(fractions) > 1.0:
            raise ValueError("curriculum fractions must sum to at most one")


def _target_distribution(features: np.ndarray) -> np.ndarray:
    untested = features[:, 2] == 0.0
    target = np.zeros(features.shape[0], dtype=np.float32)
    if not untested.any():
        return target
    utilities = np.asarray(
        [experiment_utility(row) for row in features], dtype=np.float32
    )
    best = utilities[untested].max()
    optimal = untested & np.isclose(utilities, best, atol=1e-7)
    target[optimal] = 1.0 / optimal.sum()
    return target


def generate_curriculum(
    worlds: int,
    *,
    seed: int,
    exploration_rate: float,
    fall_fraction: float = 0.25,
    machine_fraction: float = 0.20,
    transformation_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fall_fraction + machine_fraction + transformation_fraction > 1.0:
        raise ValueError("curriculum fractions must sum to at most one")
    rng = random.Random(seed)
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for world_index in range(worlds):
        draw = rng.random()
        if draw < fall_fraction:
            kind = "fall"
            space = FallEquationVersionSpace(ARMOR_IDS, HEIGHTS)
        elif draw < fall_fraction + machine_fraction:
            kind = "machine"
            space = MachineVersionSpace(MACHINE_RESOURCES)
        elif draw < fall_fraction + machine_fraction + transformation_fraction:
            kind = "transformation"
            space = TransformationLearner(PROCESSORS, TRANSFORM_RESOURCES)
        else:
            kind = "tool"
            space = ToolVersionSpace(TOOLS, TARGETS)
        truth = rng.choice(
            space.grounded_candidates()
            if kind == "transformation"
            else space.candidates
        )
        initial_pairs, _ = space.experiment_features()
        risk_profile = np.asarray(
            [
                rng.choice((0.10, 0.20, 0.35, 0.50))
                if rng.random() < 0.30
                else 0.0
                for _ in initial_pairs
            ],
            dtype=np.float32,
        )
        step = 0
        while not space.resolved():
            pairs, raw_features = space.experiment_features()
            features = np.asarray(raw_features, dtype=np.float32)
            features[:, 5] = risk_profile
            mask = features[:, 2] == 0.0
            distribution = _target_distribution(features)
            padded_features, padded_mask, padded_distribution = _pad_state(
                features, mask, distribution
            )
            states.append(padded_features)
            masks.append(padded_mask)
            targets.append(padded_distribution)
            optimal = np.flatnonzero(distribution > 0)
            untested = np.flatnonzero(mask)
            if rng.random() < exploration_rate:
                action = int(rng.choice(untested))
            else:
                action = int(rng.choice(optimal))
            first, second = pairs[action]
            step += 1
            evidence_id = f"train-{world_index:06d}-{step:02d}"
            _observe_candidate_space(
                kind,
                space,
                truth,
                first,
                second,
                evidence_id,
            )
    return (
        np.stack(states),
        np.stack(masks),
        np.stack(targets),
    )


def _observe_candidate_space(
    kind: str,
    space: Any,
    truth: Any,
    first: Any,
    second: Any,
    evidence_id: str,
) -> None:
    if kind == "fall":
        space.observe(first, second, truth.predict(first, second), evidence_id)
    elif kind == "tool":
        space.observe(first, second, truth.predicts(first, second), evidence_id)
    elif kind == "machine":
        success = truth.predicts(first, second)
        space.observe(
            first,
            second,
            success,
            evidence_id,
            "machine_output" if success else None,
        )
    elif kind == "transformation":
        output = truth.predict(first, second)
        space.observe(first, second, output is not None, output, evidence_id)
    else:
        raise ValueError(f"unknown curriculum mechanism: {kind}")


def _pad_state(
    features: np.ndarray,
    mask: np.ndarray,
    distribution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(features) > MAX_ACTIONS:
        raise ValueError("curriculum state exceeds policy action capacity")
    padded_features = np.zeros((MAX_ACTIONS, features.shape[1]), dtype=np.float32)
    padded_mask = np.zeros(MAX_ACTIONS, dtype=bool)
    padded_distribution = np.zeros(MAX_ACTIONS, dtype=np.float32)
    padded_features[: len(features)] = features
    padded_mask[: len(features)] = mask
    padded_distribution[: len(features)] = distribution
    return padded_features, padded_mask, padded_distribution


def _align_model_features(
    features: np.ndarray, model: ExperimentPolicy
) -> np.ndarray:
    if features.shape[1] > model.feature_dim:
        return features[:, : model.feature_dim]
    if features.shape[1] < model.feature_dim:
        return np.pad(
            features,
            ((0, 0), (0, model.feature_dim - features.shape[1])),
        )
    return features


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train_policy(config: TrainingConfig, output: Path) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _select_device(config.device)
    states, masks, targets = generate_curriculum(
        config.worlds,
        seed=config.seed,
        exploration_rate=config.exploration_rate,
        fall_fraction=config.fall_fraction,
        machine_fraction=config.machine_fraction,
        transformation_fraction=config.transformation_fraction,
    )
    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(masks),
        torch.from_numpy(targets),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    model = ExperimentPolicy(hidden_size=config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        total = 0.0
        count = 0
        for features, action_mask, target_probs in loader:
            features = features.to(device, non_blocking=True)
            action_mask = action_mask.to(device, non_blocking=True)
            target_probs = target_probs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, action_mask)
            loss = (
                -(target_probs * nn.functional.log_softmax(logits, dim=-1))
                .sum(dim=-1)
                .mean()
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.detach()) * features.shape[0]
            count += features.shape[0]
        losses.append(total / count)
    metrics = evaluate_policy(
        model,
        worlds=config.validation_worlds,
        seed=config.seed + 1_000_000,
        device=device,
    )
    metrics["fall"] = evaluate_fall_policy(
        model,
        worlds=config.validation_worlds,
        seed=config.seed + 1_500_000,
        device=device,
    )
    metrics["machine"] = evaluate_machine_policy(
        model,
        worlds=config.validation_worlds,
        seed=config.seed + 1_750_000,
        device=device,
    )
    metrics["transformation"] = evaluate_transformation_policy(
        model,
        worlds=config.validation_worlds,
        seed=config.seed + 2_000_000,
        device=device,
    )
    metrics["risk_aware"] = evaluate_curriculum_policy(
        model,
        worlds=config.validation_worlds,
        seed=config.seed + 2_500_000,
        device=device,
    )
    metrics.update(
        {
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "training_examples": len(dataset),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        }
    )
    save_policy_checkpoint(
        output,
        model,
        training_config=asdict(config),
        metrics=metrics,
    )
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {"training_config": asdict(config), "losses": losses, "metrics": metrics},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def evaluate_policy(
    model: ExperimentPolicy,
    *,
    worlds: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    learned_steps: list[int] = []
    oracle_steps: list[int] = []
    random_steps: list[int] = []
    optimal_actions = 0
    total_actions = 0
    successes = 0
    model.eval()
    for world_index in range(worlds):
        initial = ToolVersionSpace(TOOLS, TARGETS)
        truth = rng.choice(initial.candidates)
        learned = ToolVersionSpace(TOOLS, TARGETS)
        steps = 0
        while not learned.resolved() and steps < 9:
            pairs, raw_features = learned.experiment_features()
            features = np.asarray(raw_features, dtype=np.float32)
            mask = features[:, 2] == 0.0
            targets = _target_distribution(features)
            model_features = _align_model_features(features, model)
            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(model_features).to(device).unsqueeze(0),
                    torch.from_numpy(mask).to(device).unsqueeze(0),
                )
                action = int(logits.argmax(dim=-1).item())
            optimal_actions += int(targets[action] > 0)
            total_actions += 1
            tool, target = pairs[action]
            steps += 1
            learned.observe(
                tool,
                target,
                truth.predicts(tool, target),
                f"eval-{world_index:05d}-{steps:02d}",
            )
        learned_steps.append(steps)
        successes += int(learned.resolved())
        oracle_steps.append(_rollout_steps(truth, rng=None))
        random_steps.append(_rollout_steps(truth, rng=rng))
    return {
        "success_rate": successes / worlds,
        "optimal_action_rate": optimal_actions / max(total_actions, 1),
        "mean_experiments": float(np.mean(learned_steps)),
        "oracle_mean_experiments": float(np.mean(oracle_steps)),
        "random_mean_experiments": float(np.mean(random_steps)),
    }


def _rollout_steps(truth: Any, rng: random.Random | None) -> int:
    space = ToolVersionSpace(TOOLS, TARGETS)
    steps = 0
    while not space.resolved() and steps < 9:
        pairs, features = space.experiment_features()
        if rng is None:
            pair = space.next_experiment()
            assert pair is not None
        else:
            untested = [
                pair for index, pair in enumerate(pairs) if features[index][2] == 0.0
            ]
            pair = rng.choice(untested)
        steps += 1
        space.observe(
            pair[0],
            pair[1],
            truth.predicts(pair[0], pair[1]),
            f"rollout-{steps:02d}",
        )
    return steps


def evaluate_fall_policy(
    model: ExperimentPolicy,
    *,
    worlds: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    learned_steps: list[int] = []
    oracle_steps: list[int] = []
    random_steps: list[int] = []
    optimal_actions = 0
    total_actions = 0
    successes = 0
    for world_index in range(worlds):
        initial = FallEquationVersionSpace(ARMOR_IDS, HEIGHTS)
        truth = rng.choice(initial.candidates)
        learned = FallEquationVersionSpace(ARMOR_IDS, HEIGHTS)
        steps = 0
        while not learned.resolved() and steps < MAX_ACTIONS:
            pairs, raw_features = learned.experiment_features()
            features = np.asarray(raw_features, dtype=np.float32)
            mask = features[:, 2] == 0.0
            targets = _target_distribution(features)
            model_features = _align_model_features(features, model)
            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(model_features).to(device).unsqueeze(0),
                    torch.from_numpy(mask).to(device).unsqueeze(0),
                )
                action = int(logits.argmax(dim=-1).item())
            optimal_actions += int(targets[action] > 0)
            total_actions += 1
            armor_id, height = pairs[action]
            steps += 1
            learned.observe(
                armor_id,
                height,
                truth.predict(armor_id, height),
                f"fall-eval-{world_index:05d}-{steps:02d}",
            )
        learned_steps.append(steps)
        successes += int(learned.resolved())
        oracle_steps.append(_fall_rollout_steps(truth, rng=None))
        random_steps.append(_fall_rollout_steps(truth, rng=rng))
    return {
        "success_rate": successes / worlds,
        "optimal_action_rate": optimal_actions / max(total_actions, 1),
        "mean_experiments": float(np.mean(learned_steps)),
        "oracle_mean_experiments": float(np.mean(oracle_steps)),
        "random_mean_experiments": float(np.mean(random_steps)),
    }


def _fall_rollout_steps(truth: Any, rng: random.Random | None) -> int:
    space = FallEquationVersionSpace(ARMOR_IDS, HEIGHTS)
    steps = 0
    while not space.resolved() and steps < MAX_ACTIONS:
        pairs, features = space.experiment_features()
        if rng is None:
            pair = space.next_experiment()
            assert pair is not None
        else:
            untested = [
                pair for index, pair in enumerate(pairs) if features[index][2] == 0.0
            ]
            pair = rng.choice(untested)
        steps += 1
        space.observe(
            pair[0],
            pair[1],
            truth.predict(pair[0], pair[1]),
            f"fall-rollout-{steps:02d}",
        )
    return steps


def evaluate_machine_policy(
    model: ExperimentPolicy,
    *,
    worlds: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    return _evaluate_finite_policy(
        model,
        kind="machine",
        factory=lambda: MachineVersionSpace(MACHINE_RESOURCES),
        worlds=worlds,
        seed=seed,
        device=device,
        max_steps=len(MACHINE_RESOURCES) * 5,
    )


def evaluate_transformation_policy(
    model: ExperimentPolicy,
    *,
    worlds: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    return _evaluate_finite_policy(
        model,
        kind="transformation",
        factory=lambda: TransformationLearner(PROCESSORS, TRANSFORM_RESOURCES),
        worlds=worlds,
        seed=seed,
        device=device,
        max_steps=len(PROCESSORS) * len(TRANSFORM_RESOURCES),
    )


def _evaluate_finite_policy(
    model: ExperimentPolicy,
    *,
    kind: str,
    factory: Any,
    worlds: int,
    seed: int,
    device: torch.device,
    max_steps: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    learned_steps: list[int] = []
    oracle_steps: list[int] = []
    random_steps: list[int] = []
    optimal_actions = 0
    total_actions = 0
    successes = 0
    model.eval()
    for world_index in range(worlds):
        initial = factory()
        truth = rng.choice(
            initial.grounded_candidates()
            if kind == "transformation"
            else initial.candidates
        )
        learned = factory()
        steps = 0
        while not learned.resolved() and steps < max_steps:
            pairs, raw_features = learned.experiment_features()
            features = np.asarray(raw_features, dtype=np.float32)
            mask = features[:, 2] == 0.0
            targets = _target_distribution(features)
            model_features = _align_model_features(features, model)
            with torch.inference_mode():
                logits = model(
                    torch.from_numpy(model_features).to(device).unsqueeze(0),
                    torch.from_numpy(mask).to(device).unsqueeze(0),
                )
                action = int(logits.argmax(dim=-1).item())
            optimal_actions += int(targets[action] > 0)
            total_actions += 1
            first, second = pairs[action]
            steps += 1
            _observe_candidate_space(
                kind,
                learned,
                truth,
                first,
                second,
                f"{kind}-eval-{world_index:05d}-{steps:02d}",
            )
        learned_steps.append(steps)
        successes += int(learned.resolved())
        oracle_steps.append(
            _finite_rollout_steps(kind, factory, truth, rng=None, max_steps=max_steps)
        )
        random_steps.append(
            _finite_rollout_steps(kind, factory, truth, rng=rng, max_steps=max_steps)
        )
    return {
        "success_rate": successes / worlds,
        "optimal_action_rate": optimal_actions / max(total_actions, 1),
        "mean_experiments": float(np.mean(learned_steps)),
        "oracle_mean_experiments": float(np.mean(oracle_steps)),
        "random_mean_experiments": float(np.mean(random_steps)),
    }


def _finite_rollout_steps(
    kind: str,
    factory: Any,
    truth: Any,
    *,
    rng: random.Random | None,
    max_steps: int,
) -> int:
    space = factory()
    steps = 0
    while not space.resolved() and steps < max_steps:
        pairs, features = space.experiment_features()
        if rng is None:
            pair = space.next_experiment()
            assert pair is not None
        else:
            untested = [
                pair for index, pair in enumerate(pairs) if features[index][2] == 0.0
            ]
            pair = rng.choice(untested)
        steps += 1
        _observe_candidate_space(
            kind,
            space,
            truth,
            pair[0],
            pair[1],
            f"{kind}-rollout-{steps:02d}",
        )
    return steps


def evaluate_curriculum_policy(
    model: ExperimentPolicy,
    *,
    worlds: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    states, masks, targets = generate_curriculum(
        worlds,
        seed=seed,
        exploration_rate=0.2,
    )
    aligned = np.stack(
        [_align_model_features(features, model) for features in states]
    )
    choices: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(aligned), 1024):
            stop = start + 1024
            logits = model(
                torch.from_numpy(aligned[start:stop]).to(device),
                torch.from_numpy(masks[start:stop]).to(device),
            )
            choices.append(logits.argmax(dim=-1).cpu().numpy())
    selected = np.concatenate(choices)
    rows = np.arange(len(selected))
    learned_optimal = targets[rows, selected] > 0.0
    entropy = np.where(masks, states[:, :, 1], -np.inf)
    entropy_only = entropy.argmax(axis=1)
    entropy_optimal = targets[rows, entropy_only] > 0.0
    nonzero_risk = int(np.count_nonzero(states[:, :, 5] > 0.0))
    return {
        "states": len(states),
        "nonzero_risk_actions": nonzero_risk,
        "optimal_action_rate": float(np.mean(learned_optimal)),
        "entropy_only_optimal_action_rate": float(np.mean(entropy_optimal)),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    worlds: int,
    seed: int,
    device: str = "auto",
) -> dict[str, Any]:
    selected = _select_device(device)
    model, payload = load_policy_checkpoint(checkpoint, device=selected)
    metrics: dict[str, Any] = evaluate_policy(
        model,
        worlds=worlds,
        seed=seed,
        device=selected,
    )
    metrics["fall"] = evaluate_fall_policy(
        model,
        worlds=worlds,
        seed=seed + 500_000,
        device=selected,
    )
    metrics["machine"] = evaluate_machine_policy(
        model,
        worlds=worlds,
        seed=seed + 750_000,
        device=selected,
    )
    metrics["transformation"] = evaluate_transformation_policy(
        model,
        worlds=worlds,
        seed=seed + 1_000_000,
        device=selected,
    )
    metrics["risk_aware"] = evaluate_curriculum_policy(
        model,
        worlds=worlds,
        seed=seed + 1_250_000,
        device=selected,
    )
    metrics["device"] = str(selected)
    metrics["training_metrics"] = payload.get("metrics", {})
    return metrics

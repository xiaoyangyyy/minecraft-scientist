"""Gymnasium-compatible active discovery environment."""

from __future__ import annotations

import random
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from minecraft_scientist.discovery import ToolVersionSpace


class ToolDiscoveryEnv(gym.Env[dict[str, np.ndarray], int]):
    """Learn to choose interventions that collapse a symbolic version space."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.tools = ("tool_0", "tool_1", "tool_2")
        self.targets = ("target_0", "target_1", "target_2")
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(0.0, 1.0, shape=(9, 6), dtype=np.float32),
                "action_mask": spaces.MultiBinary(9),
            }
        )
        self._rng = random.Random(0)
        self._space = ToolVersionSpace(self.tools, self.targets)
        self._truth = self._space.candidates[0]
        self._steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._rng.seed(seed)
        self._space = ToolVersionSpace(self.tools, self.targets)
        self._truth = self._rng.choice(self._space.candidates)
        self._steps = 0
        return self._observation(), {"candidate_count": len(self._space.candidates)}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        pairs, features = self._space.experiment_features()
        if action < 0 or action >= len(pairs) or features[action][2] == 1.0:
            return self._observation(), -1.0, False, False, {"invalid_action": True}
        before = len(self._space.candidates)
        tool, target = pairs[action]
        outcome = self._truth.predicts(tool, target)
        self._steps += 1
        self._space.observe(tool, target, outcome, f"gym-{self._steps:03d}")
        after = len(self._space.candidates)
        terminated = self._space.resolved()
        reward = (before - after) / before + (1.0 if terminated else -0.01)
        return (
            self._observation(),
            float(reward),
            terminated,
            False,
            {"candidate_count": after, "outcome": outcome},
        )

    def _observation(self) -> dict[str, np.ndarray]:
        _, features = self._space.experiment_features()
        array = np.asarray(features, dtype=np.float32)
        return {
            "features": array,
            "action_mask": (array[:, 2] == 0.0).astype(np.int8),
        }

"""Adapter that applies a trained policy to the symbolic version space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from minecraft_scientist.discovery import (
    MachineVersionSpace,
    ToolVersionSpace,
    TransformationLearner,
)
from minecraft_scientist.planning.selection import experiment_utility
from minecraft_scientist.symbolic import FallEquationVersionSpace

from .policy import load_policy_checkpoint


class NeuralExperimentPlanner:
    def __init__(self, checkpoint: Path, *, device: str = "auto") -> None:
        selected = (
            torch.device("cuda")
            if device == "auto" and torch.cuda.is_available()
            else torch.device("cpu" if device == "auto" else device)
        )
        self.device = selected
        self.model, self.metadata = load_policy_checkpoint(checkpoint, device=selected)

    def select_tool_experiment(self, space: ToolVersionSpace) -> tuple[str, str] | None:
        return self._select(space)

    def select_fall_experiment(
        self, space: FallEquationVersionSpace
    ) -> tuple[str, int] | None:
        return self._select(space)

    def select_machine_experiment(
        self, space: MachineVersionSpace
    ) -> tuple[str, tuple[str, ...]] | None:
        return self._select(space)

    def select_transformation_experiment(
        self, space: TransformationLearner
    ) -> tuple[str, str] | None:
        return self._select(space)

    def _select(self, space):
        pairs, features = space.experiment_features()
        array = np.asarray(features, dtype=np.float32)
        mask = array[:, 2] == 0.0
        if not mask.any():
            return None
        utilities = np.asarray(
            [experiment_utility(row) for row in features], dtype=np.float32
        )
        best_utility = utilities[mask].max()
        constrained_mask = mask & np.isclose(
            utilities, best_utility, atol=1e-7
        )
        if array.shape[1] > self.model.feature_dim:
            array = array[:, : self.model.feature_dim]
        elif array.shape[1] < self.model.feature_dim:
            array = np.pad(
                array,
                ((0, 0), (0, self.model.feature_dim - array.shape[1])),
            )
        with torch.inference_mode():
            tensor = torch.from_numpy(array).to(self.device).unsqueeze(0)
            action_mask = (
                torch.from_numpy(constrained_mask).to(self.device).unsqueeze(0)
            )
            logits = self.model(tensor, action_mask)
            index = int(logits.argmax(dim=-1).item())
        return pairs[index]

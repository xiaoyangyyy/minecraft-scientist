"""Permutation-equivariant neural experiment policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class ExperimentPolicy(nn.Module):
    def __init__(self, feature_dim: int = 6, hidden_size: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor, action_mask: Tensor | None = None) -> Tensor:
        encoded = self.encoder(features)
        if action_mask is None:
            pooled_mean = encoded.mean(dim=-2, keepdim=True)
            pooled_max = encoded.amax(dim=-2, keepdim=True)
        else:
            valid = action_mask.bool().unsqueeze(-1)
            count = valid.sum(dim=-2, keepdim=True).clamp_min(1)
            pooled_mean = (encoded * valid).sum(dim=-2, keepdim=True) / count
            pooled_max = encoded.masked_fill(~valid, float("-inf")).amax(
                dim=-2, keepdim=True
            )
        pooled_mean = pooled_mean.expand_as(encoded)
        pooled_max = pooled_max.expand_as(encoded)
        logits = self.scorer(
            torch.cat((encoded, pooled_mean, pooled_max), dim=-1)
        ).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(
                ~action_mask.bool(), torch.finfo(logits.dtype).min
            )
        return logits


def save_policy_checkpoint(
    path: Path,
    model: ExperimentPolicy,
    *,
    training_config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "feature_schema": [
                "dominant_outcome_probability",
                "normalized_predictive_entropy",
                "already_tested",
                "posterior_fraction",
                "normalized_experiment_cost",
                "control_failure_risk",
            ],
            "model_config": {
                "feature_dim": model.feature_dim,
                "hidden_size": model.hidden_size,
            },
            "state_dict": model.state_dict(),
            "training_config": training_config,
            "metrics": metrics,
        },
        path,
    )


def load_policy_checkpoint(
    path: Path, *, device: torch.device | str = "cpu"
) -> tuple[ExperimentPolicy, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") not in {1, 2, 3}:
        raise ValueError("unsupported planner checkpoint format")
    model = ExperimentPolicy(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload

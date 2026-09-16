"""Serializable contracts for local interventional causal discovery."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CausalSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    context: dict[str, str | int | float | bool] = Field(default_factory=dict)
    intervention: dict[str, str | int | float | bool]
    outcome: dict[str, str | int | float | bool]
    compliant: bool = True
    possible_confounders: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)


class ControlledEffect(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    treatment: str
    outcome: str
    fixed_context: dict[str, str | int | float | bool]
    group_means: dict[str, float]
    absolute_effect: float = Field(ge=0)
    sample_count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)

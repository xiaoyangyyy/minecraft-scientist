"""Public, serializable contracts shared by the simulator and scientist."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionKind(StrEnum):
    SELECT = "select"
    INTERACT = "interact"
    PLACE_INPUT = "place_input"
    ACTIVATE = "activate"
    WAIT = "wait"


class TheoryRepresentation(StrEnum):
    EQUATION = "equation"
    RULE = "rule"
    STATE_MACHINE = "state_machine"
    PROGRAM = "program"


class TheoryStatus(StrEnum):
    PROPOSED = "proposed"
    PROVISIONAL = "provisional"
    ACCEPTED = "accepted"
    CHALLENGED = "challenged"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class WorldEvent(FrozenModel):
    event_type: str
    target_id: str | None = None
    success: bool = True
    duration: int = Field(default=1, ge=0)
    output_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ObjectObservation(FrozenModel):
    object_id: str
    affordances: tuple[ActionKind, ...]
    observable_features: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )


class Observation(FrozenModel):
    episode_id: int = Field(ge=0)
    step: int = Field(ge=0)
    inventory: dict[str, int]
    selected_id: str | None = None
    agent_state: dict[str, str | int | float | bool] = Field(default_factory=dict)
    objects: tuple[ObjectObservation, ...]
    last_event: WorldEvent | None = None


class PrimitiveAction(FrozenModel):
    kind: ActionKind
    target_id: str | None = None
    item_id: str | None = None
    ticks: int = Field(default=1, ge=1)


class InterventionSpec(FrozenModel):
    intervention_id: str
    mechanism: str
    controlled_variables: dict[str, str | int | float | bool]
    target_variable: str
    fixed_context: dict[str, str | int | float | bool] = Field(default_factory=dict)
    predicted_outcomes: dict[str, Any] = Field(default_factory=dict)
    stopping_condition: str = "terminal_event"


class TransitionRecord(FrozenModel):
    transition_id: str
    context: dict[str, Any]
    state_before: Observation
    intended_intervention: InterventionSpec | None = None
    executed_action: PrimitiveAction
    intervention_compliance: bool
    state_after: Observation
    observed_delta: dict[str, Any]
    event: WorldEvent
    possible_confounders: tuple[str, ...] = ()
    controller_failure: str | None = None
    provenance: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Hypothesis(FrozenModel):
    hypothesis_id: str
    mechanism: str
    payload: dict[str, Any]
    prior: float = Field(default=1.0, gt=0)


class TheoryIR(FrozenModel):
    mechanism: str
    representation: TheoryRepresentation
    scope: dict[str, Any]
    preconditions: tuple[str, ...] = ()
    latent_variables: tuple[str, ...] = ()
    effects: dict[str, Any]
    parameters: dict[str, Any] = Field(default_factory=dict)
    composition: tuple[str, ...] = ()


class TheoryVersion(FrozenModel):
    theory_id: str
    version: int = Field(ge=1)
    ir: TheoryIR
    parent_version: int | None = None
    evidence_ids: tuple[str, ...] = ()
    counterexample_ids: tuple[str, ...] = ()
    validation_score: float
    confidence: float = Field(ge=0, le=1)
    status: TheoryStatus
    created_by: str = "heuristic"


class SymbolPrimitive(FrozenModel):
    symbol_id: str
    name: str
    type_signature: str
    definition_ir: dict[str, Any]
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...] = ()
    validation_examples: tuple[str, ...] = ()
    used_by_theories: tuple[str, ...] = ()
    parent_symbols: tuple[str, ...] = ()
    generation: int = Field(default=1, ge=1)
    compression_gain: float
    predictive_gain: float
    cross_world_reuse: int = Field(default=0, ge=0)
    status: str = "candidate"


class ExperimentResult(FrozenModel):
    experiment_id: str
    mechanism: str
    intervention: InterventionSpec
    transitions: tuple[TransitionRecord, ...]
    outcome: dict[str, Any]
    compliant: bool
    cost: float = Field(default=1.0, ge=0)


class RunConfig(FrozenModel):
    seed: int = 0
    intervention_budget: int = Field(default=80, ge=1)
    change_episode: int | None = Field(default=1, ge=1)
    library_learning: bool = True
    hypothesis_generator: str = "heuristic"
    planner: str = "information_gain"
    planner_checkpoint: str | None = None
    world_name: str = "mvp"


class EvaluationReport(FrozenModel):
    seed: int
    hidden_intervention_accuracy: float = Field(ge=0, le=1)
    experiment_count: int = Field(ge=0)
    experiment_efficiency: float = Field(ge=0)
    change_detection_delay: int | None = Field(default=None, ge=0)
    revision_experiments: int = Field(default=0, ge=0)
    description_length: int = Field(ge=0)
    symbol_count: int = Field(ge=0)
    symbol_reuse: int = Field(ge=0)
    task_success_rate: float = Field(ge=0, le=1)
    mechanism_accuracy: dict[str, float]
    theory_versions: dict[str, int]
    notes: tuple[str, ...] = ()

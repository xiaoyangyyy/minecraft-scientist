"""Serializable RGB-policy action roles for visual frontier replay."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualFrontierProgram(BaseModel):
    """A replayable relative camera turn and anonymous motion intervention.

    The stored role tuple identifies only functional slots learned by the agent.  It
    deliberately contains neither simulator object IDs nor semantic control names.
    A fresh session binds ``motion_slot`` to the correspondingly sorted opaque
    movement handle exposed by the current observation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_steps: int = Field(ge=0)
    motion_slot: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    move_frames: int = Field(ge=1)

    VERSION: ClassVar[str] = "visual-frontier-v1"

    def to_roles(self) -> tuple[str, ...]:
        return (
            self.VERSION,
            f"turn-slot:{self.turn_steps}",
            f"motion-slot:{self.motion_slot}",
            f"repetitions:{self.repetitions}",
            f"move-frames:{self.move_frames}",
        )

    @classmethod
    def from_roles(cls, roles: tuple[str, ...]) -> Self:
        if len(roles) != 5 or roles[0] != cls.VERSION:
            raise ValueError("unsupported visual frontier role schema")
        fields = (
            ("turn-slot", "turn_steps"),
            ("motion-slot", "motion_slot"),
            ("repetitions", "repetitions"),
            ("move-frames", "move_frames"),
        )
        values: dict[str, int] = {}
        for role, (expected, field) in zip(roles[1:], fields, strict=True):
            prefix, separator, raw = role.partition(":")
            if separator != ":" or prefix != expected or not raw.isdecimal():
                raise ValueError("malformed visual frontier action role")
            values[field] = int(raw)
        return cls.model_validate(values)


class VisualHomingProgram(BaseModel):
    """Bounded visual closed-loop return program for a known directed edge.

    Unlike :class:`VisualFrontierProgram`, ``max_repetitions`` is a safety budget,
    not a fixed travel distance.  After every anonymous motion intervention the
    caller must acquire a fresh visual observation and stop only when the target
    place's persisted calibrated boundary accepts it.  Exhausting the budget is a
    failed traversal and must not create a place transition.

    The target place is supplied by the surrounding directed transition.  It is
    intentionally absent from the action-role tuple, as are simulator object IDs
    and semantic control names.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_steps: int = Field(ge=0)
    motion_slot: int = Field(ge=0)
    min_repetitions: int = Field(ge=1)
    max_repetitions: int = Field(ge=1)
    move_frames: int = Field(ge=1)
    settle_ticks: int = Field(ge=0)

    VERSION: ClassVar[str] = "visual-homing-v3"

    @model_validator(mode="after")
    def repetition_budget_is_valid(self) -> Self:
        if self.min_repetitions > self.max_repetitions:
            raise ValueError("visual homing minimum exceeds maximum repetitions")
        return self

    def to_roles(self) -> tuple[str, ...]:
        return (
            self.VERSION,
            f"turn-slot:{self.turn_steps}",
            f"motion-slot:{self.motion_slot}",
            f"min-repetitions:{self.min_repetitions}",
            f"max-repetitions:{self.max_repetitions}",
            f"move-frames:{self.move_frames}",
            f"settle-ticks:{self.settle_ticks}",
        )

    def evidence_match_count(self, matches: Sequence[bool]) -> int:
        """Return the target evidence used by this memoryless stopping rule."""

        return int(bool(matches) and bool(matches[-1]))

    def should_stop(self, matches: Sequence[bool]) -> bool:
        return (
            self.min_repetitions <= len(matches) <= self.max_repetitions
            and self.evidence_match_count(matches) >= 1
        )

    @classmethod
    def from_roles(cls, roles: tuple[str, ...]) -> Self:
        if len(roles) != 7 or roles[0] != cls.VERSION:
            raise ValueError("unsupported visual homing role schema")
        fields = (
            ("turn-slot", "turn_steps"),
            ("motion-slot", "motion_slot"),
            ("min-repetitions", "min_repetitions"),
            ("max-repetitions", "max_repetitions"),
            ("move-frames", "move_frames"),
            ("settle-ticks", "settle_ticks"),
        )
        values: dict[str, int] = {}
        for role, (expected, field) in zip(roles[1:], fields, strict=True):
            prefix, separator, raw = role.partition(":")
            if separator != ":" or prefix != expected or not raw.isdecimal():
                raise ValueError("malformed visual homing action role")
            values[field] = int(raw)
        return cls.model_validate(values)


class VisualEvidenceHomingProgram(VisualHomingProgram):
    """Closed-loop homing with bounded short-term visual evidence integration.

    A target observation immediately before the preregistered decision step can
    survive one transient miss from weather, animation, or viewpoint jitter. The
    fixed window is part of the persisted action role, so replay cannot silently
    change the stopping policy.
    """

    evidence_window: int = Field(ge=1)
    required_matches: int = Field(ge=1)

    VERSION: ClassVar[str] = "visual-homing-v4"

    @model_validator(mode="after")
    def evidence_requirement_is_valid(self) -> Self:
        if self.required_matches > self.evidence_window:
            raise ValueError("visual homing evidence exceeds its window")
        return self

    def evidence_match_count(self, matches: Sequence[bool]) -> int:
        return sum(bool(item) for item in matches[-self.evidence_window :])

    def should_stop(self, matches: Sequence[bool]) -> bool:
        return (
            self.min_repetitions <= len(matches) <= self.max_repetitions
            and self.evidence_match_count(matches) >= self.required_matches
        )

    def to_roles(self) -> tuple[str, ...]:
        return (
            self.VERSION,
            f"turn-slot:{self.turn_steps}",
            f"motion-slot:{self.motion_slot}",
            f"min-repetitions:{self.min_repetitions}",
            f"max-repetitions:{self.max_repetitions}",
            f"move-frames:{self.move_frames}",
            f"settle-ticks:{self.settle_ticks}",
            f"evidence-window:{self.evidence_window}",
            f"required-matches:{self.required_matches}",
        )

    @classmethod
    def from_roles(cls, roles: tuple[str, ...]) -> Self:
        if len(roles) != 9 or roles[0] != cls.VERSION:
            raise ValueError("unsupported visual evidence homing role schema")
        fields = (
            ("turn-slot", "turn_steps"),
            ("motion-slot", "motion_slot"),
            ("min-repetitions", "min_repetitions"),
            ("max-repetitions", "max_repetitions"),
            ("move-frames", "move_frames"),
            ("settle-ticks", "settle_ticks"),
            ("evidence-window", "evidence_window"),
            ("required-matches", "required_matches"),
        )
        values: dict[str, int] = {}
        for role, (expected, field) in zip(roles[1:], fields, strict=True):
            prefix, separator, raw = role.partition(":")
            if separator != ":" or prefix != expected or not raw.isdecimal():
                raise ValueError("malformed visual evidence homing action role")
            values[field] = int(raw)
        return cls.model_validate(values)

"""Serializable RGB-policy action roles for visual frontier replay."""

from __future__ import annotations

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field


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

"""Pixel-evidence-only candidate ranking for directed visual goal search."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class VisualGoalCandidate(BaseModel):
    """The complete policy input for one executed anonymous frontier probe.

    Simulator position, physical distance, and evaluator verdicts intentionally
    have no fields here. Unknown fields are rejected at this capability boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    heading_steps: int = Field(ge=0)
    signature: tuple[int, ...] = Field(min_length=1)
    recognized_place_id: str | None = None


class RankedVisualGoalCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: VisualGoalCandidate
    goal_visual_distance: float = Field(ge=0)
    source_visual_distance: float = Field(ge=0)


def rank_visual_goal_candidates(
    goal_signature: Sequence[int],
    source_signatures: dict[str, Sequence[int]],
    candidates: Sequence[VisualGoalCandidate],
    *,
    goal_place_id: str,
) -> tuple[RankedVisualGoalCandidate, ...]:
    """Rank novel/goal probes by goal distance, novelty, then stable identity.

    Each candidate receives the source scan from its own isolated session.
    A revisit to another known place cannot masquerade as a novel expansion.
    This function only ranks visual evidence; physical validation is a separate
    post-selection decision and cannot silently substitute the runner-up.
    """

    if not goal_signature:
        raise ValueError("goal panorama must be nonempty")
    identities = [item.candidate_id for item in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate goal-search candidate identity")
    ranked = []
    for candidate in candidates:
        source = source_signatures[candidate.candidate_id]
        if len(candidate.signature) != len(goal_signature) or len(source) != len(
            goal_signature
        ):
            raise ValueError("goal-search panorama widths differ")
        if candidate.recognized_place_id not in (None, goal_place_id):
            continue
        ranked.append(
            RankedVisualGoalCandidate(
                candidate=candidate,
                goal_visual_distance=math.dist(goal_signature, candidate.signature),
                source_visual_distance=math.dist(source, candidate.signature),
            )
        )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                item.goal_visual_distance,
                -item.source_visual_distance,
                item.candidate.heading_steps,
                item.candidate.candidate_id,
            ),
        )
    )

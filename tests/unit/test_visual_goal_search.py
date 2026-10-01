import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration.goal_search import (
    VisualGoalCandidate,
    rank_visual_goal_candidates,
)


def test_goal_rank_rejects_known_nongoal_revisit_and_is_order_independent() -> None:
    candidates = [
        VisualGoalCandidate(candidate_id="near", heading_steps=3, signature=(2, 0)),
        VisualGoalCandidate(candidate_id="far", heading_steps=4, signature=(5, 0)),
        VisualGoalCandidate(
            candidate_id="alias",
            heading_steps=0,
            signature=(0, 0),
            recognized_place_id="other-place",
        ),
    ]
    sources = {item.candidate_id: (10, 0) for item in candidates}
    ranked = rank_visual_goal_candidates(
        (0, 0), sources, candidates, goal_place_id="goal"
    )
    assert [item.candidate.candidate_id for item in ranked] == ["near", "far"]
    assert ranked == rank_visual_goal_candidates(
        (0, 0), sources, list(reversed(candidates)), goal_place_id="goal"
    )
    assert ranked[0].goal_visual_distance == 2
    assert ranked[0].source_visual_distance == 8


def test_goal_rank_tie_prefers_novelty_then_heading_and_accepts_goal_revisit() -> None:
    candidates = [
        VisualGoalCandidate(candidate_id="x", heading_steps=5, signature=(0, 1)),
        VisualGoalCandidate(candidate_id="y", heading_steps=2, signature=(0, 1)),
        VisualGoalCandidate(
            candidate_id="goal",
            heading_steps=8,
            signature=(0, 0),
            recognized_place_id="goal",
        ),
    ]
    sources = {"x": (0, 2), "y": (0, 3), "goal": (0, 2)}
    ranked = rank_visual_goal_candidates(
        (0, 0), sources, candidates, goal_place_id="goal"
    )
    assert [item.candidate.candidate_id for item in ranked] == ["goal", "y", "x"]
    sources["x"] = sources["y"]
    ranked = rank_visual_goal_candidates(
        (0, 0), sources, candidates, goal_place_id="goal"
    )
    assert [item.candidate.heading_steps for item in ranked] == [8, 2, 5]


def test_goal_rank_rejects_privileged_fields_and_incompatible_evidence() -> None:
    with pytest.raises(ValidationError):
        VisualGoalCandidate(
            candidate_id="leak", heading_steps=0, signature=(1,), xpos=1
        )
    candidate = VisualGoalCandidate(candidate_id="x", heading_steps=0, signature=(1,))
    with pytest.raises(ValueError, match="widths"):
        rank_visual_goal_candidates(
            (0, 0), {"x": (1, 1)}, [candidate], goal_place_id="g"
        )
    with pytest.raises(ValueError, match="duplicate"):
        rank_visual_goal_candidates(
            (0,), {"x": (1,)}, [candidate, candidate], goal_place_id="g"
        )

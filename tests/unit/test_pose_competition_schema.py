import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration import pose_competition as competition


def test_empty_candidate_is_lazy_and_does_not_certify_uniqueness(monkeypatch):
    monkeypatch.setattr(
        competition,
        "_opencv",
        lambda: pytest.fail("empty candidate loaded optional OpenCV"),
    )
    result = competition.inspect_pose_competition(None, np.eye(3))
    assert not result.ambiguity_detected and not result.trials
    assert (
        not {"supported", "unit_translation", "relative_rotation"}
        & competition.PoseCompetitionEvidence.model_fields.keys()
    )


def test_fixed_grid_is_bounded_and_configuration_frozen():
    config = competition.PoseCompetitionConfig()
    assert len(config.grid()) == 19
    with pytest.raises(ValidationError):
        config.direction_separation_degrees = 30
    for offsets in ((0,), (1, 1), (2, 1), (4,), (np.nan,)):
        with pytest.raises(ValidationError):
            competition.PoseCompetitionConfig(rotation_offsets_degrees=offsets)


def test_flag_cannot_invent_an_ambiguity_witness():
    with pytest.raises(ValidationError, match="ambiguity flag"):
        competition.PoseCompetitionEvidence(
            config=competition.PoseCompetitionConfig(),
            ambiguity_detected=True,
            reason="forged",
        )

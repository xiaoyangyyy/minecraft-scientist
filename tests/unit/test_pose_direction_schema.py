import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration import pose_direction_search as search


def test_direction_search_is_lazy_without_candidate(monkeypatch):
    monkeypatch.setattr(
        search, "_opencv", lambda: pytest.fail("empty candidate loaded OpenCV")
    )
    result = search.inspect_direction_search(None, np.eye(3))
    assert (
        not result.ambiguity_detected and not result.trials and result.reference is None
    )
    assert (
        not {"supported", "relative_rotation", "unit_translation"}
        & search.DirectionSearchEvidence.model_fields.keys()
    )


@pytest.mark.parametrize("offsets", [(20,), (45, 25), (25, 25), (61,), (np.nan,)])
def test_direction_grid_rejects_unbounded_or_nonwitness_offsets(offsets):
    with pytest.raises(ValidationError):
        search.DirectionSearchConfig(direction_offsets_degrees=offsets)


def test_reference_and_witness_flags_require_recorded_evidence():
    for flags in (
        {"ambiguity_detected": True},
        {"reference_matches_original_checks": True},
    ):
        with pytest.raises(ValidationError):
            search.DirectionSearchEvidence.model_validate(
                {
                    "config": search.DirectionSearchConfig(),
                    "ambiguity_detected": False,
                    "reason": "forged",
                    **flags,
                }
            )

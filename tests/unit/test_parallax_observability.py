import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration.parallax_observability import (
    ConditionalView,
    ParallaxConfig,
    ParallaxEvidence,
)


def test_view_diversity_requires_baseline_angle_and_parallax_together():
    config = ParallaxConfig()
    view = ConditionalView(
        frame_slot=5,
        baseline_ratio=0.5,
        turn_degrees=90,
        median_ray_angle_degrees=2,
        validation_tracks=8,
    )
    assert view.diverse(config)
    for change in (
        {"baseline_ratio": 0.1},
        {"turn_degrees": 180},
        {"median_ray_angle_degrees": 0.5},
    ):
        assert not view.model_copy(update=change).diverse(config)
    assert ParallaxEvidence(
        config=config, conditionally_diverse_view=True, reason="fixture", views=(view,)
    ).conditionally_diverse_view
    with pytest.raises(ValidationError):
        ParallaxEvidence(
            config=config,
            conditionally_diverse_view=False,
            reason="forged",
            views=(view,),
        )


def test_conditional_schema_rejects_nan_or_forged_diversity():
    with pytest.raises(ValidationError):
        ParallaxConfig(minimum_future_baseline_ratio=np.nan)
    with pytest.raises(ValidationError):
        ParallaxEvidence(
            config=ParallaxConfig(), conditionally_diverse_view=True, reason="no data"
        )

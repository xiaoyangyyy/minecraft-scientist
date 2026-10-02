import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration import shadow_motion
from minecraft_scientist.exploration.shadow_motion import (
    ShadowMotionConfig,
    ShadowMotionEvidence,
)


def test_shadow_configuration_is_default_offline_and_frozen():
    config = ShadowMotionConfig()
    assert config.future_frames == 2 and config.temporal.maximum_frames == 32
    assert config.geometry.minimum_tracks == 32
    with pytest.raises(ValidationError):
        config.future_frames = 3
    for change in (
        {"future_frames": 1},
        {"prefix_pixel_tolerance": float("nan")},
        {"yaw": 0},
    ):
        with pytest.raises(ValidationError):
            ShadowMotionConfig(**change)


def test_shadow_abstention_does_not_import_or_invoke_optional_geometry(monkeypatch):
    monkeypatch.setattr(
        shadow_motion, "_opencv", lambda: pytest.fail("abstention invoked CV")
    )
    result = shadow_motion.validate_shadow_motion(None, None, None)
    assert not result.supported and result.candidate is None
    assert result.unit_translation is None and not result.frame_checks


def test_shadow_schema_forbids_executable_pose_or_partitions_without_candidate():
    values = {"candidate": None, "supported": False, "reason": "fixture"}
    assert ShadowMotionEvidence(**values).model_dump()["supported"] is False
    for change in (
        {"supported": True},
        {"training_track_ids": (1,)},
        {"unit_translation": (1, 0, 0)},
    ):
        with pytest.raises(ValidationError):
            ShadowMotionEvidence.model_validate(values | change)

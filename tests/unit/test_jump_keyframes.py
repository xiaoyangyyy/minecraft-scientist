import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="optional jump-control geometry")
pytest.importorskip("PIL.Image", reason="optional capture recorder")

from minecraft_scientist.exploration import stable_keyframe_map as stable
from minecraft_scientist.exploration.settled_map_control import PublicControlStep
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


@pytest.fixture
def recorder():
    source = (
        Path(__file__).resolve().parents[2]
        / "environments/minestudio/capture_controlled_rgb_motion.py"
    )
    spec = importlib.util.spec_from_file_location("jump_recorder_definition", source)
    definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definition)
    return definition.ProbeRecorder


@pytest.mark.parametrize("seed", [0, 19, 207])
def test_compound_motors_keep_four_anonymous_handles_and_exact_cardinal_encoding(
    recorder, seed
):
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    old, new = recorder(), recorder(jump_controls=True)
    scenes = [item.perceive(rgb, {}, seed=seed) for item in (old, new)]
    assert scenes[0].objects == scenes[1].objects
    for index, (controller, scene) in enumerate(zip((old, new), scenes, strict=True)):
        handles = [
            obj for obj in scene.objects if obj.affordances == (ActionKind.MOVE,)
        ]
        assert len(handles) == 4
        assert all(
            "jump" not in repr(obj) and "forward" not in repr(obj) for obj in handles
        )
        for cardinal, obj in zip(
            ("forward", "back", "left", "right"), handles, strict=True
        ):
            payload = controller.encode_action(
                PrimitiveAction(kind=ActionKind.MOVE, target_id=obj.object_id), scene
            )
            assert int(payload["jump"]) == index
            assert [
                key for key in ("forward", "back", "left", "right") if int(payload[key])
            ] == [cardinal]
            assert not int(payload["sprint"]) and not int(payload["sneak"])
        for kind, target in (
            (ActionKind.WAIT, None),
            (
                ActionKind.LOOK,
                next(
                    obj.object_id
                    for obj in scene.objects
                    if ActionKind.LOOK in obj.affordances
                ),
            ),
        ):
            assert not int(
                controller.encode_action(
                    PrimitiveAction(kind=kind, target_id=target), scene
                )["jump"]
            )


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_recorder_registration_is_explicit_boolean(recorder, flag):
    with pytest.raises(ValueError, match="explicit boolean"):
        recorder(jump_controls=flag)


@pytest.mark.parametrize(
    "field", ["location_stats", "xpos", "ypos", "zpos", "yaw", "pitch"]
)
def test_jump_recorder_never_consumes_private_position_fields(recorder, field):
    with pytest.raises(RuntimeError, match="privileged"):
        recorder(jump_controls=True).perceive(
            np.zeros((224, 224, 3), dtype=np.uint8), {field: 0}, seed=19
        )


def acquire(first, execute, **kwargs):
    return stable.acquire_stable_keyframe_map(
        first,
        execute,
        ((100, 0, 112), (0, 100, 112), (0, 0, 1)),
        anchor_frame=48,
        micro_controls=True,
        responsive_controls=True,
        recovery_controls=True,
        safe_seed_controls=True,
        jump_controls=True,
        **kwargs,
    )


def test_real_public_pixels_keep_compound_registration_without_inventing_motion():
    gray = cv2.GaussianBlur(
        np.random.default_rng(29).integers(0, 256, (224, 224), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    first = np.repeat(gray[:, :, None], 3, axis=2)
    calls = []

    def execute(kind, handle):
        calls.append((kind, handle))
        return PublicControlStep(first.copy(), kind, handle, True, 1)

    result = acquire(first, execute)
    assert result["movement_program_registration"] == "opaque-four-cardinal-jump-v1"
    assert result["terminal"] == "map_unavailable" and result["frozen"] is None
    moves = [handle for kind, handle in calls if kind == "move"]
    assert len(moves) == 11 and set(moves) == set(range(4))
    assert len(result["epochs"]) == 1 and result["epochs"][0]["frame_ordinals"] == [56]
    assert (
        not result["future"]
        and not result["motion"]
        and not result["navigation_accepted"]
    )
    assert len(calls) + 48 <= 288
    assert all(
        not sample["selected"] for sample in result["samples"] if "trial" in sample
    )


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_controller_rejects_untyped_jump_flag_before_dispatch(flag):
    def forbidden(*args):
        raise AssertionError("invalid protocol dispatched a control")

    with pytest.raises(ValueError, match="explicit boolean"):
        stable.acquire_stable_keyframe_map(
            np.zeros((224, 224, 3), dtype=np.uint8),
            forbidden,
            np.eye(3),
            anchor_frame=48,
            jump_controls=flag,
        )


def test_jump_protocol_cannot_disguise_itself_as_parent_or_relax_gates():
    settings = stable.StableJumpSafeSeedKeyframePolicy().model_dump(mode="json")
    with pytest.raises(ValidationError):
        stable.StableSafeSeedKeyframePolicy.model_validate(settings)
    settings["consensus"]["minimum_models"] = 1
    with pytest.raises(ValidationError, match="minimum_models"):
        stable.StableJumpSafeSeedKeyframePolicy.model_validate(settings)
    with pytest.raises(ValueError, match="explicit safe-seed"):
        stable.acquire_stable_keyframe_map(
            np.zeros((224, 224, 3), dtype=np.uint8),
            lambda *args: None,
            np.eye(3),
            anchor_frame=48,
            jump_controls=True,
        )

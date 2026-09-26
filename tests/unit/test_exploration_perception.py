from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from minecraft_scientist.exploration import (
    AnonymousSceneEncoder,
    ExplorationMemory,
    RGBPanoramaEncoder,
    RGBSceneEncoder,
    infer_visual_rotation_period,
)
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
)


def _observation(
    *,
    origin: tuple[float, float, float],
    prefix: str,
    reverse: bool = False,
    hardness_offset: float = 0,
) -> Observation:
    relative_blocks = ((1, -1, 0, 0.5), (-2, 0, 3, 2.0), (0, 2, -4, 5.0))
    objects = [
        ObjectObservation(
            object_id=f"{prefix}_{index}",
            affordances=(ActionKind.INTERACT,),
            observable_features={
                "x": origin[0] + dx,
                "y": origin[1] + dy,
                "z": origin[2] + dz,
                "distance": 1.0,
                "diggable": True,
                "hardness": hardness + hardness_offset,
            },
        )
        for index, (dx, dy, dz, hardness) in enumerate(relative_blocks)
    ]
    if reverse:
        objects.reverse()
    return Observation(
        episode_id=1,
        step=0,
        inventory={},
        agent_state={"x": origin[0], "y": origin[1], "z": origin[2]},
        objects=tuple(objects),
    )


def test_anonymous_scene_descriptor_ignores_ids_order_and_translation() -> None:
    encoder = AnonymousSceneEncoder()
    first = _observation(origin=(0.5, 64, 0.5), prefix="obj_a")
    translated = _observation(
        origin=(100.5, 80, -39.5),
        prefix="obj_unrelated",
        reverse=True,
    )
    changed = _observation(
        origin=(0.5, 64, 0.5),
        prefix="obj_c",
        hardness_offset=4,
    )

    assert encoder.encode(first) == encoder.encode(translated)
    assert encoder.encode(first) != encoder.encode(changed)
    assert len(encoder.encode(first)) == encoder.DIMENSIONS


def test_perceptual_views_are_bounded_matchable_and_integrity_checked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    with ExplorationMemory(path, max_exemplars=2) as memory:
        first = memory.observe_place((0,), episode_id=0, step=0)
        second = memory.observe_place((10,), episode_id=0, step=0)
        memory.record_perceptual_view(first.place_id, (0, 0), episode_id=0, step=0)
        memory.record_perceptual_view(first.place_id, (1, 0), episode_id=0, step=1)
        perceptual = memory.record_perceptual_view(
            first.place_id, (4, 0), episode_id=1, step=0
        )
        memory.record_perceptual_view(second.place_id, (20, 20), episode_id=0, step=0)

        assert len(perceptual.exemplars) == 2
        assert perceptual.observations == 3
        assert perceptual.episodes == (0, 1)
        assert memory.match_perceptual_place((1, 0), tolerance=2) == first.place_id
        assert memory.match_perceptual_place((9, 9), tolerance=2) is None
        assert memory.summary()["perceptual_place_count"] == 2
        assert memory.verify_integrity()

        memory.record_perceptual_view(second.place_id, (0, 0), episode_id=1, step=1)
        assert (
            memory.match_perceptual_place((0, 0), tolerance=0, ambiguity_margin=0)
            is None
        )
        assert memory.verify_integrity()


def _rgb_fixture(*, mirrored: bool = False) -> np.ndarray:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[8:70, 40:88] = (8, 8, 8)
    frame[18:58, 54:74] = (205, 205, 205)
    if mirrored:
        frame[68:106, 16:56] = (180, 20, 20)
    else:
        frame[68:106, 68:108] = (180, 20, 20)
    return frame


def _shift(frame: np.ndarray, pixels: int) -> np.ndarray:
    result = np.zeros_like(frame)
    if pixels > 0:
        result[:, pixels:] = frame[:, :-pixels]
    elif pixels < 0:
        result[:, :pixels] = frame[:, -pixels:]
    else:
        result[:] = frame
    return result


def test_panorama_signature_is_invariant_to_scan_start_and_order() -> None:
    scene_encoder = RGBSceneEncoder(grid_size=4)
    panorama_encoder = RGBPanoramaEncoder(scene_encoder)
    views = tuple(
        np.roll(_rgb_fixture(), shift, axis=1) for shift in (-30, -10, 10, 30)
    )
    signatures = tuple(scene_encoder.encode(frame) for frame in views)

    reference = panorama_encoder.encode_signatures(signatures)
    assert reference == panorama_encoder.encode_signatures(
        signatures[2:] + signatures[:2]
    )
    assert reference == panorama_encoder.encode_signatures(tuple(reversed(signatures)))
    assert reference == panorama_encoder.encode(views)
    assert len(reference) == panorama_encoder.dimensions
    assert (
        panorama_encoder.deserialize(panorama_encoder.serialize(reference)) == reference
    )


def test_visual_rotation_period_uses_first_eligible_rgb_recurrence() -> None:
    anchor = (0, 0)
    scan = ((4, 0), (6, 0), (4, 0), (1, 0), (0, 0), (1, 0), (0, 0))

    period, distances = infer_visual_rotation_period(anchor, scan, minimum_steps=4)

    assert period == 5
    assert distances[period - 1] == 0

    with pytest.raises(ValueError, match="shorter"):
        infer_visual_rotation_period(anchor, scan[:2], minimum_steps=4)


def test_rgb_encoder_and_calibrated_modality_memory_generalize_safely(
    tmp_path: Path,
) -> None:
    encoder = RGBSceneEncoder(grid_size=8)
    anchor_frame = _rgb_fixture()
    positive_brightness = np.clip(anchor_frame.astype(float) * 0.85, 0, 255)
    positive_shift = _shift(anchor_frame, 2)
    heldout = np.clip(_shift(anchor_frame, -2).astype(float) * 1.15, 0, 255)
    negative = _rgb_fixture(mirrored=True)
    anchor = encoder.encode(anchor_frame)
    positives = (
        encoder.encode(positive_brightness),
        encoder.encode(positive_shift),
    )
    heldout_signature = encoder.encode(heldout)
    negative_signature = encoder.encode(negative)

    assert len(anchor) == encoder.dimensions
    assert math.dist(anchor, heldout_signature) < math.dist(anchor, negative_signature)

    with ExplorationMemory(tmp_path / "rgb.sqlite3", max_exemplars=4) as memory:
        place = memory.observe_place((0,), episode_id=0, step=0)
        memory.record_modality_view(
            encoder.VERSION,
            place.place_id,
            anchor,
            episode_id=0,
            step=0,
        )
        calibrated = memory.calibrate_modality_place(
            encoder.VERSION,
            place.place_id,
            positives,
            (negative_signature,),
            episode_id=0,
            step=1,
            margin=5,
        )

        assert calibrated.calibration is not None
        assert calibrated.match_radius < (
            calibrated.calibration.nearest_negative_distance
        )
        assert (
            memory.match_modality_place(encoder.VERSION, heldout_signature)
            == place.place_id
        )
        assert memory.match_modality_place(encoder.VERSION, negative_signature) is None
        exemplars_before = calibrated.exemplars
        accepted = memory.record_modality_view(
            encoder.VERSION,
            place.place_id,
            heldout_signature,
            episode_id=1,
            step=0,
        )
        assert accepted.exemplars == exemplars_before
        with pytest.raises(ValueError, match="rejected the unvalidated view"):
            memory.record_modality_view(
                encoder.VERSION,
                place.place_id,
                negative_signature,
                episode_id=1,
                step=1,
            )
        assert memory.summary()["calibrated_modality_place_count"] == 1
        assert memory.verify_integrity()

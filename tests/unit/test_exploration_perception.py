from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from minecraft_scientist.exploration import (
    AnonymousSceneEncoder,
    ExplorationMemory,
    RGBEdgeCompactSpectrumPanoramaEncoder,
    RGBEdgeCompactSpectrumSceneEncoder,
    RGBEdgeDualBandSpectrumPanoramaEncoder,
    RGBEdgeDualBandSpectrumSceneEncoder,
    RGBEdgeExpandedSpectrumPanoramaEncoder,
    RGBEdgeExpandedSpectrumSceneEncoder,
    RGBEdgeHistogramPanoramaEncoder,
    RGBEdgeHistogramSceneEncoder,
    RGBEdgePanoramaEncoder,
    RGBEdgeSceneEncoder,
    RGBEdgeSpectrumPanoramaEncoder,
    RGBEdgeSpectrumSceneEncoder,
    RGBPanoramaEncoder,
    RGBSceneEncoder,
    VisualEvidenceHomingProgram,
    VisualHomingProgram,
    align_panorama_heading,
    infer_visual_rotation_period,
    temporal_median_rgb,
)
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
)


def test_panorama_heading_alignment_recovers_cyclic_orientation() -> None:
    reference = ((0, 0), (10, 1), (20, 4), (30, 9))
    query = reference[2:] + reference[:2]

    alignment = align_panorama_heading(reference, query)

    assert alignment.reference_offset == 2
    assert alignment.right_turn_steps == 2
    assert alignment.mean_distance == 0
    assert alignment.separation > 0


def test_panorama_heading_alignment_rejects_incompatible_views() -> None:
    with pytest.raises(ValueError, match="equal multi-view periods"):
        align_panorama_heading(((0,), (1,)), ((0,),))


def test_visual_homing_program_round_trips_without_semantic_controls() -> None:
    program = VisualHomingProgram(
        turn_steps=6,
        motion_slot=1,
        min_repetitions=4,
        max_repetitions=7,
        move_frames=4,
        settle_ticks=10,
    )

    roles = program.to_roles()

    assert VisualHomingProgram.from_roles(roles) == program
    assert roles == (
        "visual-homing-v3",
        "turn-slot:6",
        "motion-slot:1",
        "min-repetitions:4",
        "max-repetitions:7",
        "move-frames:4",
        "settle-ticks:10",
    )
    assert not any(
        token in role
        for role in roles
        for token in ("forward", "jump", "back", "left", "right", "obj_")
    )
    with pytest.raises(ValueError, match="malformed visual homing"):
        VisualHomingProgram.from_roles(
            (*roles[:4], "max-repetitions:-1", *roles[5:])
        )


def test_visual_evidence_homing_integrates_only_its_bounded_window() -> None:
    program = VisualEvidenceHomingProgram(
        turn_steps=6,
        motion_slot=1,
        min_repetitions=6,
        max_repetitions=6,
        move_frames=4,
        settle_ticks=10,
        evidence_window=2,
        required_matches=1,
    )

    assert not program.should_stop((False, False, False, True, False))
    assert program.should_stop((False, False, False, False, True, False))
    assert program.evidence_match_count((True, False, False)) == 0
    assert VisualEvidenceHomingProgram.from_roles(program.to_roles()) == program
    assert program.to_roles()[0] == "visual-homing-v4"
    with pytest.raises(ValueError, match="exceeds its window"):
        VisualEvidenceHomingProgram(
            turn_steps=0,
            motion_slot=0,
            min_repetitions=1,
            max_repetitions=1,
            move_frames=1,
            settle_ticks=0,
            evidence_window=1,
            required_matches=2,
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


@pytest.mark.parametrize("encoder", [RGBSceneEncoder(), RGBEdgeSceneEncoder()])
def test_rgb_encoders_ignore_standard_hud_margins(encoder) -> None:
    frame = _rgb_fixture()
    overlaid = frame.copy()
    height, width = frame.shape[:2]
    top = int(height * 0.2)
    bottom = int(height * 0.76)
    left = int(width * 0.18)
    right = int(width * (1 - 0.18))
    overlaid[:top] = (255, 0, 255)
    overlaid[bottom:] = (255, 0, 255)
    overlaid[top:bottom, :left] = (255, 0, 255)
    overlaid[top:bottom, right:] = (255, 0, 255)

    assert encoder.encode(frame) == encoder.encode(overlaid)

    overlaid[top + 5 : top + 20, left + 5 : left + 20] = (255, 0, 255)
    assert encoder.encode(frame) != encoder.encode(overlaid)


@pytest.mark.parametrize("encoder_type", [RGBSceneEncoder, RGBEdgeSceneEncoder])
def test_rgb_encoders_reject_empty_vertical_crop(encoder_type) -> None:
    with pytest.raises(ValueError, match="vertical fractions"):
        encoder_type(top_fraction=0.8, bottom_fraction=0.8)


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


def test_edge_panorama_ignores_brightness_and_scan_order() -> None:
    edge_encoder = RGBEdgeSceneEncoder(grid_size=8)
    panorama_encoder = RGBEdgePanoramaEncoder(edge_encoder)
    views = tuple(
        np.roll(_rgb_fixture(), shift, axis=1) for shift in (-30, -10, 10, 30)
    )
    darker = tuple(
        np.clip(frame.astype(float) * 0.45, 0, 255).astype(np.uint8)
        for frame in reversed(views)
    )

    reference = panorama_encoder.encode(views)
    shifted = panorama_encoder.encode(darker)

    assert reference == shifted
    assert len(reference) == panorama_encoder.dimensions


def test_edge_histogram_panorama_tolerates_spatial_translation() -> None:
    scene_encoder = RGBEdgeHistogramSceneEncoder(grid_size=8, bin_count=6)
    panorama_encoder = RGBEdgeHistogramPanoramaEncoder(scene_encoder)
    views = tuple(
        np.roll(_rgb_fixture(), shift, axis=1) for shift in (-30, -10, 10, 30)
    )
    translated = tuple(np.roll(frame, 3, axis=0) for frame in reversed(views))

    reference = panorama_encoder.encode(views)
    shifted = panorama_encoder.encode(translated)

    assert len(reference) == panorama_encoder.dimensions == 36
    assert math.dist(reference, shifted) < 10
    assert scene_encoder.dimensions == 18


def test_edge_spectrum_panorama_tolerates_spatial_translation() -> None:
    scene_encoder = RGBEdgeSpectrumSceneEncoder(
        grid_size=8, low_frequency_radius=2
    )
    panorama_encoder = RGBEdgeSpectrumPanoramaEncoder(scene_encoder)
    views = tuple(
        np.roll(_rgb_fixture(), shift, axis=1) for shift in (-30, -10, 10, 30)
    )
    translated = tuple(
        np.roll(np.roll(frame, 8, axis=0), -8, axis=1) for frame in reversed(views)
    )

    reference = panorama_encoder.encode(views)
    shifted = panorama_encoder.encode(translated)

    assert len(reference) == panorama_encoder.dimensions == 150
    assert math.dist(reference, shifted) < 25
    assert scene_encoder.dimensions == 75


def test_compact_edge_spectrum_has_distinct_persisted_modality() -> None:
    scene_encoder = RGBEdgeCompactSpectrumSceneEncoder()
    panorama_encoder = RGBEdgeCompactSpectrumPanoramaEncoder(scene_encoder)
    views = tuple(
        np.roll(_rgb_fixture(), shift, axis=1) for shift in (-30, -10, 10, 30)
    )

    signature = panorama_encoder.encode(views)

    assert scene_encoder.low_frequency_radius == 3
    assert scene_encoder.VERSION == "rgb-edge-spectrum-r3-view-v1"
    assert panorama_encoder.VERSION == "rgb-edge-spectrum-r3-panorama-v1"
    assert len(signature) == panorama_encoder.dimensions == 294


def test_expanded_edge_spectrum_has_distinct_persisted_modality() -> None:
    scene_encoder = RGBEdgeExpandedSpectrumSceneEncoder()
    panorama_encoder = RGBEdgeExpandedSpectrumPanoramaEncoder(scene_encoder)
    views = tuple(
        np.roll(_rgb_fixture(), shift, axis=1) for shift in (-30, -10, 10, 30)
    )

    signature = panorama_encoder.encode(views)

    assert scene_encoder.low_frequency_radius == 5
    assert scene_encoder.VERSION == "rgb-edge-spectrum-r5-view-v1"
    assert panorama_encoder.VERSION == "rgb-edge-spectrum-r5-panorama-v1"
    assert len(signature) == panorama_encoder.dimensions == 726


def test_dual_band_edge_spectrum_has_distinct_persisted_modality() -> None:
    scene_encoder = RGBEdgeDualBandSpectrumSceneEncoder()
    panorama_encoder = RGBEdgeDualBandSpectrumPanoramaEncoder(scene_encoder)
    views = tuple(
        np.roll(_rgb_fixture(), shift, axis=1) for shift in (-30, -10, 10, 30)
    )

    signature = panorama_encoder.encode(views)

    assert tuple(item.low_frequency_radius for item in scene_encoder.encoders) == (
        1,
        6,
    )
    assert scene_encoder.VERSION == "rgb-edge-spectrum-r1-r6-view-v1"
    assert panorama_encoder.VERSION == "rgb-edge-spectrum-r1-r6-panorama-v1"
    assert len(signature) == panorama_encoder.dimensions == 1068


def test_temporal_median_rgb_removes_transient_weather_streaks() -> None:
    base = _rgb_fixture()
    samples = [base.copy() for _ in range(5)]
    samples[0][:, 10:12] = 255
    samples[1][:, 30:32] = 255

    aggregated = temporal_median_rgb(samples)

    assert np.array_equal(aggregated, base)
    with pytest.raises(ValueError, match="positive odd"):
        temporal_median_rgb(samples[:2])


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
        consolidated = memory.consolidate_modality_view(
            encoder.VERSION,
            place.place_id,
            heldout_signature,
            episode_id=1,
            step=1,
        )
        assert heldout_signature in consolidated.exemplars
        assert any(
            event["event_type"] == "modality_view_consolidated"
            for event in memory.iter_events()
        )
        with pytest.raises(ValueError, match="rejected the unvalidated view"):
            memory.record_modality_view(
                encoder.VERSION,
                place.place_id,
                negative_signature,
                episode_id=1,
                step=2,
            )
        with pytest.raises(ValueError, match="rejected consolidation"):
            memory.consolidate_modality_view(
                encoder.VERSION,
                place.place_id,
                negative_signature,
                episode_id=1,
                step=3,
            )
        assert memory.summary()["calibrated_modality_place_count"] == 1
        assert memory.verify_integrity()


def test_modality_memory_can_fit_training_only_maximum_gap_boundary(
    tmp_path: Path,
) -> None:
    with ExplorationMemory(tmp_path / "automatic-margin.sqlite3") as memory:
        place = memory.observe_place((0,), episode_id=0, step=0)
        memory.record_modality_view(
            "synthetic-rgb",
            place.place_id,
            (0, 0),
            episode_id=0,
            step=0,
        )
        calibrated = memory.calibrate_modality_place(
            "synthetic-rgb",
            place.place_id,
            ((3, 0), (4, 0)),
            ((10, 0),),
            episode_id=0,
            step=1,
            margin=None,
        )

        assert calibrated.calibration is not None
        boundary = calibrated.calibration
        assert boundary.maximum_positive_distance == 4
        assert boundary.nearest_negative_distance == 10
        assert boundary.radius == pytest.approx(6.999994)
        assert memory.match_modality_place("synthetic-rgb", (9, 0)) is None
        assert memory.match_modality_place("synthetic-rgb", (4, 0)) == place.place_id
        accepted = memory.score_modality_place(
            "synthetic-rgb", place.place_id, (4, 0)
        )
        rejected = memory.score_modality_place(
            "synthetic-rgb", place.place_id, (9, 0)
        )
        assert accepted.nearest_distance == 4
        assert accepted.acceptance_radius == pytest.approx(6.999994)
        assert accepted.clearance > 0
        assert accepted.accepted is True
        assert rejected.nearest_distance == 9
        assert rejected.clearance < 0
        assert rejected.accepted is False


def test_modality_consolidation_cannot_bridge_persisted_negative_boundary(
    tmp_path: Path,
) -> None:
    with ExplorationMemory(
        tmp_path / "bounded-consolidation.sqlite3", max_exemplars=4
    ) as memory:
        place = memory.observe_place((0,), episode_id=0, step=0)
        memory.record_modality_view(
            "synthetic-rgb",
            place.place_id,
            (0,),
            episode_id=0,
            step=0,
        )
        calibrated = memory.calibrate_modality_place(
            "synthetic-rgb",
            place.place_id,
            ((2,), (3,)),
            ((10,),),
            episode_id=0,
            step=1,
            margin=None,
        )

        assert calibrated.match_radius == pytest.approx(6.499997)
        assert calibrated.negative_exemplars == ((10,),)
        consolidated = memory.consolidate_modality_view(
            "synthetic-rgb",
            place.place_id,
            (4,),
            episode_id=1,
            step=0,
        )
        local_radius = consolidated.exemplar_radii[
            consolidated.exemplars.index((4,))
        ]

        assert local_radius == pytest.approx(6 - calibrated.calibration.margin)
        assert memory.match_modality_place("synthetic-rgb", (6,)) == place.place_id
        assert memory.match_modality_place("synthetic-rgb", (9,)) is None
        assert memory.verify_integrity()


def test_calibrated_condition_prototype_expands_only_to_negative_margin(
    tmp_path: Path,
) -> None:
    with ExplorationMemory(tmp_path / "condition-prototype.sqlite3") as memory:
        place = memory.observe_place((0,), episode_id=0, step=0)
        memory.record_modality_view(
            "synthetic-rgb", place.place_id, (0,), episode_id=0, step=0
        )
        memory.calibrate_modality_place(
            "synthetic-rgb",
            place.place_id,
            ((0,), (4,)),
            ((10,),),
            episode_id=0,
            step=1,
            margin=2,
        )

        promoted = memory.consolidate_modality_view(
            "synthetic-rgb", place.place_id, (4,), episode_id=0, step=2
        )

        local_radius = promoted.exemplar_radii[promoted.exemplars.index((4,))]
        assert local_radius == 4
        assert memory.match_modality_place("synthetic-rgb", (8,)) == place.place_id
        assert memory.match_modality_place("synthetic-rgb", (10,)) is None
        assert memory.verify_integrity()

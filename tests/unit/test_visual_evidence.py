from dataclasses import asdict
import json

import numpy as np
import pytest

from minecraft_scientist.minecraft.visual_evidence import (
    compare_regions,
    central_view_salience,
    hud_safe_view,
    load_grounding_evidence,
    localize_change_grid,
    visual_region_iou,
    verify_equipment_trace,
)


def test_stable_control_distinguishes_change_from_no_change():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert compare_regions(frame, frame, frame).changed is False
    assert compare_regions(frame, frame, frame + 50).changed is True


def test_unstable_control_abstains():
    frame = np.zeros((4, 4, 3))
    assert compare_regions(frame, frame + 20, frame + 100).changed is None


def test_grid_grounding_localizes_strongest_change_with_opaque_id():
    control = np.zeros((8, 8, 3), dtype=np.uint8)
    before = control.copy()
    after = control.copy()
    after[4:8, 0:4] = 80

    grounded = localize_change_grid(control, before, after, rows=2, columns=2)

    assert grounded.evidence.changed is True
    assert grounded.selected_cell_id.startswith("obj_")
    selected = next(
        cell for cell in grounded.cells if cell.object_id == grounded.selected_cell_id
    )
    assert (selected.row, selected.column) == (1, 0)
    assert grounded.selected_region_id.startswith("obj_")
    assert len(grounded.regions) == 1


def test_grid_grounding_abstains_if_any_control_cell_is_unstable():
    control = np.zeros((8, 8, 3), dtype=np.uint8)
    before = control.copy()
    before[:4, :4] = 20
    after = before.copy()
    after[:4, :4] = 100

    grounded = localize_change_grid(control, before, after, rows=2, columns=2)

    assert grounded.evidence.changed is None
    assert grounded.selected_cell_id is None


def test_grid_grounding_ignores_low_amplitude_drift_outside_stable_change():
    control = np.zeros((8, 8, 3), dtype=np.uint8)
    before = control.copy()
    before[:4, :4] = 3
    after = before.copy()
    after[:4, :4] = 7
    after[4:, 4:] = 80

    grounded = localize_change_grid(control, before, after, rows=2, columns=2)

    assert grounded.evidence.changed is True
    selected = next(
        cell for cell in grounded.cells if cell.object_id == grounded.selected_cell_id
    )
    assert (selected.row, selected.column) == (1, 1)


def test_grid_grounding_fails_closed_when_all_cells_are_unstable():
    control = np.zeros((8, 8, 3), dtype=np.uint8)
    before = np.full_like(control, 5)
    after = np.full_like(control, 80)

    grounded = localize_change_grid(control, before, after, rows=2, columns=2)

    assert grounded.evidence.changed is None
    assert grounded.selected_cell_id is None


def test_adjacent_changed_cells_form_one_persistent_region():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    after_a = frame.copy()
    after_b = frame.copy()
    after_a[4:, 4:] = 80
    after_b[4:, 4:] = 90

    first = localize_change_grid(frame, frame, after_a, rows=3, columns=3)
    second = localize_change_grid(frame, frame, after_b, rows=3, columns=3)

    assert len(first.regions) == len(second.regions) == 1
    assert first.selected_region_id == second.selected_region_id
    assert visual_region_iou(first.regions[0], second.regions[0]) == 1.0


def test_visual_region_iou_tolerates_one_cell_boundary_drift():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    after_a = frame.copy()
    after_b = frame.copy()
    after_a[4:, 4:] = 80
    after_b[4:8, 4:] = 80

    first = localize_change_grid(frame, frame, after_a, rows=3, columns=3)
    second = localize_change_grid(frame, frame, after_b, rows=3, columns=3)

    assert visual_region_iou(first.regions[0], second.regions[0]) == 0.5


def test_grid_grounding_reports_stable_no_change_without_target_guess():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    grounded = localize_change_grid(frame, frame, frame, rows=2, columns=2)
    assert grounded.evidence.changed is False
    assert grounded.selected_cell_id is None


def test_hud_safe_view_excludes_image_edges():
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    frame[:20] = 255
    frame[-20:] = 255
    frame[:, :5] = 255
    frame[:, -30:] = 255

    view = hud_safe_view(frame)

    assert view.shape == (90, 90, 3)
    assert not view.any()


def test_central_view_salience_ignores_bright_edges():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[0] = 255
    frame[-1] = 255
    frame[:, 0] = 255
    frame[:, -1] = 255
    assert central_view_salience(frame) == 0.0

    frame[2:6, 2:6] = 200
    assert central_view_salience(frame) == 200.0


def test_central_view_salience_is_symmetric_around_original_frame_center():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[4:12, 6:10] = 180
    frame[4:12, 2:6] = 255

    assert central_view_salience(frame) == 180.0


def test_grounding_artifact_round_trip_is_strict(tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    after = frame.copy()
    after[4:, 4:] = 80
    grounded = localize_change_grid(frame, frame, after, rows=2, columns=2)
    path = tmp_path / "grounding.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "evidence": asdict(grounded.evidence),
        "selected_cell_id": grounded.selected_cell_id,
        "cells": [asdict(cell) for cell in grounded.cells],
        "selected_region_id": grounded.selected_region_id,
        "regions": [asdict(region) for region in grounded.regions],
    }), encoding="utf-8")

    assert load_grounding_evidence(path) == grounded


def test_grounding_artifact_rejects_selected_cell_not_in_grid(tmp_path):
    path = tmp_path / "grounding.json"
    path.write_text(json.dumps({
        "evidence": {
            "control_mae": 0.0,
            "intervention_mae": 20.0,
            "control_stable": True,
            "changed": True,
        },
        "selected_cell_id": "obj_missing",
        "cells": [{
            "object_id": "obj_present",
            "row": 0,
            "column": 0,
            "control_mae": 0.0,
            "intervention_mae": 20.0,
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="selected cell"):
        load_grounding_evidence(path)


def test_grounding_schema_two_requires_selected_region(tmp_path):
    path = tmp_path / "grounding.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "evidence": {
            "control_mae": 0.0,
            "intervention_mae": 20.0,
            "control_stable": True,
            "changed": True,
        },
        "selected_cell_id": "obj_present",
        "cells": [{
            "object_id": "obj_present",
            "row": 0,
            "column": 0,
            "control_mae": 0.0,
            "intervention_mae": 20.0,
        }],
        "selected_region_id": None,
        "regions": [],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="selected region"):
        load_grounding_evidence(path)


@pytest.mark.parametrize("bad", [np.array([]), np.array([float('nan')])])
def test_invalid_regions_rejected(bad):
    with pytest.raises(ValueError):
        compare_regions(bad, bad, bad)


def test_equipment_audit_allows_initial_setup_frames_only():
    audit = verify_equipment_trace(
        ("air", "air", "iron_pickaxe", "iron_pickaxe"), expected="iron_pickaxe"
    )
    assert audit.valid


@pytest.mark.parametrize("trace", [(), ("air",), ("air", "air"),
                                    ("air", "air", None),
                                    ("air", "air", "wheat_seeds")])
def test_equipment_audit_fails_closed_on_missing_or_wrong_telemetry(trace):
    assert not verify_equipment_trace(trace, expected="air").valid


def test_equipment_audit_accepts_empty_hand_aliases():
    assert verify_equipment_trace(("none", "minecraft:air", "air"), expected="air",
                                  verified_from_frame=0).valid

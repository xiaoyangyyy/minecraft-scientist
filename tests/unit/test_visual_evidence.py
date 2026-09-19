import numpy as np
import pytest

from minecraft_scientist.minecraft.visual_evidence import compare_regions, verify_equipment_trace


def test_stable_control_distinguishes_change_from_no_change():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert compare_regions(frame, frame, frame).changed is False
    assert compare_regions(frame, frame, frame + 50).changed is True


def test_unstable_control_abstains():
    frame = np.zeros((4, 4, 3))
    assert compare_regions(frame, frame + 20, frame + 100).changed is None


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

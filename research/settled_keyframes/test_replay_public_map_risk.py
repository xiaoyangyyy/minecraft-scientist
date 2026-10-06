import numpy as np
import pytest

pytest.importorskip(
    "minecraft_scientist.exploration.public_map_risk",
    reason="requires separately verified map-risk prototype; not main promotion",
)

import replay_public_map_risk as replay


def step(kind="wait", handle=None):
    return {
        "absolute_frame": 102,
        "requested_kind": kind,
        "requested_handle": handle,
        "executed_kind": kind,
        "executed_handle": handle,
        "compliant": True,
        "ticks": 1,
    }


def test_matching_receipt_is_preserved_without_mutating_the_original():
    original = step()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    response, cursor = replay.recorded_receipt(
        [original], {102: image}, 0, "wait", None
    )
    assert cursor == 1
    assert response.frame is image
    assert response.executed_kind == original["executed_kind"] == "wait"
    assert response.ticks == 1
    assert original == step()


@pytest.mark.parametrize("cursor", [True, -1, 1, 0.0])
def test_missing_or_invalid_recorded_clock_cannot_invent_controls(cursor):
    with pytest.raises(replay.RecordedControlGap):
        replay.recorded_receipt([step()], {}, cursor, "wait", None)


@pytest.mark.parametrize("requested", [("wait", None), ("move", 1)])
def test_different_recorded_move_cannot_be_reinterpreted_as_a_new_action(requested):
    with pytest.raises(replay.RecordedControlGap):
        replay.recorded_receipt([step("move", 0)], {}, 0, *requested)


def test_output_protection_precedes_any_input_reads(tmp_path):
    route = tmp_path / "capture"
    route.mkdir()
    with pytest.raises(FileExistsError):
        replay.replay(route, route / "diagnostic")
    output = tmp_path / "existing-output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        replay.replay(route, output)

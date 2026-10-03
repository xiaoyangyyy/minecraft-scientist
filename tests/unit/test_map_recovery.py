import copy
import json

import numpy as np
import pytest
from pydantic import ValidationError
from test_active_causal_map import (
    runner as runner,  # noqa: PLC0414 -- fixture re-export
)
from test_causal_map_window import CAMERA
from test_causal_map_window import (
    geometry as geometry,  # noqa: PLC0414 -- fixture re-export
)

from minecraft_scientist.exploration import active_causal_map as active
from minecraft_scientist.exploration import map_recovery as recovery
from minecraft_scientist.exploration.settled_map_control import PublicControlStep
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


def pair_tracks(images, *, shift=0.05, empty=False, **kwargs):
    points = [(20 + col * 25, 20 + row * 50) for row in range(4) for col in range(8)]
    ids = () if empty else tuple(range(32))
    return TemporalTracks(
        image_shape=images[0].shape[:2],
        frame_count=2,
        seed_count=32,
        surviving_counts=(32, len(ids)),
        track_ids=ids,
        pixels=tuple(
            tuple((points[idx][0] + frame * shift, points[idx][1]) for idx in ids)
            for frame in range(2)
        ),
        minimum_patch_correlations=(1.0,) * len(ids),
        chain_cycle_errors=(0.0,) * len(ids),
    )


@pytest.fixture
def recovery_session(runner, monkeypatch):
    image, execute_acquire, _, _ = runner
    acquired = active.acquire_causal_map(
        image, execute_acquire, CAMERA, anchor_frame=48
    )
    # Explicit synthetic rejected parent; this does not constitute runtime evidence.
    parent = {
        "calibration": {"camera_intrinsics": CAMERA.tolist()},
        "acquisition": acquired | {"terminal": "rejected"},
        "settled": {"terminal": "not_started", "failure": None},
    }
    requests = []

    def execute(kind, handle):
        requests.append((kind, handle))
        return PublicControlStep(image.copy(), kind, handle, True)

    monkeypatch.setattr(recovery, "track_frame_chain", pair_tracks)

    def finish(child, frames, execute, **kwargs):
        assert len(frames) == child["evidence_frames"]
        return {"terminal": "completed", "failure": None, "steps": []}

    monkeypatch.setattr(recovery, "run_settled_controls", finish)
    return parent, image, execute, requests


def test_recovery_waits_before_fresh_acquisition_without_overwriting_parent(
    recovery_session,
):
    parent, image, execute, requests = recovery_session
    before = copy.deepcopy(parent)
    result = recovery.recover_public_map(parent, image, execute)
    assert result["terminal"] == "completed" and result["failure"] is None
    assert requests[:8] == [("wait", None)] * 8
    assert result["anchor_frame"] == 53 and result["new_anchor_frame"] == 61
    assert result["seed"] == 2005
    assert result["acquisition"]["anchor_frame"] == 61
    assert result["acquisition"]["policy"]["minimum_ticks"] == 8
    assert result["parent_sha256"] == recovery._digest(before) and parent == before
    assert len({prefix["track_namespace"] for prefix in result["flow_prefixes"]}) == 8
    assert not result["global_gauge_aligned"] and not result["navigation_accepted"]
    assert result["visual_stability_only"]


@pytest.mark.parametrize("empty", [False, True])
def test_no_public_stability_exhausts_waits_without_map_fitting(
    recovery_session, monkeypatch, empty
):
    parent, image, execute, requests = recovery_session
    monkeypatch.setattr(
        recovery,
        "track_frame_chain",
        lambda images, **kw: pair_tracks(images, shift=1.0, empty=empty),
    )
    result = recovery.recover_public_map(parent, image, execute)
    assert requests == [("wait", None)] * 32
    assert result["terminal"] == "wait_exhausted" and result["acquisition"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["kind", "handle", "ticks", "receipt", "compliance"])
def test_failed_wait_cannot_enter_recovery_evidence(recovery_session, case):
    parent, image, _, _ = recovery_session

    def execute(kind, handle):
        if case == "receipt":
            return {"frame": image}
        return PublicControlStep(
            image,
            "move" if case == "kind" else kind,
            0 if case == "handle" else None,
            case != "compliance",
            np.int64(1) if case == "ticks" else 1,
        )

    result = recovery.recover_public_map(parent, image, execute)
    assert result["terminal"] == "control_failed" and result["failure"]
    assert not result["flow_prefixes"] and result["acquisition"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["seed", "camera", "failed_parent", "endpoint"])
def test_bad_parent_refuses_before_recovery_dispatch(recovery_session, case):
    parent, image, execute, requests = recovery_session
    if case == "seed":
        parent["acquisition"]["seed"] = True
    elif case == "camera":
        parent["calibration"]["camera_intrinsics"] = [[0] * 3] * 3
    elif case == "failed_parent":
        parent["settled"]["failure"] = {"type": "injected"}
    else:
        parent["settled"].update(
            terminal="support_aborted",
            evidence_frames=6,
            prefixes=[{"absolute_frame": 54}],
        )
    with pytest.raises(ValueError):
        recovery.recover_public_map(parent, image, execute)
    assert not requests


@pytest.mark.parametrize(
    "policy",
    [
        {"maximum_wait_ticks": 33},
        {"minimum_wait_ticks": 7},
        {"minimum_tracks": 31},
        {"median_flow_pixels": 0.3},
    ],
)
def test_policy_cannot_silently_relax_recovery_gates(recovery_session, policy):
    parent, image, execute, requests = recovery_session
    with pytest.raises(ValidationError):
        recovery.recover_public_map(parent, image, execute, policy=policy)
    assert not requests


def test_parent_mutation_is_rejected_even_on_exhausted_wait(
    recovery_session, monkeypatch
):
    parent, image, execute, _ = recovery_session
    monkeypatch.setattr(
        recovery,
        "track_frame_chain",
        lambda images, **kw: pair_tracks(images, empty=True),
    )

    def mutate(kind, handle):
        parent["acquisition"]["terminal"] = "forged"
        return execute(kind, handle)

    with pytest.raises(ValueError, match="immutable parent"):
        recovery.recover_public_map(parent, image, mutate)

import numpy as np
import pytest

from minecraft_scientist.exploration import visibility_acquisition as visibility


def test_rgb_callback_receives_real_next_controls_and_cannot_overrun_budget(
    monkeypatch,
):
    from test_visibility_acquisition import tracks

    supplied, received = [], []

    def execute(handle):
        received.append(handle)
        frame = np.full((224, 224, 3), len(received), dtype=np.uint8)
        supplied.append(frame)
        return visibility.PublicProbeStep(frame, handle, True)

    def track(frames, *, config):
        assert all(not frame.flags.writeable for frame in frames)
        assert int(frames[-1][0, 0, 0]) == len(received)
        return tracks(len(frames) - 1)

    monkeypatch.setattr(visibility, "track_frame_chain", track)
    initial = np.zeros((224, 224, 3), dtype=np.uint8)
    result = visibility.acquire_visibility_probe(initial, execute, (41, 8))
    assert received == [41] * 6 + [8] * 4
    assert result.terminal == "finished" and not result.failure
    assert (
        len(result.frames) == 11 and len(result.tracks) == len(result.decisions) == 10
    )
    supplied[-1][:] = 99
    assert int(result.frames[-1][0, 0, 0]) == 10 and initial.max() == 0
    assert result.decisions[-1].holdout_ticks == (7, 10)


@pytest.mark.parametrize(
    "failure_kind",
    ["wrong_handle", "failed", "wrong_ticks", "private_info", "exception"],
)
def test_control_failures_never_become_tracking_evidence(monkeypatch, failure_kind):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    monkeypatch.setattr(
        visibility,
        "track_frame_chain",
        lambda *a, **k: pytest.fail("failed control entered scientific evidence"),
    )

    def execute(handle):
        if failure_kind == "exception":
            raise RuntimeError("controller disconnected")
        if failure_kind == "private_info":
            return {"frame": image, "xpos": 12}
        return visibility.PublicProbeStep(
            image,
            handle + 1 if failure_kind == "wrong_handle" else handle,
            failure_kind != "failed",
            ticks=2 if failure_kind == "wrong_ticks" else 1,
        )

    result = visibility.acquire_visibility_probe(image, execute, (0, 1))
    assert result.terminal == "control_failed" and result.failure
    assert len(result.steps) == 1 and not result.tracks and not result.decisions


def test_tracking_failure_preserves_completed_actions_and_prefixes(monkeypatch):
    from test_visibility_acquisition import tracks

    image = np.zeros((224, 224, 3), dtype=np.uint8)

    def track(frames, *, config):
        if len(frames) == 3:
            raise RuntimeError("tracker failed")
        return tracks(1)

    monkeypatch.setattr(visibility, "track_frame_chain", track)
    result = visibility.acquire_visibility_probe(
        image, lambda handle: visibility.PublicProbeStep(image, handle, True), (0, 1)
    )
    assert result.terminal == "tracking_failed" and result.failure
    assert len(result.steps) == 2 and len(result.frames) == 3
    assert len(result.decisions) == len(result.tracks) == 1


def test_real_temporal_tracker_drives_bounded_public_control_stream():
    cv2 = pytest.importorskip("cv2")
    random = np.random.default_rng(98)
    gray = cv2.GaussianBlur(
        random.integers(0, 256, (224, 224), dtype=np.uint8), (5, 5), 0.8
    )
    initial = np.repeat(gray[:, :, None], 3, axis=2)
    displacement = np.zeros(2)
    actions = []

    def execute(handle):
        actions.append(handle)
        displacement[:] += (1, 0) if handle == 21 else (0, 1)
        frame = cv2.warpAffine(
            initial,
            np.array(((1.0, 0, displacement[0]), (0, 1.0, displacement[1]))),
            (224, 224),
        )
        return visibility.PublicProbeStep(frame, handle, True)

    result = visibility.acquire_visibility_probe(initial, execute, (21, 4))
    assert result.terminal == "finished" and result.failure is None
    switch = next(item for item in result.decisions if item.instruction == "switch")
    assert actions[: switch.tick] == [21] * switch.tick
    assert actions[switch.tick :] == [4] * (len(actions) - switch.tick)
    assert len(result.decisions[-1].retained_ids) >= 32
    assert len(result.frames) <= 11

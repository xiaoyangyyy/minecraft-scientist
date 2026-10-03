import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackDiagnosis,
    TrackRejection,
    diagnose_frame_chain,
    track_frame_chain,
)


def frames():
    gray = np.random.default_rng(71).integers(0, 256, (224, 224), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    return [
        cv2.warpAffine(image, np.float32([[1, 0, tick], [0, 1, 0]]), (224, 224))
        for tick in range(4)
    ]


@pytest.mark.parametrize("mode", ("normal", "blank_middle", "flat", "explicit"))
def test_diagnosis_preserves_tracking_and_accounts_for_every_original_seed(mode):
    images = frames()
    if mode == "blank_middle":
        images[2][:] = 0
    if mode == "flat":
        images = [np.zeros_like(images[0])] * 4
    kwargs = (
        {"seed_pixels": ((10, 10), (100, 100), (120, 120))}
        if mode == "explicit"
        else {}
    )
    original = [image.copy() for image in images]
    diagnosis = diagnose_frame_chain(images, **kwargs)
    assert diagnosis.tracks == track_frame_chain(images, **kwargs)
    rejected = [
        identity for event in diagnosis.rejections for identity in event.track_ids
    ]
    assert len(rejected) == len(set(rejected))
    assert set(rejected) | set(diagnosis.tracks.track_ids) == set(
        range(diagnosis.tracks.seed_count)
    )
    assert diagnosis == diagnose_frame_chain(images, **kwargs)
    for left, right in zip(original, images, strict=True):
        np.testing.assert_array_equal(left, right)
    if mode == "explicit":
        assert diagnosis.rejections[0] == TrackRejection(
            image_index=0, reason="seed_roi", track_ids=(0,)
        )


@pytest.mark.parametrize("forward", (True, False))
def test_unusable_flow_has_one_disjoint_status_reason(monkeypatch, forward):
    calls = 0

    def flow(left, right, points, initial, **kwargs):
        nonlocal calls
        calls += 1
        if forward or calls == 2:
            return None, None, None
        return points.copy(), np.ones((len(points), 1), dtype=np.uint8), None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    diagnosis = diagnose_frame_chain(frames(), seed_pixels=((100, 100), (110, 110)))
    assert diagnosis.tracks.track_ids == ()
    assert diagnosis.rejections == (
        TrackRejection(
            image_index=1,
            reason="forward_status" if forward else "reverse_status",
            track_ids=(0, 1),
        ),
    )
    assert calls == (1 if forward else 2)


@pytest.mark.parametrize(
    "replacement",
    (
        {"track_ids": ()},
        {"track_ids": (1, 1)},
        {"track_ids": (-1,)},
        {"track_ids": (True,)},
        {"track_ids": (2, 1)},
        {"reason": "hidden_pose"},
        {"image_index": True},
    ),
)
def test_rejection_schema_refuses_ambiguous_or_privileged_evidence(replacement):
    with pytest.raises(ValidationError):
        TrackRejection.model_validate(
            {"image_index": 1, "reason": "step_limit", "track_ids": (0,)} | replacement
        )


def test_ledger_refuses_missing_duplicate_surviving_and_misdated_losses():
    diagnosis = diagnose_frame_chain(frames(), seed_pixels=((10, 10), (100, 100)))
    for events in (
        (),
        diagnosis.rejections * 2,
        diagnosis.rejections
        + (TrackRejection(image_index=1, reason="patch_texture", track_ids=(1,)),),
        (TrackRejection(image_index=1, reason="seed_roi", track_ids=(0,)),),
        (TrackRejection(image_index=99, reason="seed_roi", track_ids=(0,)),),
    ):
        with pytest.raises(ValidationError):
            TemporalTrackDiagnosis(tracks=diagnosis.tracks, rejections=events)


def test_outside_roi_changes_neither_tracker_nor_loss_reasons():
    images = frames()
    altered = [
        np.random.default_rng(i).integers(0, 256, image.shape, dtype=np.uint8)
        for i, image in enumerate(images)
    ]
    for image, changed in zip(images, altered, strict=True):
        changed[44:170, 40:183] = image[44:170, 40:183]
    assert diagnose_frame_chain(images) == diagnose_frame_chain(altered)


@pytest.mark.parametrize(
    "reason",
    (
        "forward_status",
        "forward_roi",
        "step_limit",
        "reverse_status",
        "reverse_roi",
        "adjacent_cycle",
        "patch_texture",
        "patch_correlation",
        "chain_reverse_status",
        "chain_reverse_roi",
        "chain_cycle",
    ),
)
def test_each_pixel_gate_records_its_first_failure_without_changing_the_result(
    monkeypatch, reason
):
    calls = 0
    images = [frames()[0]] * 3

    def flow(left, right, points, initial, **kwargs):
        nonlocal calls
        calls += 1
        result = points.copy()
        status = np.ones((len(points), 1), dtype=np.uint8)
        if calls == 1:
            if reason == "forward_status":
                status[:] = 0
                result[:] = np.nan  # ROI also fails, but status is the first gate.
            elif reason == "forward_roi":
                result[:] = 10000
            elif reason == "step_limit":
                result[:, :, 0] += 30
        if calls == 2:
            if reason == "reverse_status":
                status[:] = 0
            elif reason == "reverse_roi":
                result[:] = 10000
            elif reason == "adjacent_cycle":
                result[:, :, 0] += 2
        if calls == 5:
            if reason == "chain_reverse_status":
                status[:] = 0
            elif reason == "chain_reverse_roi":
                result[:] = 10000
        if calls == 6 and reason == "chain_cycle":
            result[:, :, 0] += 2
        return result, status, None

    patches = 0

    def patch(image, size, point):
        nonlocal patches
        patches += 1
        if reason == "patch_texture":
            return np.zeros(size, dtype=np.uint8)
        pattern = np.arange(size[0] * size[1], dtype=np.float32).reshape(size)
        return (
            -pattern if reason == "patch_correlation" and patches % 2 == 0 else pattern
        )

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    monkeypatch.setattr(cv2, "getRectSubPix", patch)
    diagnosis = diagnose_frame_chain(images, seed_pixels=((100, 100),))
    assert diagnosis.tracks.track_ids == ()
    assert len(diagnosis.rejections) == 1
    assert diagnosis.rejections[0].reason == reason
    assert diagnosis.rejections[0].track_ids == (0,)
    calls = patches = 0
    assert diagnosis.tracks == track_frame_chain(images, seed_pixels=((100, 100),))

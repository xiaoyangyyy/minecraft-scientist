import hashlib
import json

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image", reason="requires optional visual extra")

from minecraft_scientist.evaluation.visual_evidence import verify_temporal_panorama
from minecraft_scientist.exploration.perception import (
    RGBEdgePanoramaEncoder,
    RGBEdgeSceneEncoder,
)


@pytest.fixture
def panorama(tmp_path):
    encoder = RGBEdgePanoramaEncoder(RGBEdgeSceneEncoder(grid_size=4))
    frames = [
        np.random.default_rng(index).integers(5, 250, (64, 64, 3), dtype=np.uint8)
        for index in range(3)
    ]
    metadata = {
        "view_count": 3,
        "temporal_sample_count": 3,
        "image_sha256": {},
        "temporal_sample_sha256": {},
    }
    for index, frame in enumerate(frames):
        name, role = f"view-{index:02d}.png", f"view-{index:02d}"
        Image.fromarray(frame).save(tmp_path / name)
        metadata["image_sha256"][name] = hashlib.sha256(
            (tmp_path / name).read_bytes()
        ).hexdigest()
        directory = tmp_path / "samples" / role
        directory.mkdir(parents=True)
        metadata["temporal_sample_sha256"][role] = {}
        for sample_index, offset in enumerate((-1, 0, 1)):
            sample_name = f"sample-{sample_index:02d}.png"
            Image.fromarray((frame.astype(int) + offset).astype(np.uint8)).save(
                directory / sample_name
            )
            metadata["temporal_sample_sha256"][role][sample_name] = hashlib.sha256(
                (directory / sample_name).read_bytes()
            ).hexdigest()
    signature = encoder.encode(frames)
    metadata["panorama_sha256"] = hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode()
    ).hexdigest()
    return tmp_path, metadata, encoder


def test_panorama_verification_needs_no_simulator_runtime(panorama):
    root, metadata, encoder = panorama
    signature, views, valid = verify_temporal_panorama(root, metadata, encoder, 3)
    assert valid and len(signature) == encoder.dimensions and len(views) == 3


def test_sample_hash_tampering_is_rejected(panorama):
    root, metadata, encoder = panorama
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
        root / "samples/view-00/sample-00.png"
    )
    assert not verify_temporal_panorama(root, metadata, encoder, 3)[2]


def test_updated_image_hash_cannot_hide_invalid_temporal_median(panorama):
    root, metadata, encoder = panorama
    with Image.open(root / "samples/view-00/sample-00.png") as image:
        image.save(root / "view-00.png")
    metadata["image_sha256"]["view-00.png"] = hashlib.sha256(
        (root / "view-00.png").read_bytes()
    ).hexdigest()
    assert not verify_temporal_panorama(root, metadata, encoder, 3)[2]


@pytest.mark.parametrize("count", [0, 2, 103])
def test_invalid_temporal_budget_is_rejected(panorama, count):
    root, metadata, encoder = panorama
    metadata["temporal_sample_count"] = count
    assert not verify_temporal_panorama(root, metadata, encoder, 3)[2]


def test_extra_view_and_reported_signature_tampering_are_rejected(panorama):
    root, metadata, encoder = panorama
    metadata["panorama_sha256"] = "0" * 64
    assert not verify_temporal_panorama(root, metadata, encoder, 3)[2]
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(root / "view-03.png")
    assert not verify_temporal_panorama(root, metadata, encoder, 3)[2]

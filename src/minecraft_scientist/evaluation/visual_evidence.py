"""Portable RGB panorama integrity checks, independent of simulator runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from minecraft_scientist.exploration.perception import temporal_median_rgb


def _rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_panorama(
    directory: Path, encoder: Any, period: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], list[np.ndarray]]:
    if not 1 <= period <= 128:
        raise ValueError("panorama view count must be in [1, 128]")
    frames = [_rgb(directory / f"view-{index:02d}.png") for index in range(period)]
    return (
        encoder.encode(frames),
        tuple(encoder.scene_encoder.encode(frame) for frame in frames),
        frames,
    )


def verify_temporal_panorama(
    directory: Path, metadata: dict[str, Any], encoder: Any, period: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], bool]:
    """Recompute raw image hashes, temporal medians and encoded signature hashes."""
    signature, views, frames = read_panorama(directory, encoder, period)
    expected_names = {f"view-{index:02d}.png" for index in range(period)}
    signature_hash = hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode()
    ).hexdigest()
    valid = (
        metadata["view_count"] == period
        and metadata["panorama_sha256"] == signature_hash
        and set(metadata["image_sha256"]) == expected_names
        and {path.name for path in directory.glob("view-*.png")} == expected_names
    )
    count = int(metadata["temporal_sample_count"])
    if count < 1 or count % 2 == 0 or count > 101:
        return signature, views, False
    expected_roles = {f"view-{index:02d}" for index in range(period)}
    valid &= set(metadata["temporal_sample_sha256"]) == expected_roles
    for index, frame in enumerate(frames):
        name = f"view-{index:02d}.png"
        role = f"view-{index:02d}"
        valid &= _hash(directory / name) == metadata["image_sha256"].get(name)
        samples_path = directory / "samples" / role
        expected_samples = {f"sample-{index:02d}.png" for index in range(count)}
        hashes = metadata["temporal_sample_sha256"].get(role, {})
        valid &= set(hashes) == expected_samples
        valid &= {
            path.name for path in samples_path.glob("sample-*.png")
        } == expected_samples
        samples = []
        for sample_index in range(count):
            sample_name = f"sample-{sample_index:02d}.png"
            path = samples_path / sample_name
            valid &= _hash(path) == hashes.get(sample_name)
            samples.append(_rgb(path))
        valid &= bool(np.array_equal(frame, temporal_median_rgb(samples)))
    return signature, views, bool(valid)

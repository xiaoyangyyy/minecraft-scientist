"""Audited RGB place-generalization gate over disjoint MineStudio episodes."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from minecraft_scientist.exploration import ExplorationMemory, RGBSceneEncoder

ROLE_PATHS = {
    "train_anchor": Path("train-positive-0/route/frame-00.png"),
    "train_positive": Path("train-positive-1/route/frame-00.png"),
    "train_negative": Path("train-negative/route/frame-00.png"),
    "heldout_positive": Path("heldout-positive/replay/frame-00.png"),
    "heldout_negative": Path("heldout-negative/replay/frame-00.png"),
}
SHIFTS = (-4, -2, 0, 2, 4)
BRIGHTNESS = (0.85, 1.0, 1.15)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "RGB evidence loading requires the optional visual dependency"
        ) from error
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _augment(frame: np.ndarray, shift: int, brightness: float) -> np.ndarray:
    array = np.asarray(frame, dtype=np.float64)
    if shift:
        amount = abs(shift)
        padded = np.pad(array, ((0, 0), (amount, amount), (0, 0)), mode="reflect")
        start = amount - shift
        array = padded[:, start : start + frame.shape[1]]
    return np.clip(array * brightness, 0, 255).astype(np.uint8)


def _distances(
    exemplars: tuple[tuple[int, ...], ...],
    signatures: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    return tuple(
        min(math.dist(signature, exemplar) for exemplar in exemplars)
        for signature in signatures
    )


def _load_evidence(root: Path) -> dict[str, np.ndarray]:
    missing = [str(path) for path in ROLE_PATHS.values() if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"RGB evidence is incomplete: {missing[0]}")
    return {role: _load_rgb(root / path) for role, path in ROLE_PATHS.items()}


def _evaluation_signatures(
    frames: dict[str, np.ndarray], encoder: RGBSceneEncoder
) -> dict[str, tuple[tuple[int, ...], ...]]:
    training_augmentations = tuple(
        encoder.encode(_augment(frames["train_anchor"], shift, brightness))
        for shift in SHIFTS
        for brightness in BRIGHTNESS
        if (shift, brightness) != (0, 1.0)
    )
    heldout_positives = tuple(
        encoder.encode(_augment(frames["heldout_positive"], shift, brightness))
        for shift in SHIFTS
        for brightness in BRIGHTNESS
    )
    heldout_negatives = tuple(
        encoder.encode(_augment(frames["heldout_negative"], shift, brightness))
        for shift in SHIFTS
        for brightness in BRIGHTNESS
    )
    return {
        "anchor": (encoder.encode(frames["train_anchor"]),),
        "training_positives": (
            encoder.encode(frames["train_positive"]),
            *training_augmentations,
        ),
        "training_negatives": (encoder.encode(frames["train_negative"]),),
        "heldout_positives": heldout_positives,
        "heldout_negatives": heldout_negatives,
    }


def run_rgb_generalization_benchmark(
    *,
    evidence_root: Path,
    output: Path,
    margin: float = 5,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit on training episodes and gate disjoint held-out RGB episodes."""

    if not math.isfinite(margin) or margin < 0:
        raise ValueError("RGB calibration margin must be finite and nonnegative")
    report_path = output / "report.json"
    audit_path = output / "integrity-audit.json"
    memory_path = output / "exploration.sqlite3"
    evidence_output = output / "evidence"
    known_files = (report_path, audit_path, memory_path)
    if any(path.exists() for path in known_files) and not overwrite:
        raise FileExistsError(f"RGB benchmark output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    evidence_output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in known_files:
            if path.exists():
                path.unlink()
        for suffix in ("-wal", "-shm"):
            candidate = Path(f"{memory_path}{suffix}")
            if candidate.exists():
                candidate.unlink()

    for role, relative in ROLE_PATHS.items():
        source = evidence_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"RGB evidence is missing: {source}")
        destination = evidence_output / f"{role}.png"
        if destination.exists() and not overwrite:
            raise FileExistsError(f"RGB evidence output already exists: {destination}")
        shutil.copy2(source, destination)

    frames = {role: _load_rgb(evidence_output / f"{role}.png") for role in ROLE_PATHS}
    encoder = RGBSceneEncoder()
    signatures = _evaluation_signatures(frames, encoder)
    with ExplorationMemory(memory_path, max_exemplars=4) as memory:
        place = memory.observe_place((0,), episode_id=0, step=0)
        memory.record_modality_view(
            encoder.VERSION,
            place.place_id,
            signatures["anchor"][0],
            episode_id=0,
            step=0,
        )
        calibrated = memory.calibrate_modality_place(
            encoder.VERSION,
            place.place_id,
            signatures["training_positives"],
            signatures["training_negatives"],
            episode_id=0,
            step=1,
            margin=margin,
        )
        heldout_positive_matches = tuple(
            memory.match_modality_place(encoder.VERSION, item) == place.place_id
            for item in signatures["heldout_positives"]
        )
        heldout_negative_rejections = tuple(
            memory.match_modality_place(encoder.VERSION, item) is None
            for item in signatures["heldout_negatives"]
        )
        summary = memory.summary()

    assert calibrated.calibration is not None
    heldout_positive_distances = _distances(
        calibrated.exemplars, signatures["heldout_positives"]
    )
    heldout_negative_distances = _distances(
        calibrated.exemplars, signatures["heldout_negatives"]
    )
    evidence_sha256 = {
        role: _digest(evidence_output / f"{role}.png") for role in ROLE_PATHS
    }
    report: dict[str, Any] = {
        "suite": "rgb_place_generalization",
        "encoder": encoder.VERSION,
        "descriptor_dimensions": encoder.dimensions,
        "training_roles": ("train_anchor", "train_positive", "train_negative"),
        "heldout_roles": ("heldout_positive", "heldout_negative"),
        "training_positive_count": len(signatures["training_positives"]),
        "training_negative_count": len(signatures["training_negatives"]),
        "heldout_positive_count": len(signatures["heldout_positives"]),
        "heldout_negative_count": len(signatures["heldout_negatives"]),
        "coreset_size": len(calibrated.exemplars),
        "calibration": calibrated.calibration.model_dump(mode="json"),
        "heldout_maximum_positive_distance": max(heldout_positive_distances),
        "heldout_nearest_negative_distance": min(heldout_negative_distances),
        "heldout_positive_acceptance_rate": sum(heldout_positive_matches)
        / len(heldout_positive_matches),
        "heldout_negative_rejection_rate": sum(heldout_negative_rejections)
        / len(heldout_negative_rejections),
        "evidence_sha256": evidence_sha256,
        "memory": summary,
        "scope": (
            "A hand-built RGB metric calibrated on disjoint real MineStudio fixture "
            "episodes plus deterministic photometric/translation perturbations; not "
            "a learned natural-scene embedding or open-world pixel SLAM."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = audit_rgb_generalization(output)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["valid"]:
        raise RuntimeError("RGB generalization benchmark failed its audit")
    return {**report, "integrity_audit_valid": True}


def audit_rgb_generalization(output: Path) -> dict[str, Any]:
    """Independently recompute held-out decisions from copied evidence and memory."""

    report_path = output / "report.json"
    memory_path = output / "exploration.sqlite3"
    evidence_root = output / "evidence"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frames = {role: _load_rgb(evidence_root / f"{role}.png") for role in ROLE_PATHS}
    encoder = RGBSceneEncoder()
    signatures = _evaluation_signatures(frames, encoder)
    with ExplorationMemory.open(memory_path) as memory:
        modality_places = memory.modality_places(encoder.VERSION)
        if len(modality_places) != 1:
            return {"valid": False, "reason": "unexpected modality place count"}
        state = modality_places[0]
        positive_matches = tuple(
            memory.match_modality_place(encoder.VERSION, item) == state.place_id
            for item in signatures["heldout_positives"]
        )
        negative_rejections = tuple(
            memory.match_modality_place(encoder.VERSION, item) is None
            for item in signatures["heldout_negatives"]
        )
        integrity = memory.verify_integrity()
    if state.calibration is None:
        return {"valid": False, "reason": "modality place is not calibrated"}
    positive_distances = _distances(state.exemplars, signatures["heldout_positives"])
    negative_distances = _distances(state.exemplars, signatures["heldout_negatives"])
    evidence_hashes = {
        role: _digest(evidence_root / f"{role}.png") for role in ROLE_PATHS
    }
    positive_rate = sum(positive_matches) / len(positive_matches)
    negative_rate = sum(negative_rejections) / len(negative_rejections)
    roles_disjoint = not set(report["training_roles"]) & set(report["heldout_roles"])
    calibration_valid = (
        state.calibration.model_dump(mode="json") == report["calibration"]
        and len(state.exemplars) == report["coreset_size"]
    )
    distance_metrics_valid = math.isclose(
        max(positive_distances),
        report["heldout_maximum_positive_distance"],
        abs_tol=1e-12,
    ) and math.isclose(
        min(negative_distances),
        report["heldout_nearest_negative_distance"],
        abs_tol=1e-12,
    )
    valid = (
        integrity
        and evidence_hashes == report["evidence_sha256"]
        and encoder.VERSION == report["encoder"]
        and encoder.dimensions == report["descriptor_dimensions"]
        and roles_disjoint
        and calibration_valid
        and distance_metrics_valid
        and math.isclose(
            positive_rate,
            report["heldout_positive_acceptance_rate"],
            abs_tol=1e-12,
        )
        and math.isclose(
            negative_rate,
            report["heldout_negative_rejection_rate"],
            abs_tol=1e-12,
        )
        and positive_rate == 1
        and negative_rate == 1
    )
    return {
        "valid": valid,
        "memory_integrity_valid": integrity,
        "evidence_hashes_valid": evidence_hashes == report["evidence_sha256"],
        "roles_disjoint": roles_disjoint,
        "calibration_valid": calibration_valid,
        "distance_metrics_valid": distance_metrics_valid,
        "heldout_positive_acceptance_rate": positive_rate,
        "heldout_negative_rejection_rate": negative_rate,
    }

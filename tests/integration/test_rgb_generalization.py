from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from minecraft_scientist.evaluation.rgb_generalization import (
    ROLE_PATHS,
    audit_rgb_generalization,
    run_rgb_generalization_benchmark,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None,
    reason="Pillow visual extra is unavailable",
)


def _frame(*, mirrored: bool = False, brightness: float = 1) -> np.ndarray:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[8:70, 40:88] = (8, 8, 8)
    frame[18:58, 54:74] = (205, 205, 205)
    if mirrored:
        frame[68:106, 16:56] = (180, 20, 20)
    else:
        frame[68:106, 68:108] = (180, 20, 20)
    return np.clip(frame.astype(float) * brightness, 0, 255).astype(np.uint8)


def test_rgb_benchmark_audit_recomputes_and_detects_report_tampering(
    tmp_path: Path,
) -> None:
    from PIL import Image

    evidence = tmp_path / "evidence"
    role_frames = {
        "train_anchor": _frame(),
        "train_positive": _frame(brightness=0.98),
        "train_negative": _frame(mirrored=True),
        "heldout_positive": _frame(brightness=1.02),
        "heldout_negative": _frame(mirrored=True, brightness=1.02),
    }
    for role, relative in ROLE_PATHS.items():
        path = evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(role_frames[role]).save(path)

    output = tmp_path / "run"
    report = run_rgb_generalization_benchmark(
        evidence_root=evidence,
        output=output,
        margin=5,
    )

    assert report["integrity_audit_valid"] is True
    assert report["heldout_positive_acceptance_rate"] == 1
    assert report["heldout_negative_rejection_rate"] == 1
    assert audit_rgb_generalization(output)["valid"] is True

    report_path = output / "report.json"
    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["calibration"]["radius"] += 1
    report_path.write_text(json.dumps(tampered), encoding="utf-8")

    audit = audit_rgb_generalization(output)
    assert audit["valid"] is False
    assert audit["calibration_valid"] is False

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PIL", reason="requires optional visual extra")


@pytest.fixture
def auditor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("audit_dense_correspondence")


def _acquisition():
    return {
        "timeline": [
            {
                "action": {"kind": "move"},
                "start_frame": index * 2,
                "end_frame": index * 2 + 1,
            }
            for index in range(4)
        ],
        "frame_count": 9,
        "gate": {"passed": True},
        "frame_sha256": {f"frame-{index:05d}.png": "0" * 64 for index in range(9)},
    }


def _ledgers(tmp_path, auditor):
    acquisition = _acquisition()
    reference = tmp_path / "goal-reference"
    reference.mkdir()
    for index in range(8):
        (reference / f"view-{index:02d}.png").write_bytes(b"fixture")
    config = {
        "frame_sha256": {
            f"frame-{pair[key]:05d}.png": "0" * 64
            for pair in auditor.public_pairs(acquisition)
            for key in ("first", "second")
        },
        "reference_sha256": {f"view-{index:02d}.png": "0" * 64 for index in range(8)},
    }
    return acquisition, config


def test_ledger_requires_every_selected_frame_and_reference(tmp_path, auditor):
    acquisition, config = _ledgers(tmp_path, auditor)
    assert all(auditor.input_ledger_checks(config, acquisition, tmp_path).values())
    config["frame_sha256"].pop("frame-00001.png")
    assert not auditor.input_ledger_checks(config, acquisition, tmp_path)[
        "complete_frame_ledger"
    ]
    config["reference_sha256"].clear()
    assert not auditor.input_ledger_checks(config, acquisition, tmp_path)[
        "complete_reference_ledger"
    ]


def test_outside_ledger_paths_are_rejected_before_model_or_hash_loading(
    tmp_path, auditor, monkeypatch
):
    acquisition, config = _ledgers(tmp_path, auditor)
    config["frame_sha256"] = {"../private.png": "0" * 64}
    (tmp_path / "report.json").write_text(json.dumps(acquisition), encoding="utf-8")
    development = tmp_path / "development"
    development.mkdir()
    config["source_sha256"] = {}
    (development / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (development / "report.json").write_text("{}", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("failed ledger must stop before backend/hash loading")

    monkeypatch.setattr(auditor, "_hash", forbidden)
    monkeypatch.setattr(auditor, "_opencv", forbidden)
    result = auditor.audit(tmp_path, development, tmp_path / "nonexistent.ckpt")
    assert not result["valid"] and result["evaluation"] is None


def _stub_run(tmp_path, auditor, monkeypatch):
    source_paths = auditor.runtime_sources()
    monkeypatch.setattr(auditor, "runtime_sources", lambda: source_paths)
    acquisition, config = _ledgers(tmp_path, auditor)
    pairs = auditor.public_pairs(acquisition)
    config.update(
        {
            "pairs": pairs,
            "source_sha256": {name: "0" * 64 for name in source_paths},
            "input_report_sha256": "0" * 64,
            "checkpoint_sha256": "0" * 64,
            "reference_focal_pixels": 90.0,
            "reference_focal_sample_count": 20,
            "reference_focal_relative_mad": 0.01,
            "camera_intrinsics": [[90.0, 0, 111.5], [0, 160.0, 111.5], [0, 0, 1]],
            "camera_resize": {
                "render_size": (640, 360),
                "observation_size": (224, 224),
            },
            "render_size_source": "explicit_client_configuration",
            "matcher": {},
            "motion": {},
            "evaluation_thresholds": {},
        }
    )
    matches = auditor.DenseCorrespondences(
        points0=(),
        points1=(),
        confidence=(),
        image_shape0=(224, 224),
        image_shape1=(224, 224),
        raw_match_count=0,
    )
    motion = SimpleNamespace(model_dump=lambda **kwargs: {"supported": False})
    backend = {"threads": 1, "checkpoint_sha256": "0" * 64}
    report = {
        "config_sha256": "0" * 64,
        "results": [
            {
                "pair": pair,
                "correspondences": matches.model_dump(mode="json"),
                "motion": motion.model_dump(),
            }
            for pair in pairs
        ],
        "failure": None,
        "sources_and_inputs_unchanged": True,
        "navigation_accepted": False,
        "backend": backend,
        "opencv_version": "stub",
    }
    development = tmp_path / "development"
    development.mkdir()
    for path, value in (
        (tmp_path / "report.json", acquisition),
        (development / "config.json", config),
        (development / "report.json", report),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(auditor, "_hash", lambda path: "0" * 64)
    monkeypatch.setattr(auditor, "_opencv", lambda: SimpleNamespace(__version__="stub"))
    monkeypatch.setattr(
        auditor, "_rgb", lambda path: np.zeros((224, 224, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(auditor, "_features", lambda *args: (np.empty((0, 2)), None))
    return development, matches, motion, backend


def test_rejected_focal_calibration_does_not_reach_model_or_evaluator(
    tmp_path, auditor, monkeypatch
):
    development, _, _, _ = _stub_run(tmp_path, auditor, monkeypatch)
    monkeypatch.setattr(auditor, "_calibrate_focal", lambda *args: (None, 0, None))

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected focal must stop before model loading")

    monkeypatch.setattr(auditor, "LocalLoFTRBackend", forbidden)
    result = auditor.audit(tmp_path, development, tmp_path / "weights.ckpt")
    assert not result["valid"] and result["evaluation"] is None
    assert result["checks"]["focal_calibration_recomputed"] is False


def test_misaligned_evaluator_is_rejected_before_scoring_or_indexing(
    tmp_path, auditor, monkeypatch
):
    development, matches, motion, backend = _stub_run(tmp_path, auditor, monkeypatch)
    monkeypatch.setattr(auditor, "_calibrate_focal", lambda *args: (90.0, 20, 0.01))
    monkeypatch.setattr(
        auditor,
        "LocalLoFTRBackend",
        lambda *args, **kwargs: SimpleNamespace(provenance=backend),
    )
    monkeypatch.setattr(
        auditor,
        "DenseRGBMatcher",
        lambda *args, **kwargs: SimpleNamespace(match=lambda *args: matches),
    )
    monkeypatch.setattr(
        auditor, "estimate_two_view_motion", lambda *args, **kwargs: motion
    )
    (tmp_path / "evaluator-audit.json").write_text(
        json.dumps({"positions": []}), encoding="utf-8"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("misaligned telemetry must stop before scoring")

    monkeypatch.setattr(auditor, "score_relative_motion", forbidden)
    result = auditor.audit(tmp_path, development, tmp_path / "weights.ckpt")
    assert not result["valid"] and result["evaluation"] is None
    assert result["checks"]["evaluator_frame_alignment"] is False

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PIL", reason="requires optional visual extra")


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("probe_temporal_motion"), importlib.import_module(
        "audit_temporal_motion"
    )


def _acquisition():
    return {
        "timeline": [
            {
                "action": {"kind": "move"},
                "start_frame": index * 5,
                "end_frame": index * 5 + 4,
            }
            for index in range(4)
        ],
        "frame_count": 21,
        "gate": {"passed": True},
        "frame_sha256": {f"frame-{index:05d}.png": "0" * 64 for index in range(21)},
    }


def _fixture(tmp_path, modules):
    producer, _ = modules
    acquisition = _acquisition()
    reference = tmp_path / "goal-reference"
    reference.mkdir()
    for index in range(8):
        (reference / f"view-{index:02d}.png").write_bytes(b"fixture")
    pairs = producer.temporal_pairs(acquisition, 3)
    config = {
        "window_frames": 3,
        "pairs": pairs,
        "frame_sha256": {
            f"frame-{slot:05d}.png": "0" * 64
            for pair in pairs
            for slot in producer.chain_indices(pair)
        },
        "reference_sha256": {f"view-{index:02d}.png": "0" * 64 for index in range(8)},
        "source_sha256": {name: "0" * 64 for name in producer.runtime_sources()},
        "reference_config_sha256": "0" * 64,
        "acquisition_sha256": "0" * 64,
        "temporal": {},
        "multiview": {},
        "opencv_version": "stub",
        "evaluation_thresholds": {},
        "reference_horizontal_focal_pixels": 90.0,
        "reference_focal_sample_count": 20,
        "reference_focal_relative_mad": 0.01,
        "camera_resize": {"render_size": (640, 360), "observation_size": (224, 224)},
        "render_size_source": "explicit_client_configuration",
        "camera_intrinsics": [[90.0, 0, 111.5], [0, 160.0, 111.5], [0, 0, 1]],
    }
    report = {
        "config_sha256": "0" * 64,
        "results": [
            {"pair": pair, "chain_indices": list(producer.chain_indices(pair))}
            for pair in pairs
        ],
        "failure": None,
        "sources_and_inputs_unchanged": True,
        "navigation_accepted": False,
        "evaluator_file_loaded": False,
    }
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    source = tmp_path / "reference-config.json"
    values = (
        (tmp_path / "report.json", acquisition),
        (experiment / "config.json", config),
        (experiment / "report.json", report),
        (
            source,
            {
                "evaluation_thresholds": {},
                "reference_sha256": config["reference_sha256"],
                "input_report_sha256": "0" * 64,
            },
        ),
    )
    for path, value in values:
        path.write_text(json.dumps(value), encoding="utf-8")
    return acquisition, config, experiment, source, report


def test_every_short_window_is_preserved_and_controls_are_not_expanded(modules):
    producer, _ = modules
    acquisition = _acquisition()
    full = producer.temporal_pairs(acquisition)
    windows = producer.temporal_pairs(acquisition, 3)
    assert len(full) == 6 and len(windows) == 14
    assert sum(pair["kind"] == "motion_intervention" for pair in windows) == 12
    assert [pair for pair in windows if pair["kind"] != "motion_intervention"] == [
        pair for pair in full if pair["kind"] != "motion_intervention"
    ]
    assert [(pair["first"], pair["second"]) for pair in windows[:3]] == [
        (0, 2),
        (1, 3),
        (2, 4),
    ]
    assert producer.chain_indices({"first": 7, "second": 7}) == (7, 7)
    for invalid in (1, 2, 33):
        with pytest.raises(ValueError):
            producer.temporal_pairs(acquisition, invalid)
    with pytest.raises(ValueError):
        producer.chain_indices({"first": 4, "second": 2})


def test_missing_intermediate_hash_is_rejected_before_model_or_telemetry(
    tmp_path, modules, monkeypatch
):
    _, auditor = modules
    _, config, experiment, reference, _ = _fixture(tmp_path, modules)
    config["frame_sha256"].pop("frame-00001.png")
    (experiment / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "failed ledger must stop before hash/model/evaluator access"
        )

    monkeypatch.setattr(auditor, "file_hash", forbidden)
    result = auditor.audit(tmp_path, experiment, reference)
    assert not result["valid"] and result["evaluation"] is None
    assert not result["checks"]["complete_frame_ledger"]


def _stub_public_replay(auditor, monkeypatch):
    monkeypatch.setattr(auditor, "file_hash", lambda *args: "0" * 64)
    monkeypatch.setattr(auditor, "_opencv", lambda: SimpleNamespace(__version__="stub"))
    monkeypatch.setattr(
        auditor, "frame", lambda *args: np.zeros((224, 224, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(auditor, "_features", lambda *args: (np.empty((0, 2)), None))


def test_failed_public_focal_stops_before_temporal_inference(
    tmp_path, modules, monkeypatch
):
    _, auditor = modules
    _, _, experiment, reference, _ = _fixture(tmp_path, modules)
    _stub_public_replay(auditor, monkeypatch)
    monkeypatch.setattr(auditor, "_calibrate_focal", lambda *args: (None, 0, None))

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected calibration must not infer geometry")

    monkeypatch.setattr(auditor, "infer_pair", forbidden)
    result = auditor.audit(tmp_path, experiment, reference)
    assert not result["valid"] and result["evaluation"] is None
    assert not result["checks"]["pixel_center_focal_recomputed"]


def test_wrong_isotropic_camera_is_rejected_before_tracks_or_evaluator(
    tmp_path, modules, monkeypatch
):
    _, auditor = modules
    _, config, experiment, reference, _ = _fixture(tmp_path, modules)
    config["camera_intrinsics"][1][1] = 90
    (experiment / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _stub_public_replay(auditor, monkeypatch)
    monkeypatch.setattr(auditor, "_calibrate_focal", lambda *args: (90.0, 20, 0.01))

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong resize camera must not infer geometry")

    monkeypatch.setattr(auditor, "infer_pair", forbidden)
    result = auditor.audit(tmp_path, experiment, reference)
    assert not result["valid"] and result["evaluation"] is None
    assert not result["checks"]["camera_recomputed"]


def test_misaligned_motion_telemetry_is_rejected_before_scoring(
    tmp_path, modules, monkeypatch
):
    _, auditor = modules
    _, _, experiment, reference, report = _fixture(tmp_path, modules)
    _stub_public_replay(auditor, monkeypatch)
    monkeypatch.setattr(auditor, "_calibrate_focal", lambda *args: (90.0, 20, 0.01))
    monkeypatch.setattr(
        auditor,
        "infer_pair",
        lambda route, pair, config, index: report["results"][index],
    )
    (tmp_path / "evaluator-audit.json").write_text(
        json.dumps({"positions": []}), encoding="utf-8"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("misaligned evaluator must not be scored")

    monkeypatch.setattr(auditor, "score_relative_motion", forbidden)
    result = auditor.audit(tmp_path, experiment, reference)
    assert not result["valid"] and result["evaluation"] is None
    assert not result["checks"]["evaluator_alignment"]

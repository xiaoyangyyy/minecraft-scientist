import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def variants(monkeypatch):
    pytest.importorskip("PIL.Image", reason="requires optional visual extra")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("rgb_motion_variants")


def test_stationary_context_is_past_only_and_motion_boundary_is_half_open(variants):
    pair = {"first": 20, "second": 30}
    timeline = [
        {"scheduled": {"kind": "move"}, "start_frame": 10, "end_frame": 18},
        {"scheduled": {"kind": "move"}, "start_frame": 20, "end_frame": 30},
    ]
    assert variants.context_indices(pair, timeline, 3) == (18, 19, 20)
    # A move ending at the first context image is allowed; a move that crosses
    # the actual context interval must be rejected regardless of second endpoint.
    timeline[0]["end_frame"] = 19
    with pytest.raises(ValueError, match="bounded past"):
        variants.context_indices(pair, timeline, 3)
    with pytest.raises(ValueError, match="bounded past"):
        variants.context_indices({"first": 1}, [], 3)


@pytest.mark.parametrize(
    "variant", ["static-context", "consensus", "static-consensus", "plane-competition"]
)
def test_registered_variants_do_not_change_camera_or_validation_thresholds(
    variants, monkeypatch, variant
):
    baseline = {
        "scope": "public fixture",
        "camera_intrinsics": [[2, 0, 1], [0, 3, 1], [0, 0, 1]],
        "multiview": {"two_view": {"minimum_parallax_degrees": 1.0}},
        "evaluation_thresholds": {"minimum_translation_blocks": 0.15},
    }
    monkeypatch.setattr(variants, "baseline_config", lambda *args: dict(baseline))
    config = variants.backend(variant).inference_config(None, None, None, 0)
    for key in ("camera_intrinsics", "multiview", "evaluation_thresholds"):
        assert config[key] == baseline[key]
    assert config["static_context"]["context_frames"] == 3
    assert config["variant"] == variant
    if variant == "plane-competition":
        assert config["plane_motion"]["minimum_off_plane_points"] == 8
        assert (
            config["endpoint_solver"]
            == "plane_essential_independent_branch_competition"
        )


def test_unregistered_variant_rejected_without_geometry_or_evaluator(
    variants, monkeypatch, tmp_path
):
    auditor = importlib.import_module("audit_controlled_rgb_motion")
    monkeypatch.setattr(auditor, "validate_acquisition", lambda path: (None, {}, {}))
    monkeypatch.setattr(
        auditor,
        "infer_pair",
        lambda *args: pytest.fail("invalid variant reached geometry"),
    )
    monkeypatch.setattr(
        auditor,
        "score_relative_motion",
        lambda *args: pytest.fail("invalid variant reached evaluator"),
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"variant": "privileged-runner-up"}), encoding="utf-8"
    )
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    result = auditor.audit(tmp_path, tmp_path)
    assert result == {
        "valid": False,
        "checks": {"registered_variant": False},
        "evaluation": None,
    }
    with pytest.raises(ValueError, match="unregistered"):
        variants.backend("private-pose")


def test_variant_sources_include_both_algorithms_and_registry(variants):
    sources = variants.runtime_sources()
    assert {
        "stationary_context.py",
        "consensus_motion.py",
        "rgb_motion_variants.py",
        "plane_motion.py",
    } <= set(sources)
    assert all(path.is_file() for path in sources.values())

# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
from test_controlled_motion_pipeline import acquisition, modules

from minecraft_scientist.exploration.keyframe_motion import (
    KeyframeAttempt,
    KeyframeMotionConfig,
    KeyframeSelection,
)
from minecraft_scientist.exploration.multiview_motion import MultiViewMotionEvidence
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks
from minecraft_scientist.exploration.two_view_motion import TwoViewMotionEvidence


@pytest.fixture
def pipeline(modules, monkeypatch):
    common = importlib.import_module("keyframe_motion_common")
    producer = importlib.import_module("probe_keyframe_rgb_motion")
    auditor = importlib.import_module("audit_keyframe_rgb_motion")

    def baseline(route, acquisition, report, seed):
        policy = KeyframeMotionConfig()
        return {
            "seed": seed,
            "pairs": report["pairs"],
            "scope": "pipeline fixture, not real imagery",
            "camera_intrinsics": [[90, 0, 111.5], [0, 160, 111.5], [0, 0, 1]],
            "temporal": policy.temporal.model_dump(mode="json"),
            "multiview": policy.multiview.model_dump(mode="json"),
            "evaluation_thresholds": {
                "minimum_translation_blocks": 0.15,
                "maximum_direction_error_degrees": 20,
                "maximum_rotation_error_degrees": 5,
            },
        }

    def rejected(frames, camera, *, config, seed):
        attempts = []
        for offset in config.offsets(len(frames) - 1):
            count = max(2, offset + 1)
            tracks = TemporalTracks(
                image_shape=(224, 224),
                frame_count=count,
                seed_count=0,
                surviving_counts=(0,) * count,
                track_ids=(),
                pixels=((),) * count,
                minimum_patch_correlations=(),
                chain_cycle_errors=(),
            )
            attempts.append(
                KeyframeAttempt(
                    end_offset=offset,
                    tracks=tracks,
                    motion=MultiViewMotionEvidence(
                        supported=False,
                        reason="pipeline fixture rejected",
                        candidate=TwoViewMotionEvidence(
                            supported=False, reason="fixture", matches=0
                        ),
                    ),
                )
            )
        return KeyframeSelection(
            terminal_offset=len(frames) - 1, attempts=tuple(attempts)
        )

    # Windows and WSL share pytest's assertion bytecode cache. Keep this fixture
    # mock's source identity native, without changing production source ledgers.
    rejected.__code__ = rejected.__code__.replace(
        co_filename=str(Path(__file__).resolve())
    )
    monkeypatch.setattr(common, "baseline_config", baseline)
    monkeypatch.setattr(common, "infer_keyframe_prefixes", rejected)
    return common, producer, auditor


@pytest.fixture
def experiment(acquisition, pipeline, tmp_path):
    route, report = acquisition
    _, producer, _ = pipeline
    output = tmp_path / "keyframes"
    result = producer.run(route, output)
    assert result["completed"] and result["input_sources_unchanged"]
    (route / "evaluator-audit.json").write_text(
        json.dumps(
            {
                "positions": [{"xpos": 0, "ypos": 0, "zpos": 0, "yaw": 0, "pitch": 0}]
                * report["frame_count"]
            }
        ),
        encoding="utf-8",
    )
    return route, output


def test_full_public_prefix_replay_precedes_first_private_read(
    pipeline, experiment, monkeypatch
):
    _, _, auditor = pipeline
    route, output = experiment
    replay, observed = [], []
    original_infer, original_read = auditor.infer_pair, Path.read_bytes

    def inference(*args):
        result = original_infer(*args)
        replay.append(result)
        return result

    def reading(path):
        if path.name == "evaluator-audit.json":
            assert len(replay) == 30
            observed.append(path)
        return original_read(path)

    monkeypatch.setattr(auditor, "infer_pair", inference)
    monkeypatch.setattr(Path, "read_bytes", reading)
    result = auditor.audit(route, output)
    assert result["valid"] and observed and len(replay) == 30
    assert all(result["checks"].values())
    assert result["evaluation"]["groups"]["settled"]["trials"] == 8
    assert not result["evaluation"]["navigation_accepted"]


@pytest.mark.parametrize(
    "tamper",
    [
        "drop_failed_prefix",
        "forged_selection",
        "drop_pair",
        "change_selector",
        "relax_gate",
        "source_snapshot",
    ],
)
def test_corrupt_or_tuned_prediction_cannot_reach_private_scoring(
    pipeline, experiment, monkeypatch, tamper
):
    _, _, auditor = pipeline
    route, output = experiment
    report_path, config_path = output / "report.json", output / "config.json"
    report = json.loads(report_path.read_bytes())
    config = json.loads(config_path.read_bytes())
    if tamper == "drop_failed_prefix":
        report["results"][0]["selection"]["attempts"].pop(0)
    elif tamper == "forged_selection":
        report["results"][0]["selection"]["selected_offset"] = 2
    elif tamper == "drop_pair":
        report["results"].pop()
    elif tamper == "source_snapshot":
        (output / "source-snapshot" / "keyframe_motion.py").write_text(
            "changed", encoding="utf-8"
        )
    else:
        if tamper == "change_selector":
            config["selector"] = "best_privileged_direction"
        else:
            config["keyframe"]["multiview"]["stability"][
                "maximum_direction_deviation"
            ] = 100
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report["config_sha256"] = auditor.digest(config_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        auditor,
        "infer_pair",
        lambda *args: pytest.fail("corrupt declaration reached geometry"),
    )
    original_read = Path.read_bytes

    def public_only(path):
        assert path.name != "evaluator-audit.json"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", public_only)
    result = auditor.audit(route, output)
    assert not result["valid"] and result["evaluation"] is None


def test_producer_does_not_require_evaluator_and_refuses_existing_output(
    acquisition, pipeline, tmp_path
):
    route, _ = acquisition
    _, producer, _ = pipeline
    assert not (route / "evaluator-audit.json").exists()
    output = tmp_path / "no-evaluator"
    result = producer.run(route, output)
    assert result["completed"] and not result["navigation_accepted"]
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        producer.run(route, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_inference_exception_preserves_partial_failure_without_dropping_budget(
    acquisition, pipeline, tmp_path, monkeypatch
):
    route, _ = acquisition
    _, producer, _ = pipeline
    original = producer.infer_pair

    def fail_second(*args):
        if args[-1] == 1:
            raise RuntimeError("injected second-prefix failure")
        return original(*args)

    monkeypatch.setattr(producer, "infer_pair", fail_second)
    result = producer.run(route, tmp_path / "partial")
    assert not result["completed"] and len(result["results"]) == 1
    assert result["failure"]["type"] == "RuntimeError"
    config = json.loads((tmp_path / "partial" / "config.json").read_bytes())
    assert len(config["pairs"]) == 30


def test_changed_replay_is_rejected_before_evaluator(pipeline, experiment, monkeypatch):
    _, _, auditor = pipeline
    route, output = experiment
    original = auditor.infer_pair

    def changed(*args):
        result = original(*args)
        result["selection"]["attempts"][0]["motion"]["reason"] = "forged replay"
        return result

    monkeypatch.setattr(auditor, "infer_pair", changed)
    original_read = Path.read_bytes

    def public_only(path):
        assert path.name != "evaluator-audit.json"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", public_only)
    result = auditor.audit(route, output)
    assert not result["valid"] and not result["checks"]["all_public_prefixes_replayed"]

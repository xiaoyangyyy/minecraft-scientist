import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_scientist.exploration.motion_probe_schedule import MotionProbeConfig


@pytest.fixture
def duration():
    root = Path(__file__).resolve().parents[2]
    path = root / "environments/minestudio/capture_motion_duration.py"
    spec = importlib.util.spec_from_file_location("duration_wrapper_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_capture(config, output, *, error=None, calls=1):
    root = Path(__file__).resolve().parents[2]
    capture = SimpleNamespace(
        __file__=str(root / "environments/minestudio/capture_controlled_rgb_motion.py"),
        MotionProbeConfig=MotionProbeConfig,
        source_paths=lambda: {
            "capture_controlled_rgb_motion.py": root
            / "environments/minestudio/capture_controlled_rgb_motion.py"
        },
    )

    def main():
        for _ in range(calls):
            actual = capture.MotionProbeConfig(
                seed=config.seed,
                observation_size=config.observation_size,
                time_of_day=config.time_of_day,
                weather=config.weather,
            )
            assert actual == config
        if error is not None:
            raise error
        output.mkdir()
        (output / "config.json").write_text(
            json.dumps({"config": config.model_dump(mode="json")}), encoding="utf-8"
        )
        (output / "report.json").write_text("{}", encoding="utf-8")

    capture.main = main
    return capture


def test_duration_hook_is_declared_once_and_restored(duration, tmp_path):
    config = MotionProbeConfig(movement_ticks=(6, 10))
    output = tmp_path / "capture"
    capture = _fake_capture(config, output)
    argv = sys.argv
    report = duration.run(capture, config, output)
    assert report["passed"] and all(report["checks"].values())
    assert report["constructor_calls"] == [config.model_dump(mode="json")]
    assert capture.MotionProbeConfig is MotionProbeConfig and sys.argv is argv
    assert (
        tmp_path
        / "capture.duration-profile"
        / "source-snapshot"
        / "capture_motion_duration.py"
    ).is_file()


@pytest.mark.parametrize("error", [RuntimeError("startup"), SystemExit(1)])
def test_duration_failure_preserves_manifest_and_restores_process(
    duration, tmp_path, error
):
    config = MotionProbeConfig(movement_ticks=(6, 10))
    output = tmp_path / "capture"
    capture = _fake_capture(config, output, error=error)
    argv = sys.argv
    report = duration.run(capture, config, output)
    assert not report["passed"] and report["failure"]["type"] == type(error).__name__
    assert capture.MotionProbeConfig is MotionProbeConfig and sys.argv is argv
    assert (tmp_path / "capture.duration-profile" / "report.json").is_file()


@pytest.mark.parametrize("calls", [0, 2])
def test_no_or_repeated_constructor_cannot_pass_profile(duration, tmp_path, calls):
    config, output = MotionProbeConfig(movement_ticks=(6, 10)), tmp_path / "capture"
    report = duration.run(_fake_capture(config, output, calls=calls), config, output)
    assert not report["passed"] and not report["checks"]["single_declared_constructor"]


@pytest.mark.parametrize("target", ["capture", "capture.duration-profile"])
def test_existing_duration_targets_are_preserved(duration, tmp_path, target):
    config, output = MotionProbeConfig(movement_ticks=(6, 10)), tmp_path / "capture"
    directory = tmp_path / target
    directory.mkdir()
    (directory / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        duration.run(_fake_capture(config, output), config, output)
    assert (directory / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_nested_duration_hooks_are_rejected_before_file_creation(duration, tmp_path):
    config, output = MotionProbeConfig(movement_ticks=(6, 10)), tmp_path / "capture"
    capture = _fake_capture(config, output)
    capture.MotionProbeConfig = lambda **kw: config
    with pytest.raises(ValueError, match="canonical"):
        duration.run(capture, config, output)
    assert not (tmp_path / "capture.duration-profile").exists()


@pytest.fixture
def duration_auditor(monkeypatch):
    pytest.importorskip("PIL.Image", reason="requires optional visual extra")
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return importlib.import_module("audit_motion_duration")


def test_partial_or_missing_duration_profile_fails_closed(duration_auditor, tmp_path):
    assert not duration_auditor.audit(tmp_path / "missing")["valid"]


@pytest.mark.parametrize("ticks", [(0, 10), (6, 6), (6, 11)])
def test_duration_budget_is_validated_without_starting_sdk(ticks):
    with pytest.raises(ValueError):
        MotionProbeConfig(movement_ticks=ticks)

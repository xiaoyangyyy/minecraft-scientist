import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")

from minecraft_scientist.exploration.compound_motion import CompoundProbeConfig
from minecraft_scientist.exploration.motion_probe_schedule import (
    MotionProbeConfig,
    motion_probe_pairs,
    motion_probe_schedule,
    motion_probe_timeline,
)


@pytest.fixture
def wrapper(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "active_capture_fixture",
        root / "environments/minestudio/capture_active_parallax.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "validate_plan",
        lambda directory: CompoundProbeConfig(observation_size=224),
    )
    plan = tmp_path / "plan"
    plan.mkdir()
    for name in ("config.json", "plan.json"):
        (plan / name).write_text("{}")
    return module, plan


def _capture(output, failure=None):
    module = SimpleNamespace(
        __file__="fixture.py",
        MotionProbeConfig=MotionProbeConfig,
        motion_probe_schedule=motion_probe_schedule,
        motion_probe_pairs=motion_probe_pairs,
        motion_probe_timeline=motion_probe_timeline,
        source_paths=lambda: {"fixture.py": Path(__file__)},
    )

    def main():
        config = module.MotionProbeConfig(
            seed=201, observation_size=224, time_of_day="noon", weather="clear"
        )
        if failure is not None:
            raise failure
        output.mkdir()
        # Constructor/restoration fixture only, never a valid simulator capture.
        for name in ("config.json", "report.json"):
            (output / name).write_text(json.dumps(config.model_dump(mode="json")))

    module.main = main
    return module


@pytest.mark.parametrize("failure", [None, RuntimeError("injected"), SystemExit(2)])
def test_active_constructor_and_aliases_restore_even_on_failure(
    wrapper, tmp_path, failure
):
    module, plan = wrapper
    output = tmp_path / "capture"
    capture = _capture(output, failure)
    before, argv = vars(capture).copy(), sys.argv
    result = module.run(capture, plan, output)
    assert vars(capture) == before and sys.argv is argv
    assert result["checks"]["process_settings_restored"]
    assert result["checks"]["single_declared_constructor"]
    assert result["passed"] == (failure is None)
    assert (tmp_path / "capture.active-profile" / "report.json").is_file()


def test_existing_active_outputs_refuse_before_plan_replay(
    wrapper, tmp_path, monkeypatch
):
    module, plan = wrapper
    output = tmp_path / "capture"
    output.mkdir()
    monkeypatch.setattr(
        module,
        "validate_plan",
        lambda directory: pytest.fail("existing output entered replay"),
    )
    with pytest.raises(FileExistsError):
        module.run(_capture(output), plan, output)

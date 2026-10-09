"""Runner bounds, refusal to overwrite and deterministic positive/negative rows."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_adaptive_audit_development.py"
    )
    spec = importlib.util.spec_from_file_location("adaptive_audit_runner_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seeds": 0},
        {"maximum_states": 1},
        {"budget": True},
        {"horizon": 9},
    ],
)
def test_invalid_protocol_is_rejected_before_output_creation(runner, tmp_path, kwargs):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="bounded development grid"):
        runner.run(output=output, **kwargs)
    assert not output.exists()


def test_existing_directory_is_never_overwritten(runner, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        runner.run(output=output, seeds=1, maximum_states=2)
    assert list(output.iterdir()) == []


def test_small_grid_replays_byte_exactly_with_all_private_audits(runner, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    report = runner.run(output=first, seeds=1, maximum_states=2)
    assert report == runner.run(output=second, seeds=1, maximum_states=2)
    assert report["summary"]["runs"] == 4 and report["all_independent_audits_valid"]
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert "timing.json" not in manifest["sha256"]
    assert len(manifest["sha256"]) == 22
    for path in manifest["sha256"]:
        assert (first / path).read_bytes() == (second / path).read_bytes()
    for audit_path in first.glob("seed-*/*/states-*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert (
            audit["raw_public_ledger_valid"]
            and audit["public_replay_valid"]
            and audit["private_truth_opened"]
        )


def test_source_drift_retains_partial_run_without_manifest(
    runner, tmp_path, monkeypatch
):
    calls = 0

    def drift():
        nonlocal calls
        calls += 1
        return {"declared-source": str(calls)}

    monkeypatch.setattr(runner, "source_hashes", drift)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Source drift"):
        runner.run(output=output, seeds=1, maximum_states=2)
    assert (output / "registration.json").exists()
    assert not (output / "manifest.json").exists()

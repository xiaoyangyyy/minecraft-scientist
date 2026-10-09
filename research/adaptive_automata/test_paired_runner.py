"""Paired runner protocol guards, audits, source drift and byte reproduction."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_paired_transfer_development.py"
    )
    spec = importlib.util.spec_from_file_location(
        "paired_transfer_runner_fixture", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "kwargs", [{"budget": True}, {"old_budget": 0}, {"horizon": 9}, {"seeds": 0}]
)
def test_invalid_paired_protocol_never_creates_output(runner, tmp_path, kwargs):
    directory = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="paired-transfer development"):
        runner.run(output=directory, **kwargs)
    assert not directory.exists()


def test_existing_output_is_not_overwritten(runner, tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        runner.run(output=directory)
    assert list(directory.iterdir()) == []


def test_small_pair_grid_is_deterministic_with_paid_old_history_and_complete_audits(
    runner, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    report = runner.run(output=first, seeds=1, maximum_states=2)
    assert report == runner.run(output=second, seeds=1, maximum_states=2)
    assert (
        report["all_campaign_audits_valid"]
        and len(report["pairs"]) == 4
        and len(report["rows"]) == 8
    )
    assert all(p["same_paid_old_history"] for p in report["pairs"])
    assert all(
        r["spent"] == r["old_spent"] + r["new_spent"] and r["old_spent"] > 0
        for r in report["rows"]
    )
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["sha256"]) == 58 and "timing.json" not in manifest["sha256"]
    for path in manifest["sha256"]:
        assert (first / path).read_bytes() == (second / path).read_bytes()
    for path in first.glob("seed-*/*/states-*/*/audit.json"):
        audit = json.loads(path.read_text(encoding="utf-8"))
        assert audit["valid"] and audit["all_contexts_public_valid"]


def test_paired_source_drift_retains_data_but_no_final_manifest(
    runner, tmp_path, monkeypatch
):
    calls = 0

    def drift():
        nonlocal calls
        calls += 1
        return {"fixture-source": str(calls)}

    monkeypatch.setattr(runner, "hashes", drift)
    directory = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Source drift"):
        runner.run(output=directory, seeds=1, maximum_states=2)
    assert (directory / "registration.json").exists() and not (
        directory / "manifest.json"
    ).exists()

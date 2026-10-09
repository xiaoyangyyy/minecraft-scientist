"""Paid grid source binding, deterministic artifacts and safe partial failures."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def online_runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_online_automata_development.py"
    )
    spec = importlib.util.spec_from_file_location(
        "online_development_runner_fixture", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "protocol",
    [
        {"budget": True},
        {"old_budget": 0},
        {"suite": "unknown"},
        {"seed": -1},
        {"horizon": 9},
    ],
)
def test_invalid_grid_never_creates_output(online_runner, tmp_path, protocol):
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="online development runner"):
        online_runner.run(output=output, **protocol)
    assert not output.exists()


def test_existing_grid_not_overwritten(online_runner, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        online_runner.run(output=output)
    assert not list(output.iterdir())


def test_controlled_grid_replays_all_artifacts_and_preserves_negative_results(
    online_runner, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    report = online_runner.run(output=first, seeds=1, suite="controlled")
    assert report == online_runner.run(output=second, seeds=1, suite="controlled")
    assert (
        report["all_campaign_audits_valid"]
        and len(report["pairs"]) == 7
        and len(report["rows"]) == 14
    )
    assert all(p["same_paid_bootstrap_and_monitoring"] for p in report["pairs"])
    for row in report["rows"]:
        assert (
            row["spent"]
            == row["old_spent"] + row["monitor_spent"] + row["repair_spent"]
        )
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["sha256"]) == 86 and "timing.json" not in manifest["sha256"]
    assert all(
        (first / path).read_bytes() == (second / path).read_bytes()
        for path in manifest["sha256"]
    )


def test_changed_source_keeps_partial_data_without_certified_manifest(
    online_runner, monkeypatch, tmp_path
):
    count = 0

    def drift():
        nonlocal count
        count += 1
        return {"fixture": str(count)}

    monkeypatch.setattr(online_runner, "hashes", drift)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Source drift"):
        online_runner.run(output=output, seeds=1, suite="controlled", budget=1)
    assert (output / "registration.json").exists() and not (
        output / "manifest.json"
    ).exists()

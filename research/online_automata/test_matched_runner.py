"""Frozen knowledge conditions, paired artifacts and explicit partial failures."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_matched_h3_development.py"
    )
    spec = importlib.util.spec_from_file_location("matched_h3_runner_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "protocol",
    [
        {"seed": True},
        {"budget": 0},
        {"horizon": 6},
        {"boundaries": (25, 25)},
        {"boundaries": (0,)},
        {"seeds": 0},
    ],
)
def test_invalid_protocol_never_creates_output(runner, tmp_path, protocol):
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="matched H3 development"):
        runner.run(output=output, **protocol)
    assert not output.exists()


def test_existing_output_is_preserved(runner, tmp_path):
    with pytest.raises(FileExistsError):
        runner.run(output=tmp_path)


def test_four_methods_share_complete_late_window_and_deterministic_artifacts(
    runner, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    report = runner.run(output=first, seeds=1, boundaries=(100,))
    assert report == runner.run(output=second, seeds=1, boundaries=(100,))
    assert len(report["rows"]) == 28 and len(report["comparisons"]) == 21
    assert all(
        r["status"] == "bounded_monitoring_completed_no_anomaly" for r in report["rows"]
    )
    assert all(
        p["same_paid_prelude_and_monitoring"] and p["same_final_protocol"]
        for p in report["comparisons"]
    )
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["sha256"]) == 170
    assert all(
        (first / path).read_bytes() == (second / path).read_bytes()
        for path in manifest["sha256"]
    )
    for row in report["rows"]:
        assert (
            row["spent"]
            == row["old_spent"] + row["monitor_spent"] + row["repair_spent"]
        )
        assert (
            row["reset_count"] > 0 and row["atomic_action_count"] > row["reset_count"]
        )


def test_source_drift_retains_uncertified_partial_evidence(
    runner, monkeypatch, tmp_path
):
    count = 0

    def drift():
        nonlocal count
        count += 1
        return {"fixture": str(count)}

    monkeypatch.setattr(runner, "hashes", drift)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Source drift"):
        runner.run(output=output, seeds=1, budget=1, boundaries=(100,))
    assert (output / "registration.json").exists()
    assert not (output / "manifest.json").exists()

"""Actual structural inputs, artifact integrity and honest negative outcomes."""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from minecraft_scientist.cli import app
from minecraft_scientist.evaluation.active_automata import run_automata_benchmark
from minecraft_scientist.world.active_automata import AUTOMATA_FAMILIES, automata_task


@pytest.mark.parametrize("family", AUTOMATA_FAMILIES)
def test_family_changes_do_not_leak_through_public_names(family):
    small = automata_task(0, family, 2)
    large = automata_task(0, family, 5)
    other = automata_task(0, "dense", 2)
    assert small.actions == large.actions == other.actions
    assert small.outputs == large.outputs == other.outputs
    assert (
        small.view.begin_trial() == large.view.begin_trial() == other.view.begin_trial()
    )
    assert small.ground_truth != large.ground_truth
    assert small.ground_truth == automata_task(0, family, 2).ground_truth


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seeds": 0},
        {"seeds": 65},
        {"budget": False},
        {"horizon": 9},
        {"maximum_states": 1},
        {"conformance_count": 0},
        {"operation_budget": 0},
    ],
)
def test_invalid_protocol_rejected_before_creating_output(tmp_path, kwargs):
    destination = tmp_path / "absent"
    with pytest.raises(ValueError):
        run_automata_benchmark(output=destination, **kwargs)
    assert not destination.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": 1},
        {"operation_budget": 1},
        {},
    ],
)
def test_all_cases_retained_with_frozen_artifacts_and_no_false_promotion(
    tmp_path, kwargs
):
    destination = tmp_path / "first"
    report = run_automata_benchmark(
        output=destination, seeds=1, maximum_states=2, **kwargs
    )
    assert report["all_independent_audits_valid"]
    assert len(report["rows"]) == 4
    assert report["summary"]["runs"] == 4
    if kwargs:
        assert report["summary"]["unresolved"] == 4
        assert report["summary"]["provisional_conformance_passed"] == 0
    for row in report["rows"]:
        assert row["spent"] <= report["registration"]["protocol"]["budget"]
        assert row["evaluation"]["actual_controls_match_private_reference"]
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    assert manifest["nondeterministic_artifacts"] == ["timing.json"]
    assert set(manifest["sha256"]) | {"manifest.json", "timing.json"} == {
        p.relative_to(destination).as_posix()
        for p in destination.rglob("*")
        if p.is_file()
    }
    repeated = tmp_path / "repeat"
    assert (
        run_automata_benchmark(output=repeated, seeds=1, maximum_states=2, **kwargs)
        == report
    )
    for name in manifest["sha256"]:
        assert (destination / name).read_bytes() == (repeated / name).read_bytes()
    with pytest.raises(FileExistsError):
        run_automata_benchmark(output=destination)


def test_cli_invokes_the_real_baseline_and_keeps_budget_negatives(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "benchmark-automata",
            "--output",
            str(tmp_path / "cli"),
            "--seeds",
            "1",
            "--maximum-states",
            "2",
            "--budget",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "cli" / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["unresolved"] == 4
    assert "not full Adaptive L#" in report["scope"]

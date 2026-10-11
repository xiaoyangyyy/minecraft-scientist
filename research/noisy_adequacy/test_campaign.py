"""Full matrix plumbing tests, NOT substitutes for budget-96 scientific runs."""

import hashlib

import pytest

from research.compact_campaigns.store import Reader, Writer, decoded

from . import campaign
from .monitor import FamilyAdequacyMonitor


def test_complete_matrix_and_all_public_before_any_private(tmp_path, monkeypatch):
    checked = []
    actual_public, actual_write_read = campaign.public_checks, Writer.read

    def public(*args, **kwargs):
        result = actual_public(*args, **kwargs)
        checked.append(True)
        return result

    def writer_read(self, name):
        if name.endswith("/private.json"):
            assert len(checked) == 30
        return actual_write_read(self, name)

    monkeypatch.setattr(campaign, "public_checks", public)
    monkeypatch.setattr(Writer, "read", writer_read)
    output = tmp_path / "matrix"
    result = campaign.run(output=output, budget=1, horizon=1)
    assert len(result["rows"]) == 30
    assert all(row["theory_revised"] is False for row in result["rows"])
    assert result["accepted_theories"] is None
    pin = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    checked.clear()
    actual_read = Reader.read

    def reader_read(self, name):
        if name.endswith("/private.json"):
            assert len(checked) == 30
        return actual_read(self, name)

    monkeypatch.setattr(Reader, "read", reader_read)
    verified = campaign.verify_output(output, manifest_sha256=pin)
    assert verified["valid"] and verified["runs"] == 30
    assert verified["logical_artifacts"] == 152


def test_source_drift_keeps_all_actual_receipts_without_certificate(
    tmp_path, monkeypatch
):
    original, calls = campaign.hashes, []

    def drifting():
        calls.append(True)
        sources = original()
        return sources if len(calls) == 1 else {**sources, "injected": "f" * 64}

    monkeypatch.setattr(campaign, "hashes", drifting)
    output = tmp_path / "drift"
    with pytest.raises(ValueError, match="source drift"):
        campaign.run(output=output, budget=1, horizon=1)
    assert not (output / "manifest.json").exists()
    assert len(list(output.glob("seed-*/*/*/raw-trace.json.gz"))) == 30
    assert (output / "failure.json.gz").exists()


def test_actual_agent_and_monitor_timing_excludes_storage(tmp_path, monkeypatch):
    clock = [0.0]
    actual_agent, actual_write = campaign.run_monitored_agent, Writer.write

    def agent(*args, **kwargs):
        result = actual_agent(*args, **kwargs)
        clock[0] += 2.0
        return result

    def write(*args, **kwargs):
        clock[0] += 100.0
        return actual_write(*args, **kwargs)

    monkeypatch.setattr(campaign, "run_monitored_agent", agent)
    monkeypatch.setattr(Writer, "write", write)
    monkeypatch.setattr(campaign, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(campaign, "process_time", lambda: clock[0])
    output = tmp_path / "timing"
    campaign.run(output=output, budget=1, horizon=1)
    timing = decoded((output / "timing.json.gz").read_bytes(), 1_000_000)[1]
    assert len(timing["rows"]) == 30
    assert all(
        row["actual_agent_and_online_monitor"]
        == {"wall_seconds": 2.0, "cpu_seconds": 2.0}
        for row in timing["rows"]
    )


def test_post_dispatch_monitor_failure_preserves_both_ledgers_and_actual_step(
    tmp_path, monkeypatch
):
    def fail(*args, **kwargs):
        raise ValueError("injected likelihood failure after actual step")

    monkeypatch.setattr(FamilyAdequacyMonitor, "finish", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected"):
        campaign.run(output=output, budget=3, horizon=1)
    assert not (output / "manifest.json").exists()

    def read(name):
        path = next(output.glob(f"seed-*/*/*/{name}.json.gz"))
        return decoded(path.read_bytes(), 10_000_000)[1]

    shadow = read("partial-monitor-events")
    raw = read("partial-raw-trace")
    assert len(raw) == 2  # actual reset plus the command already dispatched
    assert shadow[-1]["kind"] == "step" and "transition" in shadow[-1]
    assert "prediction_before_dispatch" in shadow[-1] and "after" not in shadow[-1]
    assert read("partial-agent-events")  # core cannot falsely return this failed step
    assert read("partial-private")


@pytest.mark.parametrize("bad", [True, -1, 2**64 - 1])
def test_invalid_seed_does_not_create_output(tmp_path, bad):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        campaign.run(output=output, seed=bad)
    assert not output.exists()


@pytest.mark.parametrize("field", ["budget", "horizon"])
def test_boolean_budget_and_horizon_do_not_create_output(tmp_path, field):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        campaign.run(output=output, **{field: True})
    assert not output.exists()


def test_source_drift_during_independent_verification_cannot_pass(
    tmp_path, monkeypatch
):
    output = tmp_path / "matrix"
    campaign.run(output=output, budget=1, horizon=1)
    pin = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    original, calls = campaign.hashes, []

    def drifting():
        calls.append(True)
        sources = original()
        return sources if len(calls) == 1 else {**sources, "injected": "f" * 64}

    monkeypatch.setattr(campaign, "hashes", drifting)
    with pytest.raises(ValueError, match="source drift during independent"):
        campaign.verify_output(output, manifest_sha256=pin)
    assert len(calls) == 2

"""Complete matrix/audit barriers using minimal-budget pipeline fixtures.

Budget-one fixtures test plumbing only, NOT the default budget-96 learning result.
Actual nontrivial Agent trajectories are tested separately and the CLI matrix
must be run at its full declared budget before scientific claims.
"""

import hashlib

import pytest

from research.compact_campaigns.store import Reader, decoded

from . import campaign
from .fixture import CONDITIONS
from .planning import POLICIES


def test_complete_matrix_pipeline_and_public_before_any_private_barrier(
    tmp_path, monkeypatch
):
    output = tmp_path / "matrix"
    campaign.run(output=output, seed=0, budget=1, horizon=1)
    pin = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    checked = []
    actual_public, actual_read = campaign.verify_public, Reader.read

    def public(*args, **kwargs):
        result = actual_public(*args, **kwargs)
        checked.append(True)
        return result

    def read(self, name):
        if name.endswith("/private.json"):
            assert len(checked) == 2 * len(CONDITIONS) * len(POLICIES) == 30
        return actual_read(self, name)

    monkeypatch.setattr(campaign, "verify_public", public)
    monkeypatch.setattr(Reader, "read", read)
    result = campaign.verify_output(output, manifest_sha256=pin)
    assert (
        result["valid"] and result["runs"] == 30 and result["logical_artifacts"] == 152
    )


def test_source_drift_preserves_whole_matrix_but_never_finishes(tmp_path, monkeypatch):
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


def test_actual_policy_timing_excludes_storage(tmp_path, monkeypatch):
    clock = [0.0]
    actual_agent, actual_write = campaign.run_agent, campaign.Writer.write

    def agent(*args, **kwargs):
        result = actual_agent(*args, **kwargs)
        clock[0] += 2.0
        return result

    def write(*args, **kwargs):
        clock[0] += 100.0
        return actual_write(*args, **kwargs)

    monkeypatch.setattr(campaign, "run_agent", agent)
    monkeypatch.setattr(campaign.Writer, "write", write)
    monkeypatch.setattr(campaign, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(campaign, "process_time", lambda: clock[0])
    output = tmp_path / "timing"
    campaign.run(output=output, budget=1, horizon=1)
    timing = decoded((output / "timing.json.gz").read_bytes(), 1_000_000)[1]
    assert len(timing["rows"]) == 30
    assert all(
        row["actual_policy"] == {"wall_seconds": 2.0, "cpu_seconds": 2.0}
        for row in timing["rows"]
    )


def test_action_failure_preserves_partial_receipts_without_certificate(
    tmp_path, monkeypatch
):
    actual = campaign.run_agent

    def fail(*args, **kwargs):
        actual(*args, **kwargs)
        raise ValueError("injected after actual reset")

    monkeypatch.setattr(campaign, "run_agent", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected"):
        campaign.run(output=output, budget=1, horizon=1)
    assert not (output / "manifest.json").exists()
    raw = next(output.glob("seed-*/*/*/partial-raw-trace.json.gz"))
    assert len(decoded(raw.read_bytes(), 1_000_000)[1]) == 1


@pytest.mark.parametrize("bad", [True, -1, 2**64 - 1])
def test_invalid_seed_has_no_output(tmp_path, bad):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        campaign.run(output=output, seed=bad)
    assert not output.exists()

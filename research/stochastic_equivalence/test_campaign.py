"""Complete all-pair public proof pipeline, not physical experimentation."""

import hashlib

import pytest

from research.compact_campaigns.store import decoded

from . import campaign


def test_complete_matrix_independent_check_and_storage_excluded_timing(
    tmp_path, monkeypatch
):
    clock = [0.0]
    actual_compare, actual_write = campaign.compare, campaign.Writer.write

    def compare(*args, **kwargs):
        result = actual_compare(*args, **kwargs)
        clock[0] += 2.0
        return result

    def write(*args, **kwargs):
        clock[0] += 100.0
        return actual_write(*args, **kwargs)

    monkeypatch.setattr(campaign, "compare", compare)
    monkeypatch.setattr(campaign.Writer, "write", write)
    monkeypatch.setattr(campaign, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(campaign, "process_time", lambda: clock[0])
    output = tmp_path / "all"
    report = campaign.run(output=output)
    assert len(report["rows"]) == 2
    assert all(r["pairs"] == 153 for r in report["rows"])
    assert report["physical_trials"] == 0
    pin = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    audit = campaign.verify_output(output, manifest_sha256=pin)
    assert audit["valid"] and audit["logical_artifacts"] == 616
    timing = decoded((output / "timing.json.gz").read_bytes(), 1_000_000)[1]
    assert len(timing["rows"]) == 306
    assert all(
        r["actual_search"] == {"wall_seconds": 2.0, "cpu_seconds": 2.0}
        for r in timing["rows"]
    )


def test_source_drift_retains_all_proofs_without_completion(tmp_path, monkeypatch):
    original, calls = campaign.hashes, []

    def drift():
        calls.append(True)
        value = original()
        return value if len(calls) == 1 else {**value, "injected": "f" * 64}

    monkeypatch.setattr(campaign, "hashes", drift)
    output = tmp_path / "drift"
    with pytest.raises(ValueError, match="source drift"):
        campaign.run(output=output)
    assert len(list(output.glob("seed-*/pair-*/certificate.json.gz"))) == 306
    assert len(list(output.glob("seed-*/pair-*/audit.json.gz"))) == 306
    assert (output / "failure.json.gz").exists()
    assert not (output / "manifest.json").exists()


def test_independent_check_failure_preserves_actual_proof_without_certificate(
    tmp_path, monkeypatch
):
    def failed(*args, **kwargs):
        raise ValueError("injected independent audit failure")

    monkeypatch.setattr(campaign, "verify_certificate", failed)
    output = tmp_path / "audit-failed"
    with pytest.raises(ValueError, match="injected"):
        campaign.run(output=output)
    assert len(list(output.glob("seed-*/pair-*/certificate.json.gz"))) == 1
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("bad", [True, -1, 2**64 - 1])
def test_invalid_seed_cannot_create_output(tmp_path, bad):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        campaign.run(output=output, seed=bad)
    assert not output.exists()

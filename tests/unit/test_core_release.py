import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("verify_core_release")


def test_real_repeat_preserves_all_artifacts_and_both_crises(verifier, tmp_path):
    directory = tmp_path / "release"
    result = verifier.verify_release(output=directory, seed=3, seeds=1)
    assert result["passed"] and result["failure"] is None
    row = result["runs"][0]
    assert row["seed"] == 3 and row["identical_artifacts"]
    assert len(row["artifact_sha256"][0]) == 18
    assert row["artifact_sha256"][0] == row["artifact_sha256"][1]
    assert row["checks"]["equation_revision"]
    assert row["checks"]["full_persisted_evidence"]
    assert row["checks"]["read_only_experience_replay"]
    assert row["checks"]["executable_symbol_transfer_replayed"]
    assert not list(directory.rglob("*.sqlite3-wal"))
    assert not list(directory.rglob("*.sqlite3-shm"))
    assert row["metrics"]["law_recovery"] == 1.0


@pytest.mark.parametrize(
    "kwargs", [{"seed": -1}, {"seeds": 0}, {"seeds": True}, {"intervention_budget": 0}]
)
def test_invalid_release_budget_creates_no_output(verifier, tmp_path, kwargs):
    directory = tmp_path / "release"
    with pytest.raises(ValueError):
        verifier.verify_release(output=directory, **kwargs)
    assert not directory.exists()


def test_existing_release_is_never_overwritten(verifier, tmp_path):
    directory = tmp_path / "release"
    directory.mkdir()
    (directory / "report.json").write_bytes(b"owned evidence")
    with pytest.raises(FileExistsError):
        verifier.verify_release(output=directory, seeds=1)
    assert (directory / "report.json").read_bytes() == b"owned evidence"


def test_partial_failure_is_retained_and_cannot_pass_release(verifier, tmp_path):
    directory = tmp_path / "release"
    result = verifier.verify_release(output=directory, seeds=2, intervention_budget=1)
    assert not result["passed"] and result["failure"] is not None
    assert not result["checks"]["all_requested_seeds_complete"]
    assert json.loads((directory / "report.json").read_bytes()) == result
    assert (directory / "source-snapshot").is_dir()


def test_forged_equation_history_fails_persisted_check(verifier, tmp_path):
    directory = tmp_path / "full"
    payload = verifier.run_full_benchmark(
        seed=4, intervention_budget=80, output=directory, overwrite=False
    )
    original = json.loads((directory / "fall-theories.json").read_bytes())
    original[1]["parent_version"] = None
    (directory / "fall-theories.json").write_text(
        json.dumps(original), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="persisted_equation_history"):
        verifier.verify_persisted(directory, payload)


def test_pending_database_wal_is_never_silently_ignored(verifier, tmp_path):
    directory = tmp_path / "full"
    payload = verifier.run_full_benchmark(
        seed=2, intervention_budget=80, output=directory, overwrite=False
    )
    (directory / "experience.sqlite3-wal").write_bytes(b"pending or corrupt")
    with pytest.raises(ValueError, match="artifact set"):
        verifier.verify_persisted(directory, payload)


def test_changed_repeat_bytes_cannot_pass_release(verifier, tmp_path, monkeypatch):
    original = verifier.artifact_ledger
    calls = 0

    def changed_ledger(directory):
        nonlocal calls
        calls += 1
        ledger = original(directory)
        # Each repeat checks completeness before its independent final ledger.
        if calls == 4:
            ledger["structured/full/events.jsonl"] = "changed-repeat"
        return ledger

    monkeypatch.setattr(verifier, "artifact_ledger", changed_ledger)
    result = verifier.verify_release(output=tmp_path / "release", seeds=1)
    assert not result["passed"] and not result["runs"][0]["identical_artifacts"]
    assert result["failure"]["type"] == "ValueError"


def test_forged_transfer_actions_cannot_pass_independent_replay(verifier, tmp_path):
    directory = tmp_path / "full"
    verifier.run_full_benchmark(
        seed=0, intervention_budget=80, output=directory, overwrite=False
    )
    events = directory / "structured/library-transfer-events.jsonl"
    records = [
        json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()
    ]
    records[-1]["event"]["output_id"] = "forged-opaque-output"
    events.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="independently replay"):
        verifier.verify_transfer(directory / "structured")


def test_symbol_switch_cannot_substitute_an_unlearned_program(verifier, tmp_path):
    directory = tmp_path / "full"
    verifier.run_full_benchmark(
        seed=0, intervention_budget=80, output=directory, overwrite=False
    )
    path = directory / "structured/transfer-evidence.json"
    evidence = json.loads(path.read_bytes())
    evidence["library"]["symbol"]["definition_ir"]["body"] = [
        "activate",
        "place_input",
        "wait",
        "interact",
    ]
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="not actually learned"):
        verifier.verify_transfer(directory / "structured")

"""Complete paired denominators, public-first I/O and tamper-negative audit."""

import json
from pathlib import Path

import pytest

from research.structural_campaigns.corpus import digest
from research.structural_campaigns.test_campaign import dump, pinned_stub
from research.structural_campaigns.test_campaign import runner as historical_runner

from . import campaign


@pytest.fixture
def historical(tmp_path):
    corpus, source = tmp_path / "corpus", tmp_path / "source"
    corpus_pin, _ = pinned_stub(corpus, development=1)
    historical_runner().run(corpus=corpus, manifest_sha256=corpus_pin, output=source)
    return {
        "source": source,
        "source_manifest_sha256": digest(source / "manifest.json"),
        "corpus": corpus,
        "corpus_manifest_sha256": corpus_pin,
    }


def test_inputs_read_no_old_private_score_or_sealed_fixture(historical, monkeypatch):
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name in ("audit.json", "world.json", "metadata.json") or (
            path == historical["source"] / "report.json"
        ):
            pytest.fail("old private outcome was used in prospective validation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    selected, snapshots, _ = campaign.inputs(**historical)
    assert len(selected.fixture_ids) == 1 and len(snapshots) == 4


def test_complete_pair_runner_and_verifier_are_public_first(
    historical, tmp_path, monkeypatch
):
    prepared, checked = [], []
    prepare, verify = campaign.prepare, campaign.verify_public

    def mark_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prepared.append(True)
        return result

    def mark_verify(*args, **kwargs):
        result = verify(*args, **kwargs)
        checked.append(True)
        return result

    original = Path.read_bytes
    stage = "run"

    def guarded(path):
        if (
            path.name in ("world.json", "metadata.json")
            and historical["corpus"] in path.parents
        ):
            assert len(prepared) == 4
            if stage == "verify":
                assert len(checked) == 8
        return original(path)

    monkeypatch.setattr(campaign, "prepare", mark_prepare)
    monkeypatch.setattr(campaign, "verify_public", mark_verify)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    output = tmp_path / "paired"
    report = campaign.run(**historical, output=output)
    assert report["summary"]["paid_pairs"] == 4
    assert report["summary"]["branches"] == 8
    assert report["summary"]["independent_development_groups"] == 1
    assert all(
        not row["eligible"] and row["spent"] == row["prefix_spent"]
        for row in report["rows"]
    )
    prepared.clear()
    checked.clear()
    stage = "verify"
    assert campaign.verify_output(output, **historical)["valid"]


@pytest.mark.parametrize(
    "tamper", ("group_count", "drop_branch", "prefix", "score", "mutation_protocol")
)
def test_semantic_tampering_rejected_even_after_rehash(historical, tmp_path, tamper):
    output = tmp_path / "paired"
    report = campaign.run(**historical, output=output)
    manifest = json.loads((output / "manifest.json").read_text())
    relative = "report.json"
    if tamper == "group_count":
        report["summary"]["independent_development_groups"] = 8
        value = report
    elif tamper == "drop_branch":
        report["rows"].pop()
        value = report
    elif tamper == "mutation_protocol":
        relative = "registration.json"
        value = json.loads((output / relative).read_text())
        value["mutation_protocol"]["mutant_limit"] = 1
    else:
        relative = next(
            key
            for key in manifest["sha256"]
            if key.endswith("prefix.json" if tamper == "prefix" else "audit.json")
        )
        value = json.loads((output / relative).read_text())
        if tamper == "prefix":
            value["prefix_spent"] = 0
        else:
            value["raw_score"]["promoted_correct_at_H"] = True
    dump(output / relative, value)
    manifest["sha256"][relative] = digest(output / relative)
    dump(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        campaign.verify_output(output, **historical)


def test_input_pin_drift_and_existing_output_do_not_overwrite(historical, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        campaign.run(**historical, output=output)
    with pytest.raises(ValueError, match="manifest pin"):
        campaign.run(
            **(historical | {"source_manifest_sha256": "0" * 64}),
            output=tmp_path / "new",
        )
    assert not (tmp_path / "new").exists()

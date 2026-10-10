"""Codec compatibility, full denominators and public-first expanded adapters."""

import gc
from pathlib import Path

import pytest

from minecraft_scientist.protocols.world import _CAPABILITIES

from research.challenger_validation.campaign import run as old_pair_run
from research.structural_campaigns.corpus import digest
from research.structural_campaigns.test_campaign import pinned_stub
from research.structural_campaigns.test_campaign import runner as old_runner

from . import structural, validation
from .store import Reader, Writer


@pytest.fixture(scope="module")
def paired_sources(tmp_path_factory):
    base = tmp_path_factory.mktemp("compact-compatibility")
    corpus = base / "corpus"
    pin, _ = pinned_stub(corpus, development=1)
    old, new = base / "legacy", base / "compact"
    gc.collect()
    capability_baseline = len(_CAPABILITIES)
    old_report = old_runner().run(corpus=corpus, manifest_sha256=pin, output=old)
    new_report = structural.run(corpus=corpus, manifest_sha256=pin, output=new)
    return {
        "base": base,
        "corpus": corpus,
        "pin": pin,
        "old": old,
        "new": new,
        "old_report": old_report,
        "new_report": new_report,
        "capability_baseline": capability_baseline,
    }


def test_complete_real_campaign_and_policy_replay_release_capabilities(paired_sources):
    f = paired_sources
    structural.verify_output(f["new"], corpus=f["corpus"], manifest_sha256=f["pin"])
    gc.collect()
    assert len(_CAPABILITIES) == f["capability_baseline"]


def test_compressed_run_is_logically_identical_to_unchanged_learners(paired_sources):
    f = paired_sources
    assert f["new_report"]["summary"] == f["old_report"]["summary"]
    assert f["new_report"]["rows"] == f["old_report"]["rows"]
    reader = Reader(f["new"])
    for name, record in reader.manifest["artifacts"].items():
        if name not in ("registration.json", "report.json"):
            assert record["payload_sha256"] == digest(f["old"] / name)
    verified = structural.verify_output(
        f["new"], corpus=f["corpus"], manifest_sha256=f["pin"]
    )
    assert verified["valid"] and verified["summary"]["campaigns"] == 4


def test_structural_verifier_checks_all_public_before_private(
    paired_sources, monkeypatch
):
    f = paired_sources
    original_check, original_read = structural.check_public, Reader.read
    public_checks = []

    def marked(*args):
        result = original_check(*args)
        public_checks.append(True)
        return result

    def guarded(store, name):
        if name.endswith("/evaluator/world.json"):
            assert len(public_checks) == 4
        return original_read(store, name)

    original_bytes = Path.read_bytes

    def guard_private(path):
        if path.name in ("world.json", "metadata.json") and f["corpus"] in path.parents:
            assert len(public_checks) == 4
        return original_bytes(path)

    monkeypatch.setattr(structural, "check_public", marked)
    monkeypatch.setattr(Reader, "read", guarded)
    monkeypatch.setattr(Path, "read_bytes", guard_private)
    assert structural.verify_output(
        f["new"], corpus=f["corpus"], manifest_sha256=f["pin"]
    )["valid"]


def test_compact_pair_does_not_read_source_scores_and_matches_legacy(
    paired_sources, tmp_path, monkeypatch
):
    f = paired_sources
    old_output = tmp_path / "old-pair"
    old_report = old_pair_run(
        source=f["old"],
        source_manifest_sha256=digest(f["old"] / "manifest.json"),
        corpus=f["corpus"],
        corpus_manifest_sha256=f["pin"],
        output=old_output,
    )
    source_store = Reader(f["new"])
    original = Reader.read

    def guarded(store, name):
        if store.directory == f["new"] and name.endswith(
            ("audit.json", "report.json", "evaluator/world.json")
        ):
            pytest.fail("source private outcome used by compact prospective policy")
        return original(store, name)

    monkeypatch.setattr(Reader, "read", guarded)
    output = tmp_path / "new-pair"
    new_report = validation.run(
        source=f["new"],
        source_manifest_sha256=source_store.manifest_sha256,
        corpus=f["corpus"],
        corpus_manifest_sha256=f["pin"],
        output=output,
    )
    assert {
        k: v for k, v in new_report["summary"].items() if k != "prefix_outcomes"
    } == old_report["summary"]
    assert [
        {k: v for k, v in row.items() if k != "prefix_outcome"}
        for row in new_report["rows"]
    ] == old_report["rows"]
    reader = Reader(output)
    for name, record in reader.manifest["artifacts"].items():
        if name.endswith(("/run.json", "/audit.json")):
            assert record["payload_sha256"] == digest(old_output / name)
    assert validation.verify_output(
        output,
        source=f["new"],
        source_manifest_sha256=source_store.manifest_sha256,
        corpus=f["corpus"],
        corpus_manifest_sha256=f["pin"],
    )["valid"]


def clone_with_report_tamper(source, destination, key, value):
    reader, writer = Reader(source), Writer(destination)
    for name in reader.manifest["artifacts"]:
        data = reader.read(name)
        if name == "report.json":
            data["summary"][key] = value
        writer.write(name, data)
    writer.write(
        "timing.json", reader.nondeterministic("timing.json"), nondeterministic=True
    )
    writer.finish()


@pytest.mark.parametrize(
    "key,value", (("campaigns", 3), ("independent_development_groups", 4))
)
def test_recompressed_rehashed_denominator_tamper_fails(
    paired_sources, tmp_path, key, value
):
    f = paired_sources
    target = tmp_path / "forged"
    clone_with_report_tamper(f["new"], target, key, value)
    with pytest.raises(ValueError, match="denominators"):
        structural.verify_output(target, corpus=f["corpus"], manifest_sha256=f["pin"])


def test_existing_output_and_bad_protocol_preserve_data(paired_sources, tmp_path):
    f = paired_sources
    with pytest.raises(FileExistsError):
        structural.run(corpus=f["corpus"], manifest_sha256=f["pin"], output=f["new"])
    with pytest.raises(ValueError):
        structural.run(
            corpus=f["corpus"],
            manifest_sha256=f["pin"],
            output=tmp_path / "never-created",
            protocol=structural.PROTOCOL | {"budget": False},
        )
    assert not (tmp_path / "never-created").exists()


def test_pair_verifier_completes_all_public_checks_before_private(
    paired_sources, tmp_path, monkeypatch
):
    f = paired_sources
    output = tmp_path / "paired-public-first"
    arguments = {
        "source": f["new"],
        "source_manifest_sha256": digest(f["new"] / "manifest.json"),
        "corpus": f["corpus"],
        "corpus_manifest_sha256": f["pin"],
    }
    validation.run(**arguments, output=output)
    checks = []
    original_check, original_bytes = validation.verify_public, Path.read_bytes

    def marked(*args):
        result = original_check(*args)
        checks.append(True)
        return result

    def guarded(path):
        if path.name in ("world.json", "metadata.json") and f["corpus"] in path.parents:
            assert len(checks) >= 8
        return original_bytes(path)

    monkeypatch.setattr(validation, "verify_public", marked)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert validation.verify_output(output, **arguments)["valid"]


@pytest.mark.parametrize(
    "tag",
    (
        "eligible_tail",
        "bootstrap_incomplete",
        "monitoring_incomplete",
        "bounded_no_anomaly",
        "prior_validation_rejected",
        "no_repair_candidate",
        "prior_validation_unresolved",
    ),
)
def test_prefix_outcome_does_not_mislabel_no_anomaly_as_repair_failure(tag):
    result = {
        "public_old_theory": {},
        "anomaly": {},
        "monitoring": [{"status": "passed"}],
    }
    snapshot = {"eligible": False, "prior_counterexample_ids": [], "candidate_ir": {}}
    if tag == "eligible_tail":
        snapshot["eligible"] = True
    elif tag == "bootstrap_incomplete":
        result["public_old_theory"] = None
    elif tag in ("monitoring_incomplete", "bounded_no_anomaly"):
        result["anomaly"] = None
        if tag == "monitoring_incomplete":
            result["monitoring"][0]["status"] = "budget_exhausted"
    elif tag == "prior_validation_rejected":
        snapshot["prior_counterexample_ids"] = ["q"]
    elif tag == "no_repair_candidate":
        snapshot["candidate_ir"] = None
    assert validation.outcome(result, snapshot) == tag

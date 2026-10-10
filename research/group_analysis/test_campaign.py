"""Real complete-audit integration; synthetic corpus is only a loader fixture."""

import pytest

from research.compact_campaigns import structural, validation
from research.compact_campaigns.store import Reader, Writer
from research.structural_campaigns.corpus import digest
from research.structural_campaigns.test_campaign import pinned_stub

from . import campaign


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("group-analysis")
    corpus = root / "corpus"
    corpus_pin, _ = pinned_stub(corpus, development=2)
    source = root / "structural"
    structural.run(corpus=corpus, manifest_sha256=corpus_pin, output=source)
    source_pin = digest(source / "manifest.json")
    paired = root / "validation"
    validation.run(
        source=source,
        source_manifest_sha256=source_pin,
        corpus=corpus,
        corpus_manifest_sha256=corpus_pin,
        output=paired,
    )
    return {
        "source": source,
        "source_manifest_sha256": source_pin,
        "corpus": corpus,
        "corpus_manifest_sha256": corpus_pin,
        "validation_source": paired,
        "validation_manifest_sha256": digest(paired / "manifest.json"),
        "resamples": 100,
    }


def test_real_complete_audits_precede_each_scored_analysis(
    completed, tmp_path, monkeypatch
):
    events = []
    original_source, original_paired, original_analyze = (
        structural.verify_output,
        validation.verify_output,
        campaign.analyze,
    )

    def source_check(*args, **kwargs):
        events.append("source_audit_start")
        result = original_source(*args, **kwargs)
        events.append("source_audit_end")
        return result

    def paired_check(*args, **kwargs):
        events.append("paired_audit_start")
        result = original_paired(*args, **kwargs)
        events.append("paired_audit_end")
        return result

    def analyze(*args, **kwargs):
        events.append("analyze_" + kwargs["kind"])
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(structural, "verify_output", source_check)
    monkeypatch.setattr(validation, "verify_output", paired_check)
    monkeypatch.setattr(campaign, "analyze", analyze)
    output = tmp_path / "analysis"
    report = campaign.run(**completed, output=output)
    assert events == [
        "source_audit_start",
        "source_audit_end",
        "analyze_structural",
        "paired_audit_start",
        "paired_audit_end",
        "analyze_validation",
    ]
    assert report["analyses"]["structural"]["fixtures"] == 2
    assert report["analyses"]["structural"]["leakage_groups"] == 2
    assert report["analyses"]["structural"]["rows"] == 8
    assert report["analyses"]["validation"]["rows"] == 16
    assert report["complete_independent_audits"]["structural"]["valid"]
    assert report["complete_independent_audits"]["validation"]["valid"]
    store = Reader(output)
    store.coverage({"registration.json", "report.json"})
    assert store.read("report.json") == report
    assert (
        "not total scientific compute" in store.nondeterministic("timing.json")["scope"]
    )


@pytest.mark.parametrize(
    "failure",
    (
        "source_pin",
        "validation_pin",
        "corpus_pin",
        "partial",
        "missing_pair_pin",
        "existing",
    ),
)
def test_bad_or_partial_inputs_never_create_derived_certificate(
    completed, tmp_path, failure
):
    arguments = dict(completed)
    output = tmp_path / "no-certificate"
    if failure in ("source_pin", "validation_pin", "corpus_pin"):
        arguments[
            {
                "source_pin": "source_manifest_sha256",
                "validation_pin": "validation_manifest_sha256",
                "corpus_pin": "corpus_manifest_sha256",
            }[failure]
        ] = "0" * 64
    elif failure == "partial":
        arguments["source"] = tmp_path / "partial"
        arguments["source"].mkdir()
    elif failure == "missing_pair_pin":
        arguments["validation_manifest_sha256"] = None
    else:
        output.mkdir()
        (output / "keep.txt").write_text("retained", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        campaign.run(**arguments, output=output)
    assert not (output / "manifest.json").exists()
    if failure == "existing":
        assert (output / "keep.txt").read_text(encoding="utf-8") == "retained"
    else:
        assert not output.exists()


def test_stored_rehashed_false_summary_is_not_an_audit_certificate(completed, tmp_path):
    original = Reader(completed["source"])
    forged = Writer(tmp_path / "forged-source")
    for name in original.manifest["artifacts"]:
        value = original.read(name)
        if name == "report.json":
            value["summary"]["campaigns"] = 999
        forged.write(name, value)
    forged.write(
        "timing.json", original.nondeterministic("timing.json"), nondeterministic=True
    )
    forged.finish()
    arguments = dict(completed) | {
        "source": forged.directory,
        "source_manifest_sha256": digest(forged.directory / "manifest.json"),
        "validation_source": None,
        "validation_manifest_sha256": None,
    }
    output = tmp_path / "no-analysis"
    with pytest.raises(ValueError, match="denominators"):
        campaign.run(**arguments, output=output)
    assert not output.exists()


def test_source_drift_after_real_audit_prevents_completion(
    completed, tmp_path, monkeypatch
):
    calls = []
    original = campaign.hashes

    def drift():
        value = original()
        calls.append(True)
        return value if len(calls) == 1 else value | {"forged.py": "0" * 64}

    monkeypatch.setattr(campaign, "hashes", drift)
    arguments = dict(completed) | {
        "validation_source": None,
        "validation_manifest_sha256": None,
    }
    output = tmp_path / "drifted"
    with pytest.raises(ValueError, match="source drift"):
        campaign.run(**arguments, output=output)
    assert not output.exists()


def test_audit_rejection_cannot_be_overridden_by_stored_success(
    completed, tmp_path, monkeypatch
):
    def rejected(*args, **kwargs):
        return {"valid": False}

    def forbidden(*args, **kwargs):
        raise AssertionError("scored analysis must not run after audit rejection")

    monkeypatch.setattr(structural, "verify_output", rejected)
    monkeypatch.setattr(campaign, "analyze", forbidden)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="audit rejected"):
        campaign.run(**completed, output=output)
    assert not output.exists()


def test_manifest_changed_after_real_analysis_is_detected(
    completed, tmp_path, monkeypatch
):
    import shutil

    source = tmp_path / "source-copy"
    shutil.copytree(completed["source"], source)
    original = campaign.analyze

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        manifest = source / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(campaign, "analyze", changed)
    arguments = dict(completed) | {
        "source": source,
        "validation_source": None,
        "validation_manifest_sha256": None,
    }
    output = tmp_path / "changed-input"
    with pytest.raises(ValueError, match="pin mismatch"):
        campaign.run(**arguments, output=output)
    assert not output.exists()

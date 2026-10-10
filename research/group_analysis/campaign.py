"""Analyze completed, source-pinned development artifacts only after full audit."""

import re
from pathlib import Path
from time import perf_counter, process_time

from research.compact_campaigns import structural, validation
from research.compact_campaigns.store import Reader, Writer
from research.structural_campaigns.corpus import digest

from .metrics import DEFAULT_RESAMPLES, DEFAULT_SEED, analyze

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "post-freeze exploratory DEVELOPMENT group diagnostics; no sealed/statistical superiority/innovation/H4/Minecraft/RGB completion"


def hashes():
    paths = list((ROOT / "research" / "group_analysis").glob("*.py"))
    paths += list((ROOT / "research" / "group_analysis").glob("*.md"))
    paths += [
        ROOT / "scripts" / "analyze_development_groups.py",
        ROOT / "docs" / "group-analysis-design.md",
    ]
    return structural.hashes() | {
        p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)
    }


def pin(directory, expected):
    if (
        type(expected) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or digest(Path(directory) / "manifest.json") != expected
    ):
        raise ValueError("analysis input manifest pin mismatch or incomplete source")


def run(
    *,
    source,
    source_manifest_sha256,
    corpus,
    corpus_manifest_sha256,
    output,
    validation_source=None,
    validation_manifest_sha256=None,
    resamples=DEFAULT_RESAMPLES,
    seed=DEFAULT_SEED,
):
    # No output or derived report from incomplete/pin-mismatched sources. Calling
    # the real independent verifier is mandatory; a caller-supplied 'valid' flag
    # or stored report/summary does not certify the input.
    if Path(output).exists():
        raise FileExistsError("analysis output exists; never overwrite")
    if (validation_source is None) != (validation_manifest_sha256 is None):
        raise ValueError("validation source and pin must be supplied together")
    sources = hashes()
    pin(source, source_manifest_sha256)
    pin(corpus, corpus_manifest_sha256)
    if validation_source is not None:
        pin(validation_source, validation_manifest_sha256)
    wall, cpu = perf_counter(), process_time()
    source_audit = structural.verify_output(
        source,
        corpus=corpus,
        manifest_sha256=corpus_manifest_sha256,
        input_manifest_sha256=source_manifest_sha256,
    )
    if source_audit.get("valid") is not True:
        raise ValueError("independent source audit rejected")
    source_reader = Reader(source, manifest_sha256=source_manifest_sha256)
    source_report = source_reader.read("report.json")
    selection = source_report["registration"]["selection"]
    results = {
        "structural": analyze(
            source_report["rows"],
            selection,
            kind="structural",
            resamples=resamples,
            seed=seed,
        )
    }
    audits, timings = (
        {"structural": source_audit},
        {"structural": source_reader.nondeterministic("timing.json")},
    )
    if validation_source is not None:
        paired_audit = validation.verify_output(
            validation_source,
            source=source,
            source_manifest_sha256=source_manifest_sha256,
            corpus=corpus,
            corpus_manifest_sha256=corpus_manifest_sha256,
        )
        if paired_audit.get("valid") is not True:
            raise ValueError("independent paired audit rejected")
        paired_reader = Reader(
            validation_source, manifest_sha256=validation_manifest_sha256
        )
        paired_report = paired_reader.read("report.json")
        if paired_report["registration"]["selection"] != selection:
            raise ValueError("source/validation selection mismatch")
        results["validation"] = analyze(
            paired_report["rows"],
            selection,
            kind="validation",
            source_rows=source_report["rows"],
            resamples=resamples,
            seed=seed,
        )
        audits["validation"] = paired_audit
        timings["validation"] = paired_reader.nondeterministic("timing.json")
    pin(source, source_manifest_sha256)
    pin(corpus, corpus_manifest_sha256)
    if validation_source is not None:
        pin(validation_source, validation_manifest_sha256)
    if hashes() != sources:
        raise ValueError("analysis dependency source drift; no certificate")
    registration = {
        "schema": "paired-group-analysis-registration-v1",
        "source_sha256": sources,
        "structural_manifest_sha256": source_manifest_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "selection": selection,
        "resamples": resamples,
        "seed": seed,
        "scope": SCOPE,
    }
    report = {
        "schema": "paired-group-analysis-report-v1",
        "registration": registration,
        "complete_independent_audits": audits,
        "analyses": results,
        "scope": SCOPE,
    }
    store = Writer(output)
    store.write("registration.json", registration)
    store.write("report.json", report)
    store.write(
        "timing.json",
        {
            "inputs": timings,
            "actual_analysis_with_complete_audits": {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            },
            "scope": "nondeterministic observed runtime, not total scientific compute; validation timings inherit rather than regenerate discovery search",
        },
        nondeterministic=True,
    )
    store.finish()
    return report

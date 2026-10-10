"""Same paid challenger ablation, streamed losslessly across development groups."""

import json
import math
from pathlib import Path
from time import perf_counter, process_time

from research.challenger_validation.audit import raw_score, verify_public
from research.challenger_validation.campaign import make_row, report_summary
from research.challenger_validation.paired import POLICIES, prepare, run_branch
from research.structural_campaigns.corpus import digest
from research.structural_campaigns.runtime import (
    MATCHED_METHODS,
    prelude,
    public_config,
)
from research.structural_holdouts.audit_corpus import validate_fixture
from research.structural_holdouts.generator import instantiate

from .store import CODEC, Reader, Writer
from .structural import hashes, opened

SCOPE = "complete selected DEVELOPMENT paid validation-policy ablation; all real prefix costs, unchanged eight Wp tests, only final three slots differ; inherited discovery compute, bounded row fault domain; no sealed statistics/unknown control/noise/H4/Minecraft/RGB/innovation claim"


def outcome(result, snapshot):
    if snapshot["eligible"]:
        return "eligible_tail"
    if result["public_old_theory"] is None:
        return "bootstrap_incomplete"
    if result["anomaly"] is None:
        return (
            "monitoring_incomplete"
            if any(b["status"] != "passed" for b in result["monitoring"])
            else "bounded_no_anomaly"
        )
    if snapshot["prior_counterexample_ids"]:
        return "prior_validation_rejected"
    return (
        "no_repair_candidate"
        if snapshot["candidate_ir"] is None
        else "prior_validation_unresolved"
    )


def prefixes(source_store, selected, registration):
    """Only public results, never old score/report/private outcome selection."""
    for fid in selected.fixture_ids:
        public, baseline = selected.read(f"{fid}/public.json"), None
        for method in MATCHED_METHODS:
            relative = f"{fid}/{method}"
            config, result = (
                source_store.read(f"{relative}/config.json"),
                source_store.read(f"{relative}/run.json"),
            )
            if config != public_config(public, method, registration["protocol"]):
                raise ValueError("compact prefix private config/method mismatch")
            snapshot = prepare(config, result)
            current = prelude(result)
            if baseline is not None and baseline != current:
                raise ValueError("compact prefix common prelude mismatch")
            baseline = current
            # These projections are public and must not silently lose events.
            from research.structural_campaigns.runtime import control_groups

            if source_store.read(f"{relative}/theory_history.json") != result[
                "theory_events"
            ] or source_store.read(f"{relative}/evidence.json") != list(
                control_groups(result)
            ):
                raise ValueError("compact source history/evidence projection mismatch")
            yield (
                fid,
                method,
                {
                    "schema": "compact-paid-prefix-v1",
                    "snapshot": snapshot,
                    "prefix_outcome": outcome(result, snapshot),
                },
            )


def totals(rows, selected):
    summary = report_summary(rows, selected)
    summary["prefix_outcomes"] = {
        method: {
            tag: sum(
                r["prefix_outcome"] == tag
                for r in rows
                if r["method"] == method and r["policy"] == POLICIES[0]
            )
            for tag in (
                "eligible_tail",
                "bootstrap_incomplete",
                "monitoring_incomplete",
                "bounded_no_anomaly",
                "prior_validation_rejected",
                "no_repair_candidate",
                "prior_validation_unresolved",
            )
        }
        for method in MATCHED_METHODS
    }
    return summary


def run(*, source, source_manifest_sha256, corpus, corpus_manifest_sha256, output):
    source_store, selected, inherited = opened(
        source,
        corpus=corpus,
        manifest_sha256=corpus_manifest_sha256,
        input_manifest_sha256=source_manifest_sha256,
    )
    store = Writer(output)
    sources = hashes()
    registration = {
        "schema": "compact-validation-development-registration-v1",
        "source_sha256": sources,
        "source_campaign_manifest_sha256": source_manifest_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "selection": inherited["selection"],
        "methods": list(MATCHED_METHODS),
        "policies": list(POLICIES),
        "mutation_protocol": {
            "mutant_limit": 512,
            "operation_budget": inherited["protocol"]["operation_budget"],
            "fault_domain": "one row target or output replacement; no inserted states",
        },
        "codec": CODEC,
        "scope": SCOPE,
    }
    store.write("registration.json", registration)
    for fid, method, envelope in prefixes(source_store, selected, inherited):
        store.write(f"{fid}/{method}/prefix.json", envelope)
    # ALL validated public prefixes frozen before ANY selected private content.
    rows, timings = [], []
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        for method in MATCHED_METHODS:
            envelope = store.read(f"{fid}/{method}/prefix.json")
            snapshot = envelope["snapshot"]
            for policy in POLICIES:
                relative = f"{fid}/{method}/{policy}"
                wall, cpu = perf_counter(), process_time()
                result = run_branch(instantiate(private), snapshot, policy)
                timing = {
                    "wall_seconds": perf_counter() - wall,
                    "cpu_seconds": process_time() - cpu,
                }
                store.write(f"{relative}/run.json", result)
                verified = verify_public(snapshot, result)
                score = raw_score(snapshot, result, private)
                store.write(f"{relative}/audit.json", verified | {"raw_score": score})
                rows.append(
                    make_row(
                        fid, method, policy, snapshot, result, score, selected, metadata
                    )
                    | {"prefix_outcome": envelope["prefix_outcome"]}
                )
                timings.append(
                    {
                        "fixture_id": fid,
                        "method": method,
                        "policy": policy,
                        "paid_replay_and_validation": timing,
                    }
                )
        print(f"completed lossless paired fixture {fid}", flush=True)
    if (
        sources != hashes()
        or digest(Path(source) / "manifest.json") != source_manifest_sha256
        or digest(Path(corpus) / "manifest.json") != corpus_manifest_sha256
    ):
        raise RuntimeError(
            "source/input drift; preserve partial evidence without final manifest"
        )
    summary = totals(rows, selected)
    report = {
        "schema": "compact-validation-development-report-v1",
        "registration": registration,
        "rows": rows,
        "summary": summary,
        "scope": SCOPE,
    }
    store.write("report.json", report)
    store.write(
        "timing.json",
        {
            "rows": timings,
            "scope": "nondeterministic real prefix replay and validation; inherited discovery search not reexecuted/timed",
        },
        nondeterministic=True,
    )
    store.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)
    return report


def verify_output(
    directory, *, source, source_manifest_sha256, corpus, corpus_manifest_sha256
):
    source_store, selected, inherited = opened(
        source,
        corpus=corpus,
        manifest_sha256=corpus_manifest_sha256,
        input_manifest_sha256=source_manifest_sha256,
    )
    store = Reader(directory)
    registration = store.read("registration.json")
    expected_registration = {
        "schema": "compact-validation-development-registration-v1",
        "source_sha256": hashes(),
        "source_campaign_manifest_sha256": source_manifest_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "selection": inherited["selection"],
        "methods": list(MATCHED_METHODS),
        "policies": list(POLICIES),
        "mutation_protocol": {
            "mutant_limit": 512,
            "operation_budget": inherited["protocol"]["operation_budget"],
            "fault_domain": "one row target or output replacement; no inserted states",
        },
        "codec": CODEC,
        "scope": SCOPE,
    }
    if registration != expected_registration:
        raise ValueError("compact paid registration/source mismatch")
    expected = {"registration.json", "report.json"}
    for fid in selected.fixture_ids:
        for method in MATCHED_METHODS:
            expected.add(f"{fid}/{method}/prefix.json")
            expected.update(
                f"{fid}/{method}/{policy}/{name}"
                for policy in POLICIES
                for name in ("run.json", "audit.json")
            )
    store.coverage(expected)
    for fid, method, envelope in prefixes(source_store, selected, inherited):
        if store.read(f"{fid}/{method}/prefix.json") != envelope:
            raise ValueError("compact paid prefix drift/leaked coverage outcome")
        for policy in POLICIES:
            result = store.read(f"{fid}/{method}/{policy}/run.json")
            if result["policy"] != policy:
                raise ValueError("compact paid policy mismatch")
            verify_public(envelope["snapshot"], result)
    # All public chronology/policy checks above finish before private I/O below.
    rows = []
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        for method in MATCHED_METHODS:
            envelope = store.read(f"{fid}/{method}/prefix.json")
            snapshot = envelope["snapshot"]
            for policy in POLICIES:
                relative = f"{fid}/{method}/{policy}"
                result = store.read(f"{relative}/run.json")
                score = raw_score(snapshot, result, private)
                if store.read(f"{relative}/audit.json") != verify_public(
                    snapshot, result
                ) | {"raw_score": score}:
                    raise ValueError("compact paid raw audit mismatch")
                rows.append(
                    make_row(
                        fid, method, policy, snapshot, result, score, selected, metadata
                    )
                    | {"prefix_outcome": envelope["prefix_outcome"]}
                )
    summary = totals(rows, selected)
    if store.read("report.json") != {
        "schema": "compact-validation-development-report-v1",
        "registration": registration,
        "rows": rows,
        "summary": summary,
        "scope": SCOPE,
    }:
        raise ValueError("compact paid report/denominator mismatch")
    timing = store.nondeterministic("timing.json")
    expected_order = [
        (fid, method, policy)
        for fid in selected.fixture_ids
        for method in MATCHED_METHODS
        for policy in POLICIES
    ]
    if [
        (r["fixture_id"], r["method"], r["policy"]) for r in timing["rows"]
    ] != expected_order:
        raise ValueError("compact paid timing coverage mismatch")
    for record in timing["rows"]:
        if set(record) != {
            "fixture_id",
            "method",
            "policy",
            "paid_replay_and_validation",
        }:
            raise ValueError("compact paid timing fields mismatch")
        values = record["paid_replay_and_validation"]
        if set(values) != {"wall_seconds", "cpu_seconds"} or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in values.values()
        ):
            raise ValueError("invalid compact paid timing")
    return {
        "valid": True,
        "summary": summary,
        "logical_artifacts": len(store.manifest["artifacts"]),
        "scope": SCOPE,
    }

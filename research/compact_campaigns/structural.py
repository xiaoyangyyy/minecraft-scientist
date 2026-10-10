"""Same four paid learners and audits, without dropping evidence to fit disk."""

import json
import math
from pathlib import Path
from time import perf_counter, process_time

from run_matched_h3_development import summarize as upstream_summary
from run_structural_development import hashes as upstream_hashes

from research.structural_campaigns.corpus import digest, load_development
from research.structural_campaigns.runtime import (
    MATCHED_METHODS,
    POLICY,
    PROTOCOL,
    audit,
    check_public,
    control_groups,
    learn,
    prelude,
    public_config,
)
from research.structural_holdouts.audit_corpus import validate_fixture
from research.structural_holdouts.generator import instantiate

from .store import CODEC, Reader, Writer

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "complete selected DEVELOPMENT leakage groups; same four paid learners/protocol, lossless logical evidence; no sealed statistics/unknown control/noise/H4/Minecraft/RGB/innovation claim"


def hashes():
    paths = list((ROOT / "research" / "compact_campaigns").glob("*.py"))
    paths += list((ROOT / "research" / "compact_campaigns").glob("*.md"))
    paths += [
        ROOT / "scripts" / "run_compact_structural_development.py",
        ROOT / "scripts" / "run_compact_validation_development.py",
        ROOT / "docs" / "compact-development-design.md",
    ]
    # Bind the unchanged validator as well as the shared learner dependency set.
    paths += list((ROOT / "research" / "challenger_validation").glob("*.py"))
    paths += list((ROOT / "research" / "challenger_validation").glob("*.md"))
    paths += [
        ROOT / "scripts" / "run_challenger_validation_development.py",
        ROOT / "docs" / "challenger-validation-design.md",
    ]
    return upstream_hashes() | {
        p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)
    }


def selection(selected, max_groups):
    return {
        "split": "development",
        "policy": "complete first lexicographically sorted development leakage groups; never filter outcomes",
        "requested_max_groups": max_groups,
        "available_development_groups": selected.available_groups,
        "fixture_ids": list(selected.fixture_ids),
        "groups": selected.groups,
    }


def row(fid, method, result, verified, selected, metadata):
    return {
        "fixture_id": fid,
        "independent_group": selected.groups[fid],
        "method": method,
        "case": metadata["inputs"]["case"],
        "states": metadata["inputs"]["states"],
        "renaming_index": metadata["inputs"]["renaming_index"],
        "status": result["status"],
        "stop_reason": result["stop_reason"],
        "spent": result["spent"],
        "evaluation": verified["evaluation"],
        "raw_score": verified["raw_score"],
        "construction_diagnostics": metadata["diagnostics"],
    }


def summary(rows, selected, *, independent=False):
    def recompute(subset):
        return {
            "runs": len(subset),
            "old_correct_at_H": sum(
                r["raw_score"]["old_correct_at_H"] is True for r in subset
            ),
            "new_correct_at_H": sum(
                r["raw_score"]["promoted_correct_at_H"] is True for r in subset
            ),
            "new_wrong_at_H": sum(
                r["raw_score"]["promoted_correct_at_H"] is False for r in subset
            ),
            "finite_replacements": sum(
                r["status"] == "revision_finitely_validated" for r in subset
            ),
            "unresolved": sum(r["status"] == "unresolved" for r in subset),
            "bounded_no_anomaly": sum(
                r["status"] == "bounded_monitoring_completed_no_anomaly" for r in subset
            ),
            "detected_post_boundary": sum(
                r["evaluation"]["post_boundary_anomaly_detected"] for r in subset
            ),
            "false_or_early_anomaly": sum(
                r["evaluation"]["anomaly_without_in_H_change_or_before_boundary"]
                for r in subset
            ),
            "actual_total_cost": sum(r["spent"] for r in subset),
        }

    fn = recompute if independent else upstream_summary
    return {
        "fixtures": len(selected.fixture_ids),
        "independent_development_groups": len(set(selected.groups.values())),
        "campaigns": len(rows),
        "methods": {
            method: fn([r for r in rows if r["method"] == method])
            for method in MATCHED_METHODS
        },
    }


def run(*, corpus, manifest_sha256, output, max_groups=None, protocol=None):
    protocol = dict(PROTOCOL if protocol is None else protocol)
    selected = load_development(
        corpus, manifest_sha256=manifest_sha256, max_groups=max_groups
    )
    publics = {fid: selected.read(f"{fid}/public.json") for fid in selected.fixture_ids}
    for public in publics.values():
        public_config(public, MATCHED_METHODS[0], protocol)
    store = Writer(output)
    sources = hashes()
    registration = {
        "schema": "compact-structural-development-registration-v1",
        "source_sha256": sources,
        "learner_source_sha256": upstream_hashes(),
        "corpus_manifest_sha256": manifest_sha256,
        "selection": selection(selected, max_groups),
        "methods": list(MATCHED_METHODS),
        "protocol": protocol,
        "repair_policy": POLICY,
        "codec": CODEC,
        "scope": SCOPE,
    }
    store.write("registration.json", registration)
    rows, timings = [], []
    for fid in selected.fixture_ids:
        public = publics[fid]
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        baseline = None
        for method in MATCHED_METHODS:
            prefix = f"{fid}/{method}"
            config = public_config(public, method, protocol)
            store.write(f"{prefix}/config.json", config)
            wall, cpu = perf_counter(), process_time()
            result = learn(instantiate(private), config)
            learning_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            store.write(f"{prefix}/run.json", result)  # Freeze before private score.
            current = prelude(result)
            if baseline is not None and current != baseline:
                raise RuntimeError("paid prelude differs; preserve partial evidence")
            baseline = current
            store.write(f"{prefix}/theory_history.json", result["theory_events"])
            store.write(f"{prefix}/evidence.json", list(control_groups(result)))
            private_name = f"{prefix}/evaluator/world.json"
            store.write(private_name, private)
            wall, cpu = perf_counter(), process_time()
            verified = audit(config, result, store.reference(private_name))
            audit_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            store.write(f"{prefix}/audit.json", verified)
            if not verified["valid"]:
                raise RuntimeError(
                    f"public/raw audit failure: {fid}/{method}; preserve partial evidence"
                )
            rows.append(row(fid, method, result, verified, selected, metadata))
            timings.append(
                {
                    "fixture_id": fid,
                    "method": method,
                    "complete_learning": learning_time,
                    "complete_audit": audit_time,
                }
            )
        selected.read(f"{fid}/evaluator/world.json")
        print(f"completed lossless development fixture {fid}", flush=True)
    if sources != hashes() or digest(Path(corpus) / "manifest.json") != manifest_sha256:
        raise RuntimeError(
            "source/input drift; retain partial evidence without final manifest"
        )
    totals = summary(rows, selected)
    report = {
        "schema": "compact-structural-development-report-v1",
        "registration": registration,
        "rows": rows,
        "summary": totals,
        "scope": SCOPE,
    }
    store.write("report.json", report)
    store.write(
        "timing.json",
        {
            "rows": timings,
            "scope": "nondeterministic actual learning/audit; compression and total orchestration not included",
        },
        nondeterministic=True,
    )
    store.finish()
    print(json.dumps(totals, sort_keys=True), flush=True)
    return report


def opened(directory, *, corpus, manifest_sha256, input_manifest_sha256=None):
    store = Reader(directory, manifest_sha256=input_manifest_sha256)
    registration = store.read("registration.json")
    if (
        registration["schema"] != "compact-structural-development-registration-v1"
        or registration["source_sha256"] != hashes()
        or registration["learner_source_sha256"] != upstream_hashes()
        or registration["methods"] != list(MATCHED_METHODS)
        or registration["corpus_manifest_sha256"] != manifest_sha256
        or registration["codec"] != CODEC
        or registration["repair_policy"] != POLICY
        or registration["scope"] != SCOPE
    ):
        raise ValueError("compact source/registration mismatch")
    selected = load_development(
        corpus,
        manifest_sha256=manifest_sha256,
        max_groups=registration["selection"]["requested_max_groups"],
    )
    if registration["selection"] != selection(
        selected, registration["selection"]["requested_max_groups"]
    ):
        raise ValueError("compact incomplete/reordered development groups")
    expected = {"registration.json", "report.json"}
    expected.update(
        f"{fid}/{method}/{name}"
        for fid in selected.fixture_ids
        for method in MATCHED_METHODS
        for name in (
            "config.json",
            "run.json",
            "theory_history.json",
            "evidence.json",
            "evaluator/world.json",
            "audit.json",
        )
    )
    store.coverage(expected)
    return store, selected, registration


def check_all_public(store, selected, registration):
    # Stream public runs instead of retaining all large candidate trees in RAM.
    for fid in selected.fixture_ids:
        public, baseline = selected.read(f"{fid}/public.json"), None
        for method in MATCHED_METHODS:
            prefix = f"{fid}/{method}"
            config, result = (
                store.read(f"{prefix}/config.json"),
                store.read(f"{prefix}/run.json"),
            )
            if config != public_config(public, method, registration["protocol"]):
                raise ValueError("private information in compact learner config")
            check_public(config, result)
            current = prelude(result)
            if baseline is not None and baseline != current:
                raise ValueError("compact common paid prelude mismatch")
            baseline = current
            if store.read(f"{prefix}/theory_history.json") != result[
                "theory_events"
            ] or store.read(f"{prefix}/evidence.json") != list(control_groups(result)):
                raise ValueError("compact history/evidence projection mismatch")


def verify_output(directory, *, corpus, manifest_sha256, input_manifest_sha256=None):
    store, selected, registration = opened(
        directory,
        corpus=corpus,
        manifest_sha256=manifest_sha256,
        input_manifest_sha256=input_manifest_sha256,
    )
    check_all_public(store, selected, registration)  # ALL before ANY private content.
    rows = []
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        for method in MATCHED_METHODS:
            prefix = f"{fid}/{method}"
            config, result = (
                store.read(f"{prefix}/config.json"),
                store.read(f"{prefix}/run.json"),
            )
            private_name = f"{prefix}/evaluator/world.json"
            if store.read(private_name) != private:
                raise ValueError("compact private schedule differs from corpus pin")
            verified = audit(config, result, store.reference(private_name))
            if not verified["valid"] or verified != store.read(f"{prefix}/audit.json"):
                raise ValueError("compact public/raw audit mismatch")
            rows.append(row(fid, method, result, verified, selected, metadata))
    totals = summary(rows, selected, independent=True)
    expected_report = {
        "schema": "compact-structural-development-report-v1",
        "registration": registration,
        "rows": rows,
        "summary": totals,
        "scope": SCOPE,
    }
    if store.read("report.json") != expected_report:
        raise ValueError("compact report/denominators mismatch")
    timing = store.nondeterministic("timing.json")
    if [(r["fixture_id"], r["method"]) for r in timing["rows"]] != [
        (fid, method) for fid in selected.fixture_ids for method in MATCHED_METHODS
    ]:
        raise ValueError("compact timing coverage mismatch")
    for record in timing["rows"]:
        if set(record) != {
            "fixture_id",
            "method",
            "complete_learning",
            "complete_audit",
        }:
            raise ValueError("compact timing fields mismatch")
        for key in ("complete_learning", "complete_audit"):
            if set(record[key]) != {"wall_seconds", "cpu_seconds"} or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in record[key].values()
            ):
                raise ValueError("invalid compact timing")
    return {
        "valid": True,
        "summary": totals,
        "logical_artifacts": len(store.manifest["artifacts"]),
        "scope": SCOPE,
    }

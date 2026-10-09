"""Full artifact/denominator verification, public validation before private I/O."""

import json
import math
from pathlib import Path

from research.structural_campaigns.corpus import digest, load_development
from research.structural_campaigns.runtime import (
    MATCHED_METHODS,
    PROTOCOL,
    audit,
    check_public,
    control_groups,
    prelude,
    public_config,
)
from research.structural_holdouts.audit_corpus import validate_fixture


def verify_output(directory, *, corpus, manifest_sha256, current_sources=True):
    directory = Path(directory)

    def read(relative):
        return json.loads((directory / relative).read_text(encoding="utf-8"))

    registration, manifest = read("registration.json"), read("manifest.json")
    if (
        registration["schema"] != "structural-campaign-development-registration-v1"
        or registration["methods"] != list(MATCHED_METHODS)
        or registration["corpus_manifest_sha256"] != manifest_sha256
        or registration["selection"]["split"] != "development"
        or set(registration["protocol"]) != set(PROTOCOL)
        or manifest["schema"] != "structural-campaign-development-manifest-v1"
        or manifest["nondeterministic_artifacts"] != ["timing.json"]
    ):
        raise ValueError("campaign registration/manifest mismatch")
    selected = load_development(
        corpus,
        manifest_sha256=manifest_sha256,
        max_groups=registration["selection"]["requested_max_groups"],
    )
    if (
        registration["selection"]["fixture_ids"] != list(selected.fixture_ids)
        or registration["selection"]["groups"] != selected.groups
        or registration["selection"]["available_development_groups"]
        != selected.available_groups
    ):
        raise ValueError("campaign omitted/reordered complete development groups")
    expected_files = {"registration.json", "report.json"}
    expected_files.update(
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
    if set(manifest["sha256"]) != expected_files or {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    } != expected_files | {"manifest.json", "timing.json"}:
        raise ValueError("campaign manifest/file coverage mismatch")
    for relative, expected in manifest["sha256"].items():
        path = directory / relative
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or digest(path) != expected
        ):
            raise ValueError("campaign artifact hash/escape mismatch")
    if current_sources:
        from run_structural_development import hashes

        if hashes() != registration["source_sha256"]:
            raise ValueError("campaign runtime source drift")
    campaigns = {}
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        base = None
        for method in MATCHED_METHODS:
            relative = f"{fid}/{method}"
            config, result = (
                read(f"{relative}/config.json"),
                read(f"{relative}/run.json"),
            )
            if config != public_config(public, method, registration["protocol"]):
                raise ValueError("private data/method/protocol in learner config")
            check_public(config, result)  # All before ANY private read.
            current = prelude(result)
            if base is not None and current != base:
                raise ValueError("campaign paid prelude mismatch")
            base = current
            if read(f"{relative}/theory_history.json") != result[
                "theory_events"
            ] or read(f"{relative}/evidence.json") != list(control_groups(result)):
                raise ValueError("campaign history/evidence projection mismatch")
            campaigns[fid, method] = config, result
    rows = []
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        for method in MATCHED_METHODS:
            relative = f"{fid}/{method}"
            config, result = campaigns[fid, method]
            if read(f"{relative}/evaluator/world.json") != private:
                raise ValueError("campaign private schedule disagrees with pin")
            verified = audit(
                config, result, directory / relative / "evaluator/world.json"
            )
            if not verified["valid"] or verified != read(f"{relative}/audit.json"):
                raise ValueError("campaign public/raw score audit mismatch")
            rows.append(
                {
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
            )
    report = read("report.json")
    if (
        report["schema"] != "structural-campaign-development-report-v1"
        or report["registration"] != registration
        or report["rows"] != rows
    ):
        raise ValueError("campaign report row/provenance mismatch")
    summary = {
        "fixtures": len(selected.fixture_ids),
        "independent_development_groups": len(set(selected.groups.values())),
        "campaigns": len(rows),
        "methods": {},
    }
    for method in MATCHED_METHODS:
        subset = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
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
    if report["summary"] != summary:
        raise ValueError("campaign report denominators/summary mismatch")
    timing = read("timing.json")
    if [(r["fixture_id"], r["method"]) for r in timing["rows"]] != list(campaigns):
        raise ValueError("campaign timing coverage mismatch")
    for row in timing["rows"]:
        if set(row) != {"fixture_id", "method", "complete_learning", "complete_audit"}:
            raise ValueError("campaign timing fields mismatch")
        for key in ("complete_learning", "complete_audit"):
            values = row[key]
            if set(values) != {"wall_seconds", "cpu_seconds"} or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in values.values()
            ):
                raise ValueError("campaign invalid compute timing")
    return {
        "valid": True,
        "summary": summary,
        "deterministic_artifacts": len(expected_files),
        "scope": "complete DEVELOPMENT groups and paid public/raw audit only; no sealed evaluation or scientific superiority claim",
    }

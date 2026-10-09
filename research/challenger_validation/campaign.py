"""Pinned development-only paid validation-policy experiment and verifier."""

import json
import math
from pathlib import Path
from time import perf_counter, process_time

from run_online_automata_development import dump
from run_structural_development import hashes as upstream_hashes

from research.structural_campaigns.corpus import digest, load_development
from research.structural_campaigns.runtime import MATCHED_METHODS, public_config
from research.structural_holdouts.audit_corpus import validate_fixture
from research.structural_holdouts.generator import instantiate

from .audit import raw_score, summarize, verify_public
from .paired import POLICIES, prepare, run_branch

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "DEVELOPMENT paid validation-policy ablation; inherited discovery plans/compute, unchanged eight Wp tests, replace only final three validation queries; no sealed evaluation/statistical superiority/unique causal identification/H4/Minecraft/RGB completion"


def hashes():
    owned = list((ROOT / "research" / "challenger_validation").glob("*.py"))
    owned += list((ROOT / "research" / "challenger_validation").glob("*.md"))
    owned += [
        ROOT / "scripts" / "run_challenger_validation_development.py",
        ROOT / "docs" / "challenger-validation-design.md",
    ]
    return upstream_hashes() | {
        p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(owned)
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inputs(source, source_manifest_sha256, corpus, corpus_manifest_sha256):
    """Prepare ALL public prefixes before opening ANY fixture private content.

    Pins only the source artifacts actually used: manifest, registration,
    config and public result. No old audit/report/private outcome is selected.
    """
    source = Path(source)
    if digest(source / "manifest.json") != source_manifest_sha256:
        raise ValueError("source campaign manifest pin mismatch")
    manifest = read_json(source / "manifest.json")
    if manifest["schema"] != "structural-campaign-development-manifest-v1":
        raise ValueError("invalid inherited campaign manifest")

    def pinned(relative):
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or digest(
            path
        ) != manifest["sha256"].get(relative):
            raise ValueError("inherited public artifact pin/escape mismatch")
        return read_json(path)

    registration = pinned("registration.json")
    if registration["source_sha256"] != upstream_hashes():
        raise ValueError("inherited discovery runtime source drift")
    if (
        registration["methods"] != list(MATCHED_METHODS)
        or registration["corpus_manifest_sha256"] != corpus_manifest_sha256
    ):
        raise ValueError("inherited method/corpus mismatch")
    selected = load_development(
        corpus,
        manifest_sha256=corpus_manifest_sha256,
        max_groups=registration["selection"]["requested_max_groups"],
    )
    if (
        registration["selection"]["split"] != "development"
        or registration["selection"]["fixture_ids"] != list(selected.fixture_ids)
        or registration["selection"]["groups"] != selected.groups
        or registration["selection"]["available_development_groups"]
        != selected.available_groups
    ):
        raise ValueError("inherited incomplete/reordered development groups")
    snapshots = {}
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        for method in MATCHED_METHODS:
            config = pinned(f"{fid}/{method}/config.json")
            if config != public_config(public, method, registration["protocol"]):
                raise ValueError("inherited learner config/private data mismatch")
            snapshots[fid, method] = prepare(config, pinned(f"{fid}/{method}/run.json"))
    return selected, snapshots, registration


def report_summary(rows, selected):
    return {
        "fixtures": len(selected.fixture_ids),
        "independent_development_groups": len(set(selected.groups.values())),
        "paid_pairs": len(selected.fixture_ids) * len(MATCHED_METHODS),
        "branches": len(rows),
        "methods": {
            method: {
                policy: summarize(
                    [r for r in rows if r["method"] == method and r["policy"] == policy]
                )
                for policy in POLICIES
            }
            for method in MATCHED_METHODS
        },
    }


def make_row(fid, method, policy, snapshot, result, score, selected, metadata):
    tail = result["tail"]
    return {
        "fixture_id": fid,
        "independent_group": selected.groups[fid],
        "method": method,
        "policy": policy,
        "case": metadata["inputs"]["case"],
        "renaming_index": metadata["inputs"]["renaming_index"],
        "eligible": snapshot["eligible"],
        "status": result["status"],
        "stop_reason": result["stop_reason"],
        "spent": result["spent"],
        "prefix_spent": result["prefix_spent"],
        "mutation_operations": 0
        if tail is None or policy != "public_mutation"
        else tail["plan"]["compute_operations"],
        "raw_score": score,
    }


def run(*, source, source_manifest_sha256, corpus, corpus_manifest_sha256, output):
    selected, snapshots, inherited = inputs(
        source, source_manifest_sha256, corpus, corpus_manifest_sha256
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = hashes()
    registration = {
        "schema": "paid-validation-development-registration-v1",
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
        "scope": SCOPE,
    }
    dump(output / "registration.json", registration)
    for (fid, method), snapshot in snapshots.items():
        directory = output / fid / method
        directory.mkdir(parents=True)
        dump(directory / "prefix.json", snapshot)  # All frozen before private I/O.
    rows, timings = [], []
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        for method in MATCHED_METHODS:
            snapshot = snapshots[fid, method]
            for policy in POLICIES:
                directory = output / fid / method / policy
                directory.mkdir()
                wall, cpu = perf_counter(), process_time()
                result = run_branch(instantiate(private), snapshot, policy)
                timing = {
                    "wall_seconds": perf_counter() - wall,
                    "cpu_seconds": process_time() - cpu,
                }
                dump(directory / "run.json", result)  # Freeze before raw scoring.
                verified = verify_public(snapshot, result)
                score = raw_score(snapshot, result, private)
                dump(directory / "audit.json", verified | {"raw_score": score})
                rows.append(
                    make_row(
                        fid, method, policy, snapshot, result, score, selected, metadata
                    )
                )
                timings.append(
                    {
                        "fixture_id": fid,
                        "method": method,
                        "policy": policy,
                        "paid_replay_and_validation": timing,
                    }
                )
        print(f"completed complete paired fixture {fid}", flush=True)
    if (
        sources != hashes()
        or digest(Path(source) / "manifest.json") != source_manifest_sha256
        or digest(Path(corpus) / "manifest.json") != corpus_manifest_sha256
    ):
        raise RuntimeError(
            "source/input drift; preserve partial evidence without manifest"
        )
    summary = report_summary(rows, selected)
    report = {
        "schema": "paid-validation-development-report-v1",
        "registration": registration,
        "rows": rows,
        "summary": summary,
        "scope": SCOPE,
    }
    dump(output / "report.json", report)
    dump(
        output / "timing.json",
        {
            "rows": timings,
            "scope": "nondeterministic paid prefix replay and tail; inherited original search not timed here",
        },
    )
    artifacts = {
        p.relative_to(output).as_posix(): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "timing.json"
    }
    dump(
        output / "manifest.json",
        {
            "schema": "paid-validation-development-manifest-v1",
            "sha256": artifacts,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return report


def verify_output(
    directory, *, source, source_manifest_sha256, corpus, corpus_manifest_sha256
):
    directory = Path(directory)
    registration, manifest = (
        read_json(directory / name) for name in ("registration.json", "manifest.json")
    )
    selected, snapshots, inherited = inputs(
        source, source_manifest_sha256, corpus, corpus_manifest_sha256
    )
    if (
        registration["schema"] != "paid-validation-development-registration-v1"
        or registration["source_sha256"] != hashes()
        or registration["source_campaign_manifest_sha256"] != source_manifest_sha256
        or registration["corpus_manifest_sha256"] != corpus_manifest_sha256
        or registration["selection"] != inherited["selection"]
        or registration["methods"] != list(MATCHED_METHODS)
        or registration["policies"] != list(POLICIES)
        or registration["mutation_protocol"]
        != {
            "mutant_limit": 512,
            "operation_budget": inherited["protocol"]["operation_budget"],
            "fault_domain": "one row target or output replacement; no inserted states",
        }
        or registration["scope"] != SCOPE
        or manifest["schema"] != "paid-validation-development-manifest-v1"
        or manifest["nondeterministic_artifacts"] != ["timing.json"]
    ):
        raise ValueError("paired validation registration/source mismatch")
    expected = {"registration.json", "report.json"}
    for fid, method in snapshots:
        expected.add(f"{fid}/{method}/prefix.json")
        expected.update(
            f"{fid}/{method}/{policy}/{name}"
            for policy in POLICIES
            for name in ("run.json", "audit.json")
        )
    actual = {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }
    if set(manifest["sha256"]) != expected or actual != expected | {
        "manifest.json",
        "timing.json",
    }:
        raise ValueError("paired artifact/denominator coverage mismatch")

    def pinned(relative):
        path = directory / relative
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or digest(path) != manifest["sha256"][relative]
        ):
            raise ValueError("paired artifact pin/escape mismatch")
        return read_json(path)

    branches = {}
    for (fid, method), snapshot in snapshots.items():
        if pinned(f"{fid}/{method}/prefix.json") != snapshot:
            raise ValueError(
                "paid public prefix changed or leaked old coverage outcome"
            )
        for policy in POLICIES:
            relative = f"{fid}/{method}/{policy}"
            result = pinned(f"{relative}/run.json")
            if result["policy"] != policy:
                raise ValueError("paired validation policy identity mismatch")
            verified = verify_public(snapshot, result)  # ALL before private I/O.
            branches[fid, method, policy] = (
                result,
                verified,
                pinned(f"{relative}/audit.json"),
            )
    rows = []
    for fid in selected.fixture_ids:
        public = selected.read(f"{fid}/public.json")
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        for method in MATCHED_METHODS:
            for policy in POLICIES:
                result, verified, stored = branches[fid, method, policy]
                score = raw_score(snapshots[fid, method], result, private)
                if stored != verified | {"raw_score": score}:
                    raise ValueError("paired raw audit mismatch")
                rows.append(
                    make_row(
                        fid,
                        method,
                        policy,
                        snapshots[fid, method],
                        result,
                        score,
                        selected,
                        metadata,
                    )
                )
    summary = report_summary(rows, selected)
    report = pinned("report.json")
    if report != {
        "schema": "paid-validation-development-report-v1",
        "registration": registration,
        "rows": rows,
        "summary": summary,
        "scope": SCOPE,
    }:
        raise ValueError("paired report/denominators mismatch")
    timing = read_json(directory / "timing.json")
    if [(r["fixture_id"], r["method"], r["policy"]) for r in timing["rows"]] != list(
        branches
    ):
        raise ValueError("paired timing coverage mismatch")
    for row in timing["rows"]:
        values = row["paid_replay_and_validation"]
        if set(values) != {"wall_seconds", "cpu_seconds"} or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in values.values()
        ):
            raise ValueError("invalid paired timing")
    return {
        "valid": True,
        "summary": summary,
        "deterministic_artifacts": len(expected),
        "scope": SCOPE,
    }

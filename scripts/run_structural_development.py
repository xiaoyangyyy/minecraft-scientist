"""Paid four-method generated-world DEVELOPMENT only; sealed splits never read."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_matched_h3_development import hashes as upstream_hashes
from run_matched_h3_development import summarize
from run_online_automata_development import dump

from research.structural_campaigns.corpus import digest, load_development
from research.structural_campaigns.runtime import (
    MATCHED_METHODS,
    POLICY,
    PROTOCOL,
    audit,
    control_groups,
    learn,
    prelude,
    public_config,
)
from research.structural_holdouts.audit_corpus import validate_fixture
from research.structural_holdouts.generator import instantiate


def hashes():
    owned = (
        list((ROOT / "research" / "structural_campaigns").glob("*.py"))
        + list((ROOT / "research" / "structural_campaigns").glob("*.md"))
        + list((ROOT / "research" / "structural_holdouts").glob("*.py"))
        + list((ROOT / "research" / "structural_holdouts").glob("*.md"))
        + [Path(__file__).resolve(), ROOT / "scripts" / "build_structural_corpus.py"]
    )
    return upstream_hashes() | {
        p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(owned)
    }


def run(*, corpus, manifest_sha256, output, max_groups=None, protocol=None):
    protocol = dict(PROTOCOL if protocol is None else protocol)
    selected = load_development(
        corpus, manifest_sha256=manifest_sha256, max_groups=max_groups
    )
    publics = {fid: selected.read(f"{fid}/public.json") for fid in selected.fixture_ids}
    # Invalid protocol must not create a directory or open selected truth.
    for public in publics.values():
        public_config(public, MATCHED_METHODS[0], protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = hashes()
    scope = "generated structures DEVELOPMENT only, complete pinned development groups, same paid lifecycle; no sealed evaluation, formal statistics, unknown controls/noise, H4, innovation or Minecraft/RGB completion"
    registration = {
        "schema": "structural-campaign-development-registration-v1",
        "source_sha256": sources,
        "corpus_manifest_sha256": selected.manifest_sha256,
        "selection": {
            "split": "development",
            "policy": "all members of first lexicographically sorted development group IDs; no case/outcome filtering",
            "requested_max_groups": max_groups,
            "available_development_groups": selected.available_groups,
            "fixture_ids": list(selected.fixture_ids),
            "groups": selected.groups,
        },
        "methods": list(MATCHED_METHODS),
        "protocol": protocol,
        "repair_policy": POLICY,
        "knowledge_boundary": "same public before factor family; no fixture ID/group/metadata/split/seed/after/boundary enters learner; edit-language versus general-Mealy bias disclosed",
        "scope": scope,
    }
    dump(output / "registration.json", registration)
    rows, timings = [], []
    for fid in selected.fixture_ids:
        public = publics[fid]
        metadata = selected.read(f"{fid}/metadata.json")
        private = selected.read(f"{fid}/evaluator/world.json")
        validate_fixture(public, private, metadata)
        baseline = None
        for method in MATCHED_METHODS:
            directory = output / fid / method
            directory.mkdir(parents=True)
            config = public_config(public, method, protocol)
            dump(directory / "config.json", config)
            wall, cpu = perf_counter(), process_time()
            result = learn(instantiate(private), config)
            learning_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "run.json", result)  # Freeze before private score.
            current = prelude(result)
            if baseline is not None and current != baseline:
                raise RuntimeError("common prelude mismatch; preserve partial evidence")
            baseline = current
            dump(directory / "theory_history.json", result["theory_events"])
            dump(directory / "evidence.json", list(control_groups(result)))
            private_path = directory / "evaluator" / "world.json"
            private_path.parent.mkdir()
            dump(private_path, private)
            wall, cpu = perf_counter(), process_time()
            verified = audit(config, result, private_path)
            audit_time = {
                "wall_seconds": perf_counter() - wall,
                "cpu_seconds": process_time() - cpu,
            }
            dump(directory / "audit.json", verified)
            if not verified["valid"]:
                raise RuntimeError(
                    f"public/raw audit failed: {fid}/{method}: {verified}"
                )
            row = {
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
            rows.append(row)
            timings.append(
                {"fixture_id": fid, "method": method}
                | {"complete_learning": learning_time, "complete_audit": audit_time}
            )
        # Recheck selected private bytes after all methods; no feedback to them.
        selected.read(f"{fid}/evaluator/world.json")
        print(f"completed development fixture {fid}", flush=True)
    if sources != hashes() or digest(Path(corpus) / "manifest.json") != manifest_sha256:
        raise RuntimeError("source/corpus drift; retain partial data without manifest")
    summary = {
        "fixtures": len(selected.fixture_ids),
        "independent_development_groups": len(set(selected.groups.values())),
        "campaigns": len(rows),
        "methods": {
            method: summarize([r for r in rows if r["method"] == method])
            for method in MATCHED_METHODS
        },
    }
    report = {
        "schema": "structural-campaign-development-report-v1",
        "registration": registration,
        "rows": rows,
        "summary": summary,
        "scope": scope,
        "limitations": "all failure/miss/wrong/unresolved denominators retained; bounded validation is not all-H proof; independent unit is leakage group, not names or repeats; no statistical comparison",
    }
    dump(output / "report.json", report)
    dump(output / "timing.json", {"rows": timings, "scope": "nondeterministic"})
    manifest = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "timing.json"
    }
    dump(
        output / "manifest.json",
        {
            "schema": "structural-campaign-development-manifest-v1",
            "sha256": manifest,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int)
    run(**vars(parser.parse_args()))

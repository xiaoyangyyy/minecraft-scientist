"""Verify that a failed RGB goal-search decision preserved accepted memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_natural_visual_goal_search import (
    _digest,
    _hashes_valid,
    _load,
    _panorama,
    _physical,
    _scan,
    _source_paths,
)
from natural_visual_loop_benchmark import PANORAMA_ENCODERS, PRIVILEGED_ARTIFACT_TOKENS

from minecraft_scientist.exploration import ExplorationMemory
from minecraft_scientist.protocols import TransitionRecord


def audit(root: Path) -> dict[str, Any]:
    import math

    report = _load(root / "report.json")
    config = _load(root / "config.json")
    evaluator = _load(root / "evaluator-audit.json")
    spec = config["source_spec"]
    encoder = PANORAMA_ENCODERS["edge"]()
    period = int(spec["period"])
    goal, _views, _frames = _panorama(root / "goal-reference", encoder, period)
    signatures = {}
    valid_scans = True
    for role, metadata in report["scans"].items():
        signatures[role], _views, valid = _scan(root, role, metadata, encoder, period)
        valid_scans = valid_scans and valid
    ranking = []
    inputs_valid = True
    localizations_valid = True
    metrics_valid = True
    with ExplorationMemory.open(root / "source-evidence/exploration.sqlite3") as memory:
        for candidate in report["candidate_results"]:
            for node in candidate["replay"]["localizations"]:
                match = memory.match_modality_place(
                    spec["encoder"], signatures[node["scan_role"]]
                )
                localizations_valid = (
                    localizations_valid
                    and match == node["place_id"] == node["recognized_place_id"]
                )
            signature = signatures[candidate["target_scan_role"]]
            source = signatures[candidate["replay"]["localizations"][-1]["scan_role"]]
            match = memory.match_modality_place(spec["encoder"], signature)
            inputs_valid = inputs_valid and (
                list(signature) == candidate["signature"]
                and list(source) == candidate["source_signature"]
                and match == candidate["recognized_place_id"]
            )
            if match in (None, spec["goal"]):
                ranking.append(
                    {
                        "candidate": {
                            "candidate_id": candidate["candidate_id"],
                            "heading_steps": candidate["heading_steps"],
                            "signature": list(signature),
                            "recognized_place_id": match,
                        },
                        "goal_visual_distance": math.dist(goal, signature),
                        "source_visual_distance": math.dist(source, signature),
                    }
                )
            metrics = _physical(
                candidate,
                evaluator["sessions"][candidate["session_index"]]["positions"],
                config,
                spec["goal"],
            )
            metrics_valid = (
                metrics_valid
                and metrics == evaluator["probe_metrics"][candidate["candidate_id"]]
            )
    ranking.sort(
        key=lambda item: (
            item["goal_visual_distance"],
            -item["source_visual_distance"],
            item["candidate"]["heading_steps"],
            item["candidate"]["candidate_id"],
        )
    )
    selected = report["selected_result"]
    selected_metrics = evaluator["probe_metrics"][selected["candidate_id"]]
    public = (root / "events.jsonl").read_text(encoding="utf-8")
    records = [
        TransitionRecord.model_validate_json(line)
        for line in public.splitlines()
        if line
    ]
    current_paths = _source_paths()
    source_audit = _load(root / "source-evidence/integrity-audit.json")
    with ExplorationMemory.open(root / "exploration.sqlite3") as memory:
        memory_valid = memory.verify_integrity()
    checks = {
        "source_evidence_accepted": source_audit["valid"] is True
        and all(source_audit["checks"].values()),
        "source_evidence_hashes_valid": _hashes_valid(
            root / "source-evidence", report["source_evidence_sha256"]
        ),
        "frozen_sources_valid": _hashes_valid(root / "source", report["source_sha256"]),
        "current_sources_match_frozen": set(current_paths)
        == set(report["source_sha256"])
        and all(
            _digest(current_paths[name]) == expected
            for name, expected in report["source_sha256"].items()
        ),
        "reference_and_goal_hashes_valid": _hashes_valid(
            root / "references", report["initial_reference_sha256"]
        )
        and _hashes_valid(root / "goal-reference", report["goal_reference_sha256"]),
        "scans_recomputed_from_png": valid_scans,
        "all_prior_route_nodes_recognized": localizations_valid,
        "visual_inputs_recomputed": inputs_valid,
        "rgb_ranking_recomputed": ranking == report["ranking"]
        and ranking[0]["candidate"]["candidate_id"] == selected["candidate_id"],
        "physical_metrics_recomputed": metrics_valid,
        "selected_candidate_rejected_by_physics": selected_metrics["accepted"] is False,
        "no_evaluator_runner_up_substitution": report["place_ids"]["target"] is None
        and "validation_result" not in report
        and "frontier_id" not in report
        and "persisted_route" not in report,
        "memory_byte_identical_to_source": (root / "exploration.sqlite3").read_bytes()
        == (root / "source-evidence/exploration.sqlite3").read_bytes(),
        "memory_integrity": memory_valid,
        "gate_reports_failure": report["gate"]["passed"] is False
        and report["gate"]["selected_physical_progress"] is False,
        "actions_compliant": bool(records)
        and all(
            item.intervention_compliance and item.event.success for item in records
        ),
        "public_telemetry_clean": not any(
            token in public for token in PRIVILEGED_ARTIFACT_TOKENS
        ),
    }
    return {
        "suite": "visual_goal_search_rejection_audit",
        "valid": all(checks.values()),
        "checks": checks,
        "root": str(root.resolve()),
        "auditor_sha256": _digest(Path(__file__).resolve()),
        "selected_heading_steps": selected["heading_steps"],
        "selected_physical_metrics": selected_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.write:
        (args.root / "rejection-audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

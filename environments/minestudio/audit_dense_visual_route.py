"""Recompute dense RGB acquisition, reference-scale routing and memory isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from minecraft_scientist.evaluation.visual_evidence import (
    read_panorama,
    verify_temporal_panorama,
)
from minecraft_scientist.exploration import ExplorationMemory, align_panorama_heading
from minecraft_scientist.exploration.perception import RGBEdgePanoramaEncoder
from minecraft_scientist.exploration.resolution import ResolutionNormalizedSceneEncoder
from minecraft_scientist.protocols import TransitionRecord

PRIVILEGED_ARTIFACT_TOKENS = (
    "location_stats",
    "xpos",
    "ypos",
    "zpos",
    '"yaw"',
    '"pitch"',
)


def _source_paths() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    return {
        name: here / name
        for name in (
            "natural_visual_goal_search_benchmark.py",
            "audit_natural_visual_goal_search.py",
            "natural_visual_expedition_benchmark.py",
            "natural_visual_loop_benchmark.py",
            "navigation_probe.py",
        )
    } | {
        name: repository / "src/minecraft_scientist" / relative
        for name, relative in {
            "goal_search.py": "exploration/goal_search.py",
            "frontier_program.py": "exploration/frontier_program.py",
            "memory.py": "exploration/memory.py",
            "perception.py": "exploration/perception.py",
            "minestudio.py": "minecraft/minestudio.py",
        }.items()
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, source: Path) -> dict:
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    spec = report["source_spec"]
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    source_audit = json.loads(
        (source / "integrity-audit.json").read_text(encoding="utf-8")
    )
    checks = {
        "supported_source_suite": source_report["suite"]
        == "natural_visual_goal_search",
        "reported_acquisition_passed": report["gate"]["passed"] is True,
        "accepted_source": source_report["gate"]["passed"] is True
        and source_audit["valid"] is True
        and all(source_audit["checks"].values()),
        "source_evidence_byte_identical": all(
            _hash(root / "source-evidence" / name) == _hash(source / name)
            for name in ("report.json", "integrity-audit.json", "exploration.sqlite3")
        ),
        "accepted_memory_byte_identical": _hash(root / "exploration.sqlite3")
        == _hash(source / "exploration.sqlite3"),
    }
    paths = _source_paths() | {
        "capture_visual_odometry_route.py": Path(__file__).with_name(
            "capture_visual_odometry_route.py"
        ),
        "resolution.py": Path(
            ResolutionNormalizedSceneEncoder.encode.__code__.co_filename
        ),
    }
    hashes = report["source_sha256"]
    checks["runtime_source_file_set"] = set(paths) == set(hashes)
    checks["runtime_source_snapshot_hashes"] = checks[
        "runtime_source_file_set"
    ] and all(
        _hash(root / "source-snapshot" / name) == digest
        for name, digest in hashes.items()
    )
    checks["current_runtime_sources_match"] = checks["runtime_source_file_set"] and all(
        _hash(path) == hashes[name] for name, path in paths.items()
    )
    frame_count = int(report["frame_count"])
    names = {f"frame-{index:05d}.png" for index in range(frame_count)}
    checks["dense_frame_file_set"] = (
        names
        == set(report["frame_sha256"])
        == {path.name for path in (root / "frames").glob("*.png")}
    )
    checks["rgb_ledger_length"] = len(report["rgb_sha256"]) == frame_count
    frames_valid = checks["rgb_ledger_length"]
    for index in range(frame_count):
        path = root / "frames" / f"frame-{index:05d}.png"
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        frames_valid &= _hash(path) == report["frame_sha256"][path.name]
        frames_valid &= rgb.shape == (
            report["observation_size"],
            report["observation_size"],
            3,
        )
        if index < len(report["rgb_sha256"]):
            frames_valid &= (
                hashlib.sha256(rgb.tobytes()).hexdigest() == report["rgb_sha256"][index]
            )
    checks["dense_png_and_rgb_hashes"] = bool(frames_valid)
    public = (root / "events.jsonl").read_text(encoding="utf-8")
    records = [
        TransitionRecord.model_validate_json(line)
        for line in public.splitlines()
        if line
    ]
    checks["public_telemetry_clean"] = not any(
        token in public for token in PRIVILEGED_ARTIFACT_TOKENS
    )
    checks["actions_compliant"] = bool(records) and all(
        record.intervention_compliance and record.event.success for record in records
    )
    checks["timeline_length"] = len(records) == len(report["timeline"])
    index, aligned = 0, True
    for record, item in zip(records, report["timeline"], strict=False):
        start = index
        index += int(record.event.duration)
        aligned &= (
            item["start_frame"] == start
            and item["end_frame"] == index
            and item["transition_id"] == record.transition_id
            and item["action"] == record.executed_action.model_dump(mode="json")
        )
        if index >= frame_count:
            aligned = False
            continue
        aligned &= (
            record.state_before.agent_state["frame_sha256"]
            == report["rgb_sha256"][start]
        )
        aligned &= (
            record.state_after.agent_state["frame_sha256"]
            == report["rgb_sha256"][index]
        )
    checks["action_frame_alignment"] = bool(aligned) and index == frame_count - 1
    encoder = RGBEdgePanoramaEncoder()
    with Image.open(root / "goal-reference/view-00.png") as reference:
        reference_size = reference.size
    checks["reference_resolution_matches_metadata"] = (
        list(reference_size) == report["policy_reference_size"]
    )
    if reference_size != (report["observation_size"], report["observation_size"]):
        encoder.scene_encoder = ResolutionNormalizedSceneEncoder(
            encoder.scene_encoder, size=reference_size
        )
    scans = {
        role: verify_temporal_panorama(
            root / "scans" / role, metadata, encoder, spec["period"]
        )
        for role, metadata in report["scans"].items()
    }
    checks["scan_hashes_medians_and_features"] = all(
        values[2] for values in scans.values()
    )
    replay = report["replay"]
    route_valid, localization_valid, alignment_valid = False, False, False
    if replay is not None:
        memory_path = root / "exploration.sqlite3"
        before_hash = _hash(memory_path)
        with ExplorationMemory.open(memory_path) as memory:
            checks["memory_integrity"] = memory.verify_integrity()
            planned = memory.plan_route(spec["origin"], spec["source"])
            route_valid = (
                planned is not None and planned.model_dump(mode="json") == spec["route"]
            )
            localizations = replay["localizations"]
            localization_valid = [item["place_id"] for item in localizations] == spec[
                "route"
            ]["place_path"]
            for item in localizations:
                recognized = memory.match_modality_place(
                    spec["encoder"], scans[item["scan_role"]][0]
                )
                localization_valid &= (
                    recognized == item["place_id"] == item["recognized_place_id"]
                )
            alignment_valid = len(replay["executions"]) == len(localizations) - 1
            for step, execution in enumerate(replay["executions"]):
                reference = root / "references" / localizations[step]["place_id"]
                _, reference_views, _ = read_panorama(
                    reference, encoder, spec["period"]
                )
                query_views = scans[execution["source_scan_role"]][1]
                alignment = align_panorama_heading(reference_views, query_views)
                alignment_valid &= (
                    alignment.model_dump(mode="json") == execution["alignment"]
                )
                alignment_valid &= (
                    execution["action_roles"]
                    == spec["route"]["travel_action_roles"][step]
                )
        checks["auditing_did_not_modify_memory"] = _hash(memory_path) == before_hash
    checks["persistent_route_recomputed"] = bool(route_valid)
    checks["all_route_nodes_recognized_from_png"] = bool(localization_valid)
    checks["heading_alignments_recomputed"] = bool(alignment_valid)
    # Load privileged telemetry only after the independent public checks.
    waypoint_errors = {}
    if all(checks.values()):
        positions = json.loads(
            (root / "evaluator-audit.json").read_text(encoding="utf-8")
        )["positions"]
        checks["evaluator_frame_count"] = len(positions) == frame_count
        source_truth = json.loads(
            (source / "evaluator-audit.json").read_text(encoding="utf-8")
        )
        restart = source_report["restart_result"]
        original_positions = source_truth["sessions"][restart["session_index"]][
            "positions"
        ]
        source_nodes = {
            item["place_id"]: original_positions[item["position_index"]]
            for item in restart["replay"]["localizations"]
        }
        tolerance = json.loads((source / "config.json").read_text(encoding="utf-8"))[
            "goal_tolerance"
        ]
        for item in replay["localizations"]:
            observed = positions[item["position_index"]]
            expected = source_nodes[item["place_id"]]
            waypoint_errors[item["place_id"]] = math.dist(
                tuple(observed[key] for key in ("xpos", "ypos", "zpos")),
                tuple(expected[key] for key in ("xpos", "ypos", "zpos")),
            )
        checks["physical_waypoints_match_accepted_source"] = all(
            error <= tolerance for error in waypoint_errors.values()
        )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "frame_count": frame_count,
        "suite": "dense_visual_route_integrity_audit",
        "auditor_sha256": _hash(Path(__file__)),
        "waypoint_errors_3d": waypoint_errors,
        "evidence_verifier_sha256": _hash(
            Path(verify_temporal_panorama.__code__.co_filename)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite dense acquisition audit")
    result = audit(args.run, args.source_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

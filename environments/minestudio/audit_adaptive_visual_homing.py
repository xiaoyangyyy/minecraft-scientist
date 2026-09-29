"""Independent artifact audit for adaptive_visual_homing_benchmark.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
)
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    RGBEdgeCompactSpectrumPanoramaEncoder,
    RGBEdgeDualBandSpectrumPanoramaEncoder,
    RGBEdgeExpandedSpectrumPanoramaEncoder,
    RGBEdgeSpectrumPanoramaEncoder,
    VisualEvidenceHomingProgram,
    VisualFrontierProgram,
    VisualHomingProgram,
    align_panorama_heading,
    temporal_median_rgb,
)

HOMING_ENCODERS = {
    RGBEdgeSpectrumPanoramaEncoder.VERSION: RGBEdgeSpectrumPanoramaEncoder,
    RGBEdgeCompactSpectrumPanoramaEncoder.VERSION: (
        RGBEdgeCompactSpectrumPanoramaEncoder
    ),
    RGBEdgeDualBandSpectrumPanoramaEncoder.VERSION: (
        RGBEdgeDualBandSpectrumPanoramaEncoder
    ),
    RGBEdgeExpandedSpectrumPanoramaEncoder.VERSION: (
        RGBEdgeExpandedSpectrumPanoramaEncoder
    ),
}
ARRIVAL_TOLERANCE_3D = 0.75
HomingProgram = VisualHomingProgram | VisualEvidenceHomingProgram


def _homing_program_from_payload(payload: dict[str, Any]) -> HomingProgram:
    if "evidence_window" in payload or "required_matches" in payload:
        return VisualEvidenceHomingProgram.model_validate(payload)
    return VisualHomingProgram.model_validate(payload)


def _homing_program_from_roles(roles: tuple[str, ...]) -> HomingProgram:
    if roles and roles[0] == VisualEvidenceHomingProgram.VERSION:
        return VisualEvidenceHomingProgram.from_roles(roles)
    return VisualHomingProgram.from_roles(roles)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature_digest(signature: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metrics(positions: list[dict[str, float]]) -> dict[str, float | None]:
    compatible = [item for item in positions if "xpos" in item and "zpos" in item]
    if not compatible:
        return {"maximum_excursion": None, "return_error": None}
    start, end = compatible[0], compatible[-1]
    return {
        "maximum_excursion": max(
            math.hypot(item["xpos"] - start["xpos"], item["zpos"] - start["zpos"])
            for item in compatible
        ),
        "return_error": math.hypot(
            end["xpos"] - start["xpos"], end["zpos"] - start["zpos"]
        ),
    }


def _close_metrics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(key) is not None
        and right.get(key) is not None
        and math.isclose(float(left[key]), float(right[key]), abs_tol=1e-12)
        for key in ("maximum_excursion", "return_error")
    )


def _position_distance_3d(
    positions: list[dict[str, float]], left_index: int, right_index: int
) -> float:
    left = positions[left_index]
    right = positions[right_index]
    return math.sqrt(
        (float(left["xpos"]) - float(right["xpos"])) ** 2
        + (float(left["ypos"]) - float(right["ypos"])) ** 2
        + (float(left["zpos"]) - float(right["zpos"])) ** 2
    )


def _load_scan(
    root: Path,
    role: str,
    metadata: dict[str, Any],
    *,
    encoder: Any,
    period: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], bool]:
    frames = []
    valid = True
    directory = root / "scans" / role
    for index in range(period):
        path = directory / f"view-{index:02d}.png"
        valid = valid and path.is_file() and _digest(path) == metadata["image_sha256"][
            path.name
        ]
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    panorama = encoder.encode(frames)
    views = tuple(encoder.scene_encoder.encode(frame) for frame in frames)
    valid = valid and int(metadata["view_count"]) == period
    valid = valid and _signature_digest(panorama) == metadata["panorama_sha256"]
    sample_count = metadata.get("temporal_sample_count")
    if sample_count is not None:
        for view_index, frame in enumerate(frames):
            view_name = f"view-{view_index:02d}"
            samples = []
            for sample_index in range(int(sample_count)):
                sample_name = f"sample-{sample_index:02d}.png"
                path = directory / "samples" / view_name / sample_name
                valid = valid and path.is_file() and _digest(path) == metadata[
                    "temporal_sample_sha256"
                ][view_name][sample_name]
                with Image.open(path) as image:
                    samples.append(np.asarray(image.convert("RGB")))
            valid = valid and bool(np.array_equal(temporal_median_rgb(samples), frame))
    return panorama, views, valid


def _load_reference_views(
    root: Path, role: str, *, encoder: Any, period: int
) -> tuple[tuple[int, ...], ...]:
    frames = []
    for index in range(period):
        with Image.open(root / "scans" / role / f"view-{index:02d}.png") as image:
            frames.append(np.asarray(image.convert("RGB")))
    return tuple(encoder.scene_encoder.encode(frame) for frame in frames)


def _source_paths(root: Path) -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "adaptive_visual_homing_benchmark.py": Path(__file__).with_name(
            "adaptive_visual_homing_benchmark.py"
        ),
        "audit_adaptive_visual_homing.py": Path(__file__).resolve(),
        "natural_visual_expedition_benchmark.py": Path(__file__).with_name(
            "natural_visual_expedition_benchmark.py"
        ),
        "natural_visual_loop_benchmark.py": Path(__file__).with_name(
            "natural_visual_loop_benchmark.py"
        ),
        "navigation_probe.py": Path(__file__).with_name("navigation_probe.py"),
        "frontier_program.py": repository
        / "src/minecraft_scientist/exploration/frontier_program.py",
        "exploration_init.py": repository
        / "src/minecraft_scientist/exploration/__init__.py",
        "memory.py": repository / "src/minecraft_scientist/exploration/memory.py",
        "perception.py": repository
        / "src/minecraft_scientist/exploration/perception.py",
        "minestudio.py": repository
        / "src/minecraft_scientist/minecraft/minestudio.py",
    }


def audit(root: Path) -> dict[str, Any]:
    report = _load(root / "report.json")
    evaluator = _load(root / "evaluator-audit.json")
    source_hashes = _load(root / "source-evidence-sha256.json")
    source_root = root / "source-evidence"
    source_evidence_valid = all(
        (source_root / name).is_file()
        and _digest(source_root / name) == expected
        for name, expected in source_hashes.items()
    )
    source_report = _load(source_root / "report.json")
    source_audit = _load(source_root / "integrity-audit.json")
    source_accepted = (
        source_report.get("suite") == "natural_visual_expedition"
        and source_report.get("gate", {}).get("passed") is True
        and source_audit.get("valid") is True
        and source_report.get("terrain_modified") is False
    )

    encoder_key = "edge" if str(report["encoder"]).startswith("rgb-edge") else "lab"
    encoder = PANORAMA_ENCODERS[encoder_key]()
    homing_edge_count = int(report["homing_edge_count"])
    homing_encoder_versions = tuple(
        report.get(
            "homing_encoders", [report["homing_encoder"]] * homing_edge_count
        )
    )
    if len(homing_encoder_versions) != homing_edge_count or any(
        version not in HOMING_ENCODERS for version in homing_encoder_versions
    ):
        raise ValueError("unsupported homing encoder in audit artifact")
    homing_encoders = tuple(
        HOMING_ENCODERS[version]() for version in homing_encoder_versions
    )
    period = int(report["inferred_camera_period"])
    panoramas = {}
    views = {}
    scans_valid = True
    for role, metadata in report["scans"].items():
        panorama, scan_views, valid = _load_scan(
            root, role, metadata, encoder=encoder, period=period
        )
        panoramas[role] = panorama
        views[role] = scan_views
        scans_valid = scans_valid and valid
    reference_views = {
        place_id: _load_reference_views(
            source_root, role, encoder=encoder, period=period
        )
        for place_id, role in report["source_reference_roles"].items()
    }
    homing_calibration_valid = True
    homing_calibrations = report.get(
        "homing_calibrations",
        {report["homing_encoder"]: report["homing_calibration"]},
    )
    unique_homing_encoders = {
        encoder.VERSION: encoder for encoder in homing_encoders
    }
    homing_calibration_valid = homing_calibration_valid and (
        set(homing_calibrations) == set(unique_homing_encoders)
    )
    for version, homing_encoder in unique_homing_encoders.items():
        for item in homing_calibrations[version].values():
            positives = [
                homing_encoder.encode_signatures(
                    _load_reference_views(
                        source_root, role, encoder=homing_encoder, period=period
                    )
                )
                for role in item["positive_roles"]
            ]
            negatives = [
                homing_encoder.encode_signatures(
                    _load_reference_views(
                        source_root, role, encoder=homing_encoder, period=period
                    )
                )
                for role in item["negative_roles"]
            ]
            positive_distance = max(
                math.dist(positives[0], signature) for signature in positives
            )
            negative_distance = min(
                math.dist(positives[0], signature) for signature in negatives
            )
            margin = (negative_distance - positive_distance) * 0.499999
            boundary = item["boundary"]
            homing_calibration_valid = homing_calibration_valid and all(
                math.isclose(float(actual), expected, abs_tol=1e-12)
                for actual, expected in (
                    (boundary["maximum_positive_distance"], positive_distance),
                    (boundary["nearest_negative_distance"], negative_distance),
                    (boundary["margin"], margin),
                    (boundary["radius"], positive_distance + margin),
                )
            )

    program = _homing_program_from_payload(report["homing_program"])
    programs = tuple(
        _homing_program_from_payload(item)
        for item in report.get(
            "homing_programs", [report["homing_program"]] * homing_edge_count
        )
    )
    roles_by_edge = report.get(
        "homing_action_roles_by_edge",
        [report["homing_action_roles"]] * homing_edge_count,
    )
    roles_valid = list(program.to_roles()) == report["homing_action_roles"]
    roles_valid = roles_valid and len(programs) == homing_edge_count
    roles_valid = roles_valid and len(roles_by_edge) == homing_edge_count
    roles_valid = roles_valid and all(
        list(edge_program.to_roles()) == edge_roles
        and edge_program.turn_steps < period
        for edge_program, edge_roles in zip(programs, roles_by_edge, strict=True)
    )
    roles_valid = roles_valid and not any(
        token in role
        for edge_program in programs
        for role in edge_program.to_roles()
        for token in ("forward", "jump", "back", "left", "right", "obj_")
    )
    memory_path = root / "exploration.sqlite3"
    recognition_valid = True
    alignments_valid = True
    stopping_valid = True
    with ExplorationMemory.open(memory_path) as memory:
        for phase in ("discovery", "validation"):
            phase_result = report[phase]
            for execution in phase_result["outward_executions"]:
                parsed = VisualFrontierProgram.from_roles(
                    tuple(execution["action_roles"])
                )
                roles_valid = roles_valid and parsed.turn_steps < period
                alignment = align_panorama_heading(
                    reference_views[
                        next(
                            place_id
                            for place_id, role in report[
                                "source_reference_roles"
                            ].items()
                            if role == execution["reference_scan_role"]
                        )
                    ],
                    views[execution["source_scan_role"]],
                )
                alignments_valid = alignments_valid and (
                    alignment.model_dump(mode="json") == execution["alignment"]
                )
            for edge_index, edge in enumerate(phase_result["homing_edges"]):
                homing_encoder = homing_encoders[edge_index]
                edge_program = _homing_program_from_roles(tuple(edge["action_roles"]))
                roles_valid = roles_valid and edge_program == programs[edge_index]
                roles_valid = roles_valid and (
                    edge.get("homing_encoder", report["homing_encoder"])
                    == homing_encoder.VERSION
                )
                alignment = align_panorama_heading(
                    reference_views[edge["source_place_id"]],
                    views[edge["source_scan_role"]],
                )
                alignments_valid = alignments_valid and (
                    alignment.model_dump(mode="json") == edge["alignment"]
                )
                first_acceptance = None
                target_evidence: list[bool] = []
                for index, step in enumerate(edge["steps"], start=1):
                    signature = homing_encoder.encode_signatures(
                        _load_reference_views(
                            root,
                            step["scan_role"],
                            encoder=homing_encoder,
                            period=period,
                        )
                    )
                    match = memory.match_modality_place(
                        homing_encoder.VERSION, signature
                    )
                    score = memory.score_modality_place(
                        homing_encoder.VERSION, edge["target_place_id"], signature
                    )
                    target_evidence.append(match == edge["target_place_id"])
                    evidence_match_count = edge_program.evidence_match_count(
                        target_evidence
                    )
                    recognition_valid = recognition_valid and (
                        match == step["recognized_place_id"]
                        and score.model_dump(mode="json") == step["target_score"]
                        and step["acceptance_eligible"]
                        == (index >= edge_program.min_repetitions)
                        and step.get(
                            "evidence_match_count",
                            int(match == edge["target_place_id"]),
                        )
                        == evidence_match_count
                    )
                    if edge_program.should_stop(target_evidence) and first_acceptance is None:
                        first_acceptance = index
                stopping_valid = stopping_valid and (
                    edge["used_repetitions"] == len(edge["steps"])
                    and edge["success"] is True
                    and first_acceptance == len(edge["steps"])
                    and len(edge["steps"]) <= edge_program.max_repetitions
                )
        memory_integrity = memory.verify_integrity()
        memory_summary_valid = memory.summary() == report["memory"]
        all_place_ids = tuple(report["place_ids"][key] for key in "abcd")
        place_ids = tuple(report.get("route_place_ids", all_place_ids))
        expected_home = tuple(reversed(place_ids))[: homing_edge_count + 1]
        return_plan = memory.plan_route(expected_home[0], expected_home[-1])
        return_plan_valid = return_plan is not None and (
            return_plan.model_dump(mode="json") == report["return_plan"]
            and return_plan.place_path == expected_home
        )
        reverse_transitions = {
            (item.source_place_id, item.target_place_id)
            for item in memory.transitions()
            if item.action_roles
            and item.action_roles[0]
            in (
                VisualHomingProgram.VERSION,
                VisualEvidenceHomingProgram.VERSION,
            )
        }
        reverse_edges_valid = reverse_transitions == set(pairwise(expected_home))

    evaluator_metrics_valid = True
    physical_arrivals_valid = math.isclose(
        float(report.get("arrival_tolerance_3d", math.inf)),
        ARRIVAL_TOLERANCE_3D,
        abs_tol=1e-12,
    )
    for phase in ("discovery", "validation"):
        session_index = int(evaluator[f"{phase}_session_index"])
        positions = evaluator["sessions"][session_index]["positions"]
        recomputed = _metrics(positions)
        evaluator_metrics_valid = evaluator_metrics_valid and _close_metrics(
            recomputed, evaluator[f"{phase}_metrics"]
        )
        evaluator_metrics_valid = evaluator_metrics_valid and _close_metrics(
            recomputed, report[phase]["evaluator_metrics"]
        )
        place_indices = dict(
            zip(
                (report["place_ids"][key] for key in "abcd"),
                report[phase]["outward_position_indices"],
                strict=True,
            )
        )
        for edge in report[phase]["homing_edges"]:
            target_distance = _position_distance_3d(
                positions,
                int(edge["evaluator_position_index"]),
                int(place_indices[edge["target_place_id"]]),
            )
            evaluator_metrics_valid = evaluator_metrics_valid and math.isclose(
                target_distance,
                float(edge["evaluator_metrics"]["target_distance_3d"]),
                abs_tol=1e-12,
            )
            physical_arrivals_valid = physical_arrivals_valid and (
                target_distance <= ARRIVAL_TOLERANCE_3D
            )
    evaluator_separate = all(
        any(key in item["raw_metadata_keys"] for key in ("xpos", "location_stats"))
        for item in evaluator["sessions"]
    )
    public_text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("events.jsonl", "memory-events.jsonl")
    )
    public_clean = not any(token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS)
    source_snapshot_valid = all(
        (root / "source" / name).is_file()
        and _digest(root / "source" / name) == expected
        for name, expected in report["source_sha256"].items()
    )
    current_paths = _source_paths(root)
    current_sources_valid = all(
        name in current_paths
        and current_paths[name].is_file()
        and _digest(current_paths[name]) == expected
        for name, expected in report["source_sha256"].items()
    )
    all_place_ids = [report["place_ids"][key] for key in "abcd"]
    outward_edge_count = int(report.get("outward_edge_count", 3))
    expected_outward = all_place_ids[: outward_edge_count + 1]
    route_place_ids_valid = report.get("route_place_ids", all_place_ids) == (
        expected_outward
    )
    expected_home = list(reversed(expected_outward))[
        : int(report["homing_edge_count"]) + 1
    ]
    report_relations_valid = route_place_ids_valid and all(
        report[phase]["outward_matches"] == expected_outward
        and report[phase]["homing_matches"] == expected_home
        and len(report[phase]["homing_edges"])
        == int(report["homing_edge_count"])
        for phase in ("discovery", "validation")
    )
    checks = {
        "current_source_matches_frozen": current_sources_valid,
        "evaluator_is_separate": evaluator_separate,
        "evaluator_metrics_recomputed": evaluator_metrics_valid,
        "homing_alignments_recomputed": alignments_valid,
        "homing_recognition_recomputed": recognition_valid,
        "homing_calibration_recomputed": homing_calibration_valid,
        "memory_integrity": memory_integrity,
        "memory_summary_valid": memory_summary_valid,
        "opaque_action_roles_valid": roles_valid,
        "physical_arrivals_within_tolerance": physical_arrivals_valid,
        "public_telemetry_clean": public_clean,
        "report_relations_valid": report_relations_valid,
        "return_plan_recomputed": return_plan_valid,
        "reverse_edges_valid": reverse_edges_valid,
        "scans_recomputed_from_png": scans_valid,
        "source_evidence_accepted": source_accepted,
        "source_evidence_hashes_valid": source_evidence_valid,
        "source_snapshot_valid": source_snapshot_valid,
        "stops_at_first_calibrated_match": stopping_valid,
        "reported_gate_passed": report["gate"]["passed"] is True
        and all(report["gate"][key] for key in report["gate"] if key != "passed"),
    }
    return {"checks": checks, "valid": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.write:
        (args.root / "integrity-audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

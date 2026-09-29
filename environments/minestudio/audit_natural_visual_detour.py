"""Independent artifact audit for natural_visual_detour_benchmark.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
    _candidate_headings,
    view_openness,
)
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    RGBEdgeDualBandSpectrumPanoramaEncoder,
    VisualFrontierProgram,
    align_panorama_heading,
    temporal_median_rgb,
)

RETURN_TOLERANCE_3D = 0.75


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature_digest(signature: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_paths() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "natural_visual_detour_benchmark.py": Path(__file__).with_name(
            "natural_visual_detour_benchmark.py"
        ),
        "audit_natural_visual_detour.py": Path(__file__).resolve(),
        "adaptive_visual_homing_benchmark.py": Path(__file__).with_name(
            "adaptive_visual_homing_benchmark.py"
        ),
        "natural_visual_expedition_benchmark.py": Path(__file__).with_name(
            "natural_visual_expedition_benchmark.py"
        ),
        "natural_visual_loop_benchmark.py": Path(__file__).with_name(
            "natural_visual_loop_benchmark.py"
        ),
        "frontier_program.py": repository
        / "src/minecraft_scientist/exploration/frontier_program.py",
        "memory.py": repository / "src/minecraft_scientist/exploration/memory.py",
        "perception.py": repository
        / "src/minecraft_scientist/exploration/perception.py",
        "minestudio.py": repository
        / "src/minecraft_scientist/minecraft/minestudio.py",
    }


def _load_scan(
    root: Path,
    role: str,
    metadata: dict[str, Any],
    *,
    encoder: Any,
    period: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], bool]:
    directory = root / "scans" / role
    frames = []
    valid = True
    for index in range(period):
        path = directory / f"view-{index:02d}.png"
        valid = valid and path.is_file()
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
        valid = valid and _digest(path) == metadata["image_sha256"][path.name]
    panorama = encoder.encode(frames)
    views = tuple(encoder.scene_encoder.encode(frame) for frame in frames)
    valid = valid and int(metadata["view_count"]) == period
    valid = valid and _signature_digest(panorama) == metadata["panorama_sha256"]
    sample_count = int(metadata.get("temporal_sample_count", 1))
    for view_index, frame in enumerate(frames):
        view_name = f"view-{view_index:02d}"
        samples = []
        for sample_index in range(sample_count):
            sample_name = f"sample-{sample_index:02d}.png"
            path = directory / "samples" / view_name / sample_name
            valid = valid and path.is_file()
            valid = valid and _digest(path) == metadata["temporal_sample_sha256"][
                view_name
            ][sample_name]
            with Image.open(path) as image:
                samples.append(np.asarray(image.convert("RGB")))
        valid = valid and bool(np.array_equal(temporal_median_rgb(samples), frame))
    return panorama, views, valid


def _load_signature(
    root: Path, role: str, *, encoder: Any, period: int
) -> tuple[int, ...]:
    frames = []
    for index in range(period):
        with Image.open(root / "scans" / role / f"view-{index:02d}.png") as image:
            frames.append(np.asarray(image.convert("RGB")))
    return encoder.encode(frames)


def _recognition(
    memory: ExplorationMemory,
    modality: str,
    signature: tuple[int, ...],
    required_separation: float,
) -> tuple[str | None, dict[str, Any]]:
    strict_match = memory.match_modality_place(modality, signature)
    scores = sorted(
        (
            memory.score_modality_place(modality, state.place_id, signature)
            for state in memory.modality_places(modality)
        ),
        key=lambda item: (item.nearest_distance, item.place_id),
    )
    separation = (
        scores[1].nearest_distance - scores[0].nearest_distance
        if len(scores) > 1
        else math.inf
    )
    ranked_match = scores[0].place_id if separation >= required_separation else None
    return strict_match or ranked_match, {
        "strict_match": strict_match,
        "ranked_match": ranked_match,
        "nearest_separation": separation,
        "required_separation": required_separation,
        "ranked": [item.model_dump(mode="json") for item in scores],
    }


def _distance_3d(
    positions: list[dict[str, float]], left_index: int, right_index: int
) -> float:
    left, right = positions[left_index], positions[right_index]
    return math.sqrt(
        (float(left["xpos"]) - float(right["xpos"])) ** 2
        + (float(left["ypos"]) - float(right["ypos"])) ** 2
        + (float(left["zpos"]) - float(right["zpos"])) ** 2
    )


def _vertical_drop(
    positions: list[dict[str, float]], start_index: int, end_index: int
) -> float:
    start_y = float(positions[start_index]["ypos"])
    return max(
        start_y - float(item["ypos"])
        for item in positions[start_index : end_index + 1]
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), abs_tol=1e-12)


def audit(root: Path) -> dict[str, Any]:
    report = _load(root / "report.json")
    config = _load(root / "config.json")
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

    encoder = PANORAMA_ENCODERS["edge"]()
    recognition_encoder = RGBEdgeDualBandSpectrumPanoramaEncoder()
    period = int(report["inferred_camera_period"])
    panoramas: dict[str, tuple[int, ...]] = {}
    views: dict[str, tuple[tuple[int, ...], ...]] = {}
    scans_valid = True
    for role, metadata in report["scans"].items():
        panorama, scan_views, valid = _load_scan(
            root, role, metadata, encoder=encoder, period=period
        )
        panoramas[role] = panorama
        views[role] = scan_views
        scans_valid = scans_valid and valid
    source_a_views = []
    for index in range(period):
        with Image.open(
            source_root / "scans" / "train-a1" / f"view-{index:02d}.png"
        ) as image:
            source_a_views.append(
                encoder.scene_encoder.encode(np.asarray(image.convert("RGB")))
            )
    views["train-a1"] = tuple(source_a_views)

    place_b = str(report["source_place_id"])
    required_separation = float(report["ranked_recognition_margin"])
    recognition_valid = True
    roles_valid = True
    alignments_valid = True
    metrics_valid = True
    recomputed_candidates = []
    memory_path = root / "exploration.sqlite3"
    with ExplorationMemory.open(memory_path) as memory:
        for result in report["candidate_results"]:
            for key in (
                "replay_execution",
                "outbound_execution",
                "return_execution",
            ):
                execution = result[key]
                roles = tuple(str(item) for item in execution["action_roles"])
                try:
                    program = VisualFrontierProgram.from_roles(roles)
                    roles_valid = roles_valid and program.turn_steps < period
                    roles_valid = roles_valid and not any(
                        token in role
                        for role in roles
                        for token in (
                            "forward",
                            "jump",
                            "back",
                            "left",
                            "right",
                            "obj_",
                        )
                    )
                except ValueError:
                    roles_valid = False
                alignment = align_panorama_heading(
                    views[execution["reference_scan_role"]],
                    views[execution["source_scan_role"]],
                )
                alignments_valid = alignments_valid and (
                    alignment.model_dump(mode="json") == execution["alignment"]
                )

            b_role = str(result["b_scan_role"])
            e_role = str(result["e_scan_role"])
            return_role = str(result["return_scan_role"])
            outbound_distance = math.dist(panoramas[b_role], panoramas[e_role])
            return_distance = math.dist(panoramas[b_role], panoramas[return_role])
            score = outbound_distance / (return_distance + 1.0)
            recognition_signature = _load_signature(
                root, return_role, encoder=recognition_encoder, period=period
            )
            match, recognition = _recognition(
                memory,
                recognition_encoder.VERSION,
                recognition_signature,
                required_separation,
            )
            recognition_valid = recognition_valid and (
                match == result["return_match"]
                and recognition == result["return_recognition"]
                and bool(match == place_b) == result["visual_return_recognized"]
            )
            recomputed_candidates.append(
                {
                    "heading": int(result["heading_steps"]),
                    "visual": match == place_b,
                    "score": score,
                    "outbound": outbound_distance,
                }
            )
            metrics_valid = metrics_valid and all(
                _close(left, right)
                for left, right in (
                    (outbound_distance, result["outbound_distance"]),
                    (return_distance, result["return_distance"]),
                    (score, result["reversibility_score"]),
                )
            )
            session = evaluator["sessions"][int(result["session_index"])]
            positions = session["positions"]
            indices = result["evaluator"]
            outbound_3d = _distance_3d(
                positions,
                int(indices["b_position_index"]),
                int(indices["e_position_index"]),
            )
            return_3d = _distance_3d(
                positions,
                int(indices["b_position_index"]),
                int(indices["return_position_index"]),
            )
            drop = _vertical_drop(
                positions,
                int(indices["b_position_index"]),
                int(indices["return_position_index"]),
            )
            metrics_valid = metrics_valid and all(
                _close(left, right)
                for left, right in (
                    (outbound_3d, indices["outbound_distance_3d"]),
                    (return_3d, indices["return_error_3d"]),
                    (drop, indices["maximum_vertical_drop"]),
                )
            )

        validation = report["validation_result"]
        for key in ("replay_execution", "outbound_execution", "return_execution"):
            execution = validation[key]
            alignment = align_panorama_heading(
                views[execution["reference_scan_role"]],
                views[execution["source_scan_role"]],
            )
            alignments_valid = alignments_valid and (
                alignment.model_dump(mode="json") == execution["alignment"]
            )
        validation_signature = _load_signature(
            root,
            str(validation["return_scan_role"]),
            encoder=recognition_encoder,
            period=period,
        )
        validation_match, validation_recognition = _recognition(
            memory,
            recognition_encoder.VERSION,
            validation_signature,
            required_separation,
        )
        recognition_valid = recognition_valid and (
            validation_match == validation["return_match"]
            and validation_recognition == validation["return_recognition"]
            and validation_match == place_b
        )

        validation_positions = evaluator["sessions"][
            int(evaluator["validation_session_index"])
        ]["positions"]
        validation_indices = validation["evaluator"]
        validation_outbound = _distance_3d(
            validation_positions,
            int(validation_indices["b_position_index"]),
            int(validation_indices["e_position_index"]),
        )
        validation_return = _distance_3d(
            validation_positions,
            int(validation_indices["b_position_index"]),
            int(validation_indices["return_position_index"]),
        )
        validation_drop = _vertical_drop(
            validation_positions,
            int(validation_indices["b_position_index"]),
            int(validation_indices["return_position_index"]),
        )
        metrics_valid = metrics_valid and all(
            _close(left, right)
            for left, right in (
                (validation_outbound, validation_indices["outbound_distance_3d"]),
                (validation_return, validation_indices["return_error_3d"]),
                (validation_drop, validation_indices["maximum_vertical_drop"]),
            )
        )

        place_e = report["place_ids"]["e"]
        memory_integrity = memory.verify_integrity()
        memory_summary_valid = (
            memory.summary() == report["persisted_detour"]["summary"]
        )
        forward_plan = memory.plan_route(place_b, place_e)
        reverse_plan = memory.plan_route(place_e, place_b)
        plans_valid = (
            forward_plan is not None
            and reverse_plan is not None
            and forward_plan.model_dump(mode="json")
            == report["persisted_detour"]["forward"]
            and reverse_plan.model_dump(mode="json")
            == report["persisted_detour"]["reverse"]
        )
        frontiers_valid = all(
            memory.get_frontier(frontier_id).status.value == "resolved"
            for frontier_id in report["frontier_ids"].values()
        )

    eligible = [item for item in recomputed_candidates if item["visual"]]
    selected = max(
        eligible,
        key=lambda item: (item["score"], item["outbound"], -item["heading"]),
    )
    selector_valid = selected["heading"] == int(report["selected_heading_steps"])
    source_b_frames = []
    for index in range(period):
        with Image.open(
            source_root / "scans" / "train-b1" / f"view-{index:02d}.png"
        ) as image:
            source_b_frames.append(np.asarray(image.convert("RGB")))
    openness = tuple(view_openness(frame) for frame in source_b_frames)
    candidates = _candidate_headings(openness, int(config["candidate_count"]))
    candidate_generation_valid = (
        list(candidates) == report["candidate_headings"]
        and all(
            _close(left, right)
            for left, right in zip(openness, report["openness_scores"], strict=True)
        )
    )

    source_snapshot_valid = all(
        (root / "source" / name).is_file()
        and _digest(root / "source" / name) == expected
        for name, expected in report["source_sha256"].items()
    )
    current_paths = _source_paths()
    current_sources_match = set(current_paths) == set(report["source_sha256"]) and all(
        _digest(current_paths[name]) == expected
        for name, expected in report["source_sha256"].items()
    )
    public_text = (root / "events.jsonl").read_text(encoding="utf-8")
    public_clean = not any(token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS)
    evaluator_separate = all(
        any(key in item["raw_metadata_keys"] for key in ("xpos", "location_stats"))
        for item in evaluator["sessions"]
    )
    physical_gate_valid = (
        _close(report["return_tolerance_3d"], RETURN_TOLERANCE_3D)
        and validation_return <= RETURN_TOLERANCE_3D
        and float(report["selected_result"]["evaluator"]["return_error_3d"])
        <= RETURN_TOLERANCE_3D
        and validation_outbound >= 1.0
    )
    report_relations_valid = (
        report["suite"] == "natural_visual_detour"
        and report["backend"] == "real-minestudio"
        and report["terrain_modified"] is False
        and report["validation_result"]["heading_steps"]
        == report["selected_heading_steps"]
        and report["selected_result"]
        == next(
            item
            for item in report["candidate_results"]
            if item["heading_steps"] == report["selected_heading_steps"]
        )
    )
    checks = {
        "candidate_generation_recomputed": candidate_generation_valid,
        "candidate_metrics_recomputed": metrics_valid,
        "candidate_selection_recomputed": selector_valid,
        "current_source_matches_frozen": current_sources_match,
        "evaluator_is_separate": evaluator_separate,
        "execution_alignments_recomputed": alignments_valid,
        "memory_integrity": memory_integrity,
        "memory_summary_valid": memory_summary_valid,
        "opaque_action_roles_valid": roles_valid,
        "persistent_bidirectional_plans_recomputed": plans_valid,
        "physical_return_within_tolerance": physical_gate_valid,
        "public_telemetry_clean": public_clean,
        "recognition_recomputed": recognition_valid,
        "report_relations_valid": report_relations_valid,
        "resolved_frontiers_valid": frontiers_valid,
        "scans_recomputed_from_png": scans_valid,
        "source_evidence_accepted": source_accepted,
        "source_evidence_hashes_valid": source_evidence_valid,
        "source_snapshot_valid": source_snapshot_valid,
        "reported_gate_passed": report["gate"]["passed"] is True
        and all(report["gate"][key] for key in report["gate"] if key != "passed"),
    }
    return {
        "suite": "natural_visual_detour_audit",
        "root": str(root.resolve()),
        "checks": checks,
        "valid": all(checks.values()),
    }


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

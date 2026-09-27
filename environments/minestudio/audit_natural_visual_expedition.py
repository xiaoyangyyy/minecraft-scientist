"""Independent audit for a natural_visual_expedition evidence directory."""

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
)
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    VisualFrontierProgram,
    align_panorama_heading,
    temporal_median_rgb,
)

HEADING_MODALITY_SUFFIX = ".ordered-heading-v1"


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
    start = compatible[0]
    end = compatible[-1]
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


def _load_scans(
    root: Path, report: dict[str, Any]
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[tuple[int, ...], ...]], bool]:
    encoder = PANORAMA_ENCODERS[
        "edge" if str(report["encoder"]).startswith("rgb-edge") else "lab"
    ]()
    period = int(report["inferred_camera_period"])
    panoramas = {}
    views = {}
    valid = True
    for role, metadata in report["scans"].items():
        directory = root / "scans" / role
        frames = []
        for index in range(period):
            path = directory / f"view-{index:02d}.png"
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("RGB")))
            valid = valid and (
                _digest(path) == metadata["image_sha256"][path.name]
            )
        panorama = encoder.encode(frames)
        scene_views = tuple(encoder.scene_encoder.encode(frame) for frame in frames)
        panoramas[role] = panorama
        views[role] = scene_views
        valid = valid and int(metadata["view_count"]) == period
        valid = valid and _signature_digest(panorama) == metadata["panorama_sha256"]
        sample_count = metadata.get("temporal_sample_count")
        if sample_count is not None:
            for view_index in range(period):
                view_name = f"view-{view_index:02d}"
                samples = []
                for sample_index in range(int(sample_count)):
                    sample_name = f"sample-{sample_index:02d}.png"
                    path = directory / "samples" / view_name / sample_name
                    valid = valid and (
                        _digest(path)
                        == metadata["temporal_sample_sha256"][view_name][sample_name]
                    )
                    with Image.open(path) as image:
                        samples.append(np.asarray(image.convert("RGB")))
                valid = valid and bool(
                    np.array_equal(temporal_median_rgb(samples), frames[view_index])
                )
    return panoramas, views, valid


def _source_paths(root: Path) -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "natural_visual_expedition_benchmark.py": Path(__file__).with_name(
            "natural_visual_expedition_benchmark.py"
        ),
        "audit_natural_visual_expedition.py": Path(__file__).resolve(),
        "natural_visual_loop_benchmark.py": Path(__file__).with_name(
            "natural_visual_loop_benchmark.py"
        ),
        "navigation_probe.py": Path(__file__).with_name("navigation_probe.py"),
        "frontier_program.py": repository
        / "src/minecraft_scientist/exploration/frontier_program.py",
        "memory.py": repository / "src/minecraft_scientist/exploration/memory.py",
        "perception.py": repository / "src/minecraft_scientist/exploration/perception.py",
        "minestudio.py": repository / "src/minecraft_scientist/minecraft/minestudio.py",
    }


def audit(root: Path) -> dict[str, Any]:
    report = _load(root / "report.json")
    config = _load(root / "config.json")
    evaluator = _load(root / "evaluator-audit.json")
    panoramas, views, scans_valid = _load_scans(root, report)

    execution_alignments_valid = True
    roles_valid = True
    execution_items = list(report["executions"])
    for result in report["frontier_probe"]["results"]:
        execution_items.extend(result["executions"])
    for execution in execution_items:
        roles = tuple(str(item) for item in execution["action_roles"])
        try:
            program = VisualFrontierProgram.from_roles(roles)
        except ValueError:
            roles_valid = False
            continue
        roles_valid = roles_valid and program.turn_steps < int(
            report["inferred_camera_period"]
        )
        roles_valid = roles_valid and not any(
            token in role
            for role in roles
            for token in ("forward", "jump", "back", "left", "right", "obj_")
        )
        alignment = align_panorama_heading(
            views[execution["reference_scan_role"]],
            views[execution["source_scan_role"]],
        )
        execution_alignments_valid = execution_alignments_valid and (
            alignment.model_dump(mode="json") == execution["alignment"]
        )

    frontier_probe_valid = True
    recomputed_probe_results = []
    for index, result in enumerate(report["frontier_probe"]["results"]):
        prefix = f"frontier-probe-{index:02d}"
        outbound_distance = math.dist(
            panoramas[f"{prefix}-c"], panoramas[f"{prefix}-outbound"]
        )
        return_distance = math.dist(
            panoramas[f"{prefix}-c"], panoramas[f"{prefix}-return"]
        )
        score = outbound_distance / (return_distance + 1.0)
        frontier_probe_valid = frontier_probe_valid and all(
            math.isclose(left, float(right), abs_tol=1e-12)
            for left, right in (
                (outbound_distance, result["outbound_distance"]),
                (return_distance, result["return_distance"]),
                (score, result["reversibility_score"]),
            )
        )
        recomputed_probe_results.append(
            (score, outbound_distance, int(result["heading_steps"]))
        )
    selected_probe = max(
        recomputed_probe_results,
        key=lambda item: (item[0], item[1], -item[2]),
    )
    frontier_probe_valid = frontier_probe_valid and (
        selected_probe[2]
        == int(report["frontier_probe"]["selected_heading_steps"])
        == int(report["frontier_heading_steps"])
    )

    place_ids = report["place_ids"]
    expected_abc = [place_ids[key] for key in "abc"]
    expected_abcd = [place_ids[key] for key in "abcd"]
    memory_path = root / "exploration.sqlite3"
    with ExplorationMemory.open(memory_path) as memory:
        memory_integrity = memory.verify_integrity()
        memory_summary_valid = memory.summary() == report["memory"]
        frontier = memory.get_frontier(report["remote_frontier_id"])
        remote_frontier_valid = (
            frontier.status.value == "resolved"
            and frontier.discovered_place_id == place_ids["d"]
        )
        route = memory.plan_route(place_ids["a"], place_ids["d"])
        persisted_route_valid = route is not None and (
            route.model_dump(mode="json") == report["revisit_plan"]
        )
        stored_role_sets = [item.action_roles for item in memory.frontiers()]
        stored_role_sets.extend(item.action_roles for item in memory.transitions())
        try:
            for role_set in stored_role_sets:
                VisualFrontierProgram.from_roles(role_set)
        except ValueError:
            roles_valid = False
        heading_states = memory.modality_places(
            str(report["encoder"]) + HEADING_MODALITY_SUFFIX
        )
        dimensions = int(report["descriptor_dimensions"]) // 2
        heading_memory_valid = (
            len(heading_states) == 4
            and {item.place_id for item in heading_states} == set(expected_abcd)
            and all(
                len(item.exemplars) == 1
                and len(item.exemplars[0])
                == int(report["inferred_camera_period"]) * dimensions
                for item in heading_states
            )
        )
        panorama_recognition = {
            role: memory.match_modality_place(str(report["encoder"]), signature)
            for role, signature in panoramas.items()
            if role.startswith(("calibration-", "heldout-", "revisit-"))
        }

    evaluator_metrics_valid = True
    for phase in ("training", "calibration", "heldout", "revisit"):
        session = evaluator["sessions"][evaluator[f"{phase}_session_index"]]
        recomputed = _metrics(session["positions"])
        evaluator_metrics_valid = evaluator_metrics_valid and _close_metrics(
            recomputed, evaluator[f"{phase}_metrics"]
        )
        evaluator_metrics_valid = evaluator_metrics_valid and _close_metrics(
            recomputed, report[f"{phase}_evaluator_metrics"]
        )
    evaluator_is_separate = all(
        any(key in item["raw_metadata_keys"] for key in ("xpos", "location_stats"))
        for item in evaluator["sessions"]
    )

    public_text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("events.jsonl", "memory-events.jsonl")
    )
    public_telemetry_clean = not any(
        token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS
    )
    source_snapshot_valid = all(
        _digest(root / "source" / name) == digest
        for name, digest in report["source_sha256"].items()
    )
    current_sources = _source_paths(root)
    current_source_matches_frozen = set(current_sources) == set(
        report["source_sha256"]
    ) and all(
        _digest(path) == report["source_sha256"][name]
        for name, path in current_sources.items()
    )
    conditions_distinct = len(
        {
            (config["training_time"], config["training_weather"]),
            (config["calibration_time"], config["calibration_weather"]),
            (config["heldout_time"], config["heldout_weather"]),
        }
    ) == 3
    report_relations_valid = (
        report["suite"] == "natural_visual_expedition"
        and report["backend"] == "real-minestudio"
        and report["terrain_modified"] is False
        and report["calibration_recognition"]
        == {key: place_ids[key] for key in "abc"}
        and report["heldout_matches"] == expected_abc
        and report["revisit_matches"] == expected_abcd
        and report["d_match_before_learning"] is None
        and report["frontier_plan"]["place_path"] == expected_abc
        and report["frontier_plan"]["frontier_id"]
        == report["remote_frontier_id"]
    )
    image_recognition_consistent = all(
        panorama_recognition.get(role) == place_ids[place]
        for role, place in {
            "calibration-a": "a",
            "calibration-b": "b",
            "calibration-c": "c",
            "heldout-a": "a",
            "heldout-route-1": "b",
            "heldout-route-2": "c",
            "heldout-d1": "d",
            "revisit-a": "a",
            "revisit-route-1": "b",
            "revisit-route-2": "c",
            "revisit-route-3": "d",
        }.items()
    )
    checks = {
        "scans_recomputed_from_png": scans_valid,
        "execution_alignments_recomputed": execution_alignments_valid,
        "frontier_probe_recomputed": frontier_probe_valid,
        "opaque_action_roles_valid": roles_valid,
        "memory_integrity": memory_integrity,
        "memory_summary_valid": memory_summary_valid,
        "remote_frontier_resolved": remote_frontier_valid,
        "persisted_route_recomputed": persisted_route_valid,
        "ordered_heading_memory_valid": heading_memory_valid,
        "image_recognition_consistent": image_recognition_consistent,
        "evaluator_metrics_recomputed": evaluator_metrics_valid,
        "evaluator_is_separate": evaluator_is_separate,
        "public_telemetry_clean": public_telemetry_clean,
        "source_snapshot_valid": source_snapshot_valid,
        "current_source_matches_frozen": current_source_matches_frozen,
        "conditions_distinct": conditions_distinct,
        "report_relations_valid": report_relations_valid,
        "reported_gate_passed": report["gate"]["passed"] is True
        and all(value is True for key, value in report["gate"].items() if key != "passed"),
    }
    return {
        "suite": "natural_visual_expedition_audit",
        "root": str(root.resolve()),
        "checks": checks,
        "valid": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.root)
    if args.write:
        (args.root / "integrity-audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

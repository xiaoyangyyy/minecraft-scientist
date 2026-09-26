"""Independent audit for a natural_visual_loop_benchmark evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from natural_visual_loop_benchmark import (
    NATURAL_WORLD_COMMANDS,
    PRIVILEGED_ARTIFACT_TOKENS,
    natural_world_commands,
    view_openness,
)
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    RGBEdgePanoramaEncoder,
    RGBPanoramaEncoder,
    RGBSceneEncoder,
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature_digest(signature: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _encoder(version: str) -> RGBPanoramaEncoder:
    options = {
        RGBPanoramaEncoder.VERSION: RGBPanoramaEncoder,
        RGBEdgePanoramaEncoder.VERSION: RGBEdgePanoramaEncoder,
    }
    try:
        return options[version]()
    except KeyError as error:
        raise ValueError(f"unsupported panorama encoder: {version}") from error


def _load_scan(
    root: Path,
    role: str,
    count: int,
    encoder: RGBPanoramaEncoder,
) -> tuple[int, ...]:
    role_path = root / "scans" / role.replace("_", "-")
    frames = []
    for index in range(count):
        path = role_path / f"view-{index:02d}.png"
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    return encoder.encode(frames)


def _position_metrics(positions: list[dict[str, float]]) -> dict[str, float | None]:
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


def _orientation_error(positions: list[dict[str, float]]) -> float | None:
    compatible = [item for item in positions if "yaw" in item]
    if not compatible:
        return None
    delta = compatible[-1]["yaw"] - compatible[0]["yaw"]
    return abs((delta + 180) % 360 - 180)


def audit(root: Path) -> dict[str, object]:
    report = _load_json(root / "report.json")
    evaluator = _load_json(root / "evaluator-audit.json")
    period = int(report["inferred_camera_period"])
    encoder = _encoder(report["encoder"])
    scan_signatures = {
        role: _load_scan(root, role, int(metadata["view_count"]), encoder)
        for role, metadata in report["scans"].items()
    }
    scan_hashes_valid = all(
        _signature_digest(scan_signatures[role]) == metadata["panorama_sha256"]
        and all(
            _digest(root / "scans" / role.replace("_", "-") / name) == digest
            for name, digest in metadata["image_sha256"].items()
        )
        for role, metadata in report["scans"].items()
    )
    scan_counts_valid = all(
        int(metadata["view_count"]) == period for metadata in report["scans"].values()
    )
    temporal_aggregation_valid = True
    expected_temporal_samples = int(report.get("temporal_samples", 1))
    for role, metadata in report["scans"].items():
        if role == "discovery" or expected_temporal_samples == 1:
            continue
        temporal_aggregation_valid = temporal_aggregation_valid and (
            int(metadata.get("temporal_sample_count", 0))
            == expected_temporal_samples
        )
        for view_index in range(period):
            view_name = f"view-{view_index:02d}"
            samples = []
            for sample_index in range(expected_temporal_samples):
                sample_name = f"sample-{sample_index:02d}.png"
                path = (
                    root
                    / "scans"
                    / role.replace("_", "-")
                    / "samples"
                    / view_name
                    / sample_name
                )
                temporal_aggregation_valid = temporal_aggregation_valid and (
                    _digest(path)
                    == metadata["temporal_sample_sha256"][view_name][sample_name]
                )
                with Image.open(path) as image:
                    samples.append(np.asarray(image.convert("RGB")))
            recomputed = np.median(np.stack(samples), axis=0).astype(np.uint8)
            with Image.open(
                root
                / "scans"
                / role.replace("_", "-")
                / f"{view_name}.png"
            ) as image:
                saved = np.asarray(image.convert("RGB"))
            temporal_aggregation_valid = temporal_aggregation_valid and bool(
                np.array_equal(recomputed, saved)
            )
    discovery_metadata = report["scans"]["discovery"]
    if "openness_scores" in discovery_metadata:
        discovery_frames = []
        for index in range(period):
            with Image.open(
                root / "scans/discovery" / f"view-{index:02d}.png"
            ) as image:
                discovery_frames.append(np.asarray(image.convert("RGB")))
        openness_scores = tuple(view_openness(frame) for frame in discovery_frames)
        expected_open_heading = max(
            range(period), key=lambda index: (openness_scores[index], -index)
        )
        selected_heading = int(report["selected_heading_steps"])
        openness_valid = (
            len(openness_scores) == len(discovery_metadata["openness_scores"])
            and all(
                math.isclose(left, float(right), abs_tol=1e-12)
                for left, right in zip(
                    openness_scores,
                    discovery_metadata["openness_scores"],
                    strict=True,
                )
            )
            and int(discovery_metadata["selected_heading_steps"])
            == expected_open_heading
        )
        probe = report.get("heading_probe")
        if probe is not None:
            expected_candidates = tuple(
                sorted(
                    range(period),
                    key=lambda index: (-openness_scores[index], index),
                )[: int(probe["candidate_count"])]
            )
            probe_results_valid = True
            recomputed_results = []
            for result in probe["results"]:
                heading = int(result["heading_steps"])
                root_path = root / "heading-probes" / f"heading-{heading:02d}"
                signatures = {}
                hashes_valid = True
                for role in ("start", "outbound", "return"):
                    path = root_path / f"{role}.png"
                    hashes_valid = hashes_valid and (
                        _digest(path) == result["image_sha256"][f"{role}.png"]
                    )
                    with Image.open(path) as image:
                        signatures[role] = encoder.scene_encoder.encode(
                            np.asarray(image.convert("RGB"))
                        )
                    hashes_valid = hashes_valid and (
                        _signature_digest(signatures[role])
                        == result["signature_sha256"][role]
                    )
                outbound_distance = math.dist(
                    signatures["start"], signatures["outbound"]
                )
                return_distance = math.dist(
                    signatures["start"], signatures["return"]
                )
                score = outbound_distance / (return_distance + 1.0)
                probe_results_valid = probe_results_valid and hashes_valid and all(
                    math.isclose(left, float(right), abs_tol=1e-12)
                    for left, right in (
                        (outbound_distance, result["outbound_distance"]),
                        (return_distance, result["return_distance"]),
                        (score, result["reversibility_score"]),
                    )
                )
                recomputed_results.append((score, outbound_distance, heading))
            expected_selected = max(
                recomputed_results,
                key=lambda item: (item[0], item[1], -item[2]),
            )[2]
            heading_selection_valid = (
                openness_valid
                and report.get("reversible_heading_selection_enabled") is True
                and report["open_heading_selection_enabled"] is False
                and tuple(probe["candidate_headings"]) == expected_candidates
                and tuple(item[2] for item in recomputed_results)
                == expected_candidates
                and probe_results_valid
                and int(probe["selected_heading_steps"]) == expected_selected
                and selected_heading == expected_selected
            )
        else:
            heading_selection_valid = openness_valid and selected_heading == (
                expected_open_heading
                if report["open_heading_selection_enabled"]
                else 0
            )
    else:
        heading_selection_valid = (
            "selected_heading_steps" not in report
            and "open_heading_selection_enabled" not in report
        )
    homing_valid = True
    for phase, metadata in report.get("visual_homing", {}).items():
        prefix = "train_homing" if phase == "training" else "heldout_homing"
        reference_role = "discovery" if phase == "training" else "heldout-start"
        reference_index = (
            (int(report["selected_heading_steps"]) + period // 2) % period
            if phase == "training"
            else period // 2
        )
        with Image.open(
            root / "scans" / reference_role / f"view-{reference_index:02d}.png"
        ) as image:
            reference = RGBSceneEncoder().encode(np.asarray(image.convert("RGB")))
        reference_hash = hashlib.sha256(
            json.dumps(reference, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        recomputed_items = []
        image_hashes_valid = True
        for step in range(1, int(metadata["candidate_steps_taken"]) + 1):
            name = f"frame-{step:02d}.png"
            path = root / "homing" / prefix.replace("_", "-") / name
            image_hashes_valid = image_hashes_valid and (
                _digest(path) == metadata["candidate_image_sha256"][name]
            )
            with Image.open(path) as image:
                signature = RGBSceneEncoder().encode(np.asarray(image.convert("RGB")))
            recomputed_items.append(math.dist(reference, signature))
        recomputed = tuple(recomputed_items)
        best_step = min(range(len(recomputed)), key=lambda index: recomputed[index]) + 1
        homing_valid = homing_valid and (
            len(recomputed) == len(metadata["candidate_distances"])
            and image_hashes_valid
            and reference_hash == metadata["reference_view_sha256"]
            and all(
                math.isclose(left, float(right), abs_tol=1e-12)
                for left, right in zip(
                    recomputed, metadata["candidate_distances"], strict=True
                )
            )
            and best_step == int(metadata["best_candidate_step"])
            and metadata["final_role"] in scan_signatures
        )

    with ExplorationMemory.open(root / "exploration.sqlite3") as memory:
        states = memory.modality_places(encoder.VERSION)
        memory_integrity = memory.verify_integrity()
        memory_events = tuple(memory.iter_events())
        if len(states) != 1:
            return {"valid": False, "reason": "unexpected panorama place count"}
        state = states[0]
        decisions = {
            role: memory.match_modality_place(encoder.VERSION, signature)
            for role, signature in scan_signatures.items()
        }
    adaptation = report.get("continual_adaptation")
    if adaptation is None:
        continual_adaptation_valid = True
    else:
        consolidation_events = tuple(
            event
            for event in memory_events
            if event["event_type"] == "modality_view_consolidated"
        )
        calibration_events = tuple(
            event
            for event in memory_events
            if event["event_type"] == "modality_view_calibrated"
        )
        continual_adaptation_valid = (
            adaptation["policy"] == "accepted-view-bounded-coreset-v1"
            and len(calibration_events) == 1
            and len(consolidation_events)
            == int(adaptation["heldout_start_consolidated"])
            + int(adaptation["heldout_return_consolidated"])
        )
        previous_state = calibration_events[0]["payload"]["state"]
        for role, expected_episode, expected_step, flag_name in (
            ("heldout_start", 4, 0, "heldout_start_consolidated"),
            ("heldout_return", 4, 1, "heldout_return_consolidated"),
        ):
            if not adaptation[flag_name]:
                continue
            matching = tuple(
                event
                for event in consolidation_events
                if event["episode_id"] == expected_episode
                and event["step"] == expected_step
            )
            continual_adaptation_valid = continual_adaptation_valid and len(matching) == 1
            if not matching:
                continue
            event = matching[0]
            signature = tuple(int(value) for value in event["payload"]["signature"])
            exemplars = tuple(
                tuple(int(value) for value in exemplar)
                for exemplar in previous_state["exemplars"]
            )
            distance = min(math.dist(signature, exemplar) for exemplar in exemplars)
            continual_adaptation_valid = continual_adaptation_valid and (
                signature == scan_signatures[role]
                and distance <= float(previous_state["match_radius"])
                and math.isclose(
                    distance,
                    float(event["payload"]["distance_before"]),
                    abs_tol=1e-12,
                )
            )
            previous_state = event["payload"]["state"]
    calibration_valid = (
        state.calibration is not None
        and state.calibration.model_dump(mode="json") == report["calibration"]
    )
    calibration_strategy = report.get("calibration_strategy")
    if calibration_strategy is None:
        calibration_strategy_valid = True
    elif calibration_strategy == "maximum-gap-training-only":
        calibration = report["calibration"]
        expected_margin = (
            float(calibration["nearest_negative_distance"])
            - float(calibration["maximum_positive_distance"])
        ) * 0.499999
        calibration_strategy_valid = math.isclose(
            float(calibration["margin"]), expected_margin, abs_tol=1e-12
        )
    else:
        calibration_strategy_valid = calibration_strategy == "fixed-margin"
    decisions_valid = (
        decisions["heldout_start"] == state.place_id
        and decisions["heldout_return"] == state.place_id
        and decisions["heldout_restart"] == state.place_id
        and decisions["heldout_outbound"] is None
        and decisions["train_negative"] is None
    )

    public_paths = (root / "config.json", root / "events.jsonl", root / "report.json")
    public_payload = "\n".join(
        path.read_text(encoding="utf-8") for path in public_paths
    )
    public_telemetry_clean = not any(
        token in public_payload for token in PRIVILEGED_ARTIFACT_TOKENS
    )
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actions_compliant = bool(events) and all(
        event["intervention_compliance"] and event["event"]["success"]
        for event in events
    )
    condition_split = report.get("condition_split")
    if condition_split is None:
        command_groups = {
            "training": tuple(report["controlled_conditions"]),
            "heldout": tuple(report["controlled_conditions"]),
        }
        expected_groups = {
            "training": NATURAL_WORLD_COMMANDS,
            "heldout": NATURAL_WORLD_COMMANDS,
        }
        condition_split_valid = True
    else:
        command_groups = {
            role: tuple(report["controlled_conditions"][role])
            for role in ("training", "heldout")
        }
        expected_groups = {
            role: natural_world_commands(
                time_of_day=condition_split[role]["time"],
                weather=condition_split[role]["weather"],
            )
            for role in ("training", "heldout")
        }
        shifted = expected_groups["training"] != expected_groups["heldout"]
        condition_split_valid = condition_split["shifted"] is shifted and (
            shifted or not condition_split["required"]
        )
    commands_valid = (
        report["terrain_modified"] is False
        and command_groups == expected_groups
        and condition_split_valid
        and not any(
            token in command.lower()
            for commands in command_groups.values()
            for command in commands
            for token in (" fill ", "setblock", " clone ", " tp ", "teleport")
        )
    )

    loop_index = int(evaluator["loop_session_index"])
    loop_positions = evaluator["sessions"][loop_index]["positions"]
    recomputed_loop = _position_metrics(loop_positions)
    loop_metrics_valid = all(
        report["loop_evaluator_metrics"][key] is not None
        and recomputed_loop[key] is not None
        and math.isclose(
            float(report["loop_evaluator_metrics"][key]),
            float(recomputed_loop[key]),
            abs_tol=1e-12,
        )
        for key in ("maximum_excursion", "return_error")
    )
    physical_loop_valid = (
        recomputed_loop["maximum_excursion"] is not None
        and recomputed_loop["maximum_excursion"] >= 2
        and recomputed_loop["return_error"] is not None
        and recomputed_loop["return_error"] <= 1.5
    )
    orientation_error = _orientation_error(evaluator["sessions"][0]["positions"])
    orientation_valid = (
        period % 2 == 0
        and orientation_error is not None
        and orientation_error <= 1
        and report["evaluator_orientation_return_error"] is not None
        and math.isclose(
            orientation_error,
            float(report["evaluator_orientation_return_error"]),
            abs_tol=1e-12,
        )
    )
    source_hashes_valid = all(
        _digest(root / "source" / name) == digest
        for name, digest in report["source_sha256"].items()
    )
    checks = {
        "scan_hashes_valid": scan_hashes_valid,
        "scan_counts_valid": scan_counts_valid,
        "temporal_aggregation_valid": temporal_aggregation_valid,
        "heading_selection_valid": heading_selection_valid,
        "visual_homing_valid": homing_valid,
        "calibration_valid": calibration_valid,
        "calibration_strategy_valid": calibration_strategy_valid,
        "continual_adaptation_valid": continual_adaptation_valid,
        "decisions_valid": decisions_valid,
        "memory_integrity_valid": memory_integrity,
        "public_telemetry_clean": public_telemetry_clean,
        "actions_compliant": actions_compliant,
        "natural_terrain_commands_valid": commands_valid,
        "condition_split_valid": condition_split_valid,
        "loop_metrics_valid": loop_metrics_valid,
        "physical_loop_valid": physical_loop_valid,
        "orientation_valid": orientation_valid,
        "source_hashes_valid": source_hashes_valid,
    }
    return {
        **checks,
        "maximum_excursion": recomputed_loop["maximum_excursion"],
        "return_error": recomputed_loop["return_error"],
        "valid": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the verified result to integrity-audit.json",
    )
    args = parser.parse_args()
    result = audit(args.run)
    result["auditor_sha256"] = _digest(Path(__file__).resolve())
    if args.write:
        (args.run / "integrity-audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

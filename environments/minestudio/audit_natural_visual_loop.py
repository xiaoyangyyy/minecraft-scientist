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
)
from PIL import Image

from minecraft_scientist.exploration import ExplorationMemory, RGBPanoramaEncoder


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature_digest(signature: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_scan(root: Path, role: str, count: int) -> tuple[int, ...]:
    role_path = root / "scans" / role.replace("_", "-")
    frames = []
    for index in range(count):
        path = role_path / f"view-{index:02d}.png"
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    return RGBPanoramaEncoder().encode(frames)


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
    scan_signatures = {
        role: _load_scan(root, role, int(metadata["view_count"]))
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

    encoder = RGBPanoramaEncoder()
    with ExplorationMemory.open(root / "exploration.sqlite3") as memory:
        states = memory.modality_places(encoder.VERSION)
        memory_integrity = memory.verify_integrity()
        if len(states) != 1:
            return {"valid": False, "reason": "unexpected panorama place count"}
        state = states[0]
        decisions = {
            role: memory.match_modality_place(encoder.VERSION, signature)
            for role, signature in scan_signatures.items()
        }
    calibration_valid = (
        state.calibration is not None
        and state.calibration.model_dump(mode="json") == report["calibration"]
    )
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
    commands_valid = (
        report["terrain_modified"] is False
        and tuple(report["controlled_conditions"]) == NATURAL_WORLD_COMMANDS
        and not any(
            token in command.lower()
            for command in report["controlled_conditions"]
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
        "calibration_valid": calibration_valid,
        "decisions_valid": decisions_valid,
        "memory_integrity_valid": memory_integrity,
        "public_telemetry_clean": public_telemetry_clean,
        "actions_compliant": actions_compliant,
        "natural_terrain_commands_valid": commands_valid,
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

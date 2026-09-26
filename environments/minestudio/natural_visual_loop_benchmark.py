"""Pixel-only place loop on unmodified natural MineStudio terrain.

The Agent discovers the camera-turn period from RGB recurrence, builds an
orientation-independent panorama, learns a persistent place boundary from
separate simulator sessions, then walks out and back without coordinates.
Evaluator-only position telemetry proves physical excursion and return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks.commands import CommandsCallback
from navigation_probe import PositionAudit, PositionAuditSimulator
from PIL import Image

from minecraft_scientist.exploration import (
    ExplorationMemory,
    RGBPanoramaEncoder,
    infer_visual_rotation_period,
    rgb_signature_from_observation,
)
from minecraft_scientist.minecraft import GridVisualController, MineStudioVisualAdapter
from minecraft_scientist.protocols import ActionKind, PrimitiveAction, TransitionRecord

NATURAL_WORLD_COMMANDS = (
    "/gamerule sendCommandFeedback false",
    "/gamerule announceAdvancements false",
    "/gamerule doTileDrops false",
    "/gamemode creative",
    "/difficulty peaceful",
    "/weather clear",
    "/time set noon",
    "/gamerule doDaylightCycle false",
    "/gamerule doWeatherCycle false",
    "/clear @p",
)
PRIVILEGED_KEYS = {"location_stats", "xpos", "ypos", "zpos", "yaw", "pitch"}
PRIVILEGED_ARTIFACT_TOKENS = (
    "location_stats",
    "xpos",
    "ypos",
    "zpos",
    '"yaw"',
    '"pitch"',
)


class NaturalRecorder:
    """RGB-only controller; position telemetry is rejected at the boundary."""

    def __init__(self) -> None:
        self.grid = GridVisualController(
            rows=3,
            columns=3,
            hotbar_slots=1,
            movement_controls=("forward",),
        )
        self.frames: list[np.ndarray] = []

    def perceive(self, frame: np.ndarray, info: dict[str, Any], *, seed: int):
        if any(key in info for key in PRIVILEGED_KEYS):
            raise RuntimeError("privileged position telemetry reached RGB perception")
        array = np.asarray(frame).copy()
        self.frames.append(array)
        return self.grid.perceive(array, {}, seed=seed)

    def encode_action(self, action, scene):
        return self.grid.encode_action(action, scene)


@dataclass
class Session:
    world: MineStudioVisualAdapter
    recorder: NaturalRecorder
    audit: PositionAudit
    records: list[TransitionRecord]


def _factory(config, audit: PositionAudit):
    simulator = MinecraftSim(
        action_type="env",
        seed=config.seed,
        callbacks=[CommandsCallback(list(NATURAL_WORLD_COMMANDS))],
    )
    return PositionAuditSimulator(simulator, audit)


def _open_session(seed: int) -> Session:
    recorder = NaturalRecorder()
    audit = PositionAudit()
    world = MineStudioVisualAdapter(
        recorder,
        simulator_factory=lambda config: _factory(config, audit),
    )
    world.reset(seed)
    records: list[TransitionRecord] = []
    records.append(world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=20)))
    return Session(world=world, recorder=recorder, audit=audit, records=records)


def _target(
    world: MineStudioVisualAdapter, kind: ActionKind, column: int | None = None
) -> str:
    options = []
    for item in world.observe().objects:
        if kind not in item.affordances:
            continue
        if (
            column is not None
            and item.observable_features.get("screen_column") != column
        ):
            continue
        if column is not None and item.observable_features.get("screen_row") != 1:
            continue
        options.append(item.object_id)
    if len(options) != 1:
        raise RuntimeError(
            f"expected one opaque {kind.value} handle, got {len(options)}"
        )
    return options[0]


def _turn(session: Session, target_id: str) -> None:
    session.records.append(
        session.world.step(PrimitiveAction(kind=ActionKind.LOOK, target_id=target_id))
    )


def _signature(session: Session) -> tuple[int, ...]:
    return rgb_signature_from_observation(session.world.observe())


def _save_scan(
    output: Path,
    frames: tuple[np.ndarray, ...],
    panorama: tuple[int, ...],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for index, frame in enumerate(frames):
        path = output / f"view-{index:02d}.png"
        Image.fromarray(frame).save(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "view_count": len(frames),
        "panorama_sha256": hashlib.sha256(
            json.dumps(panorama, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "image_sha256": hashes,
    }


def _discover_scan(
    session: Session,
    output: Path,
    *,
    minimum_steps: int,
    maximum_steps: int,
) -> tuple[int, tuple[int, ...], dict[str, Any]]:
    right = _target(session.world, ActionKind.LOOK, 2)
    left = _target(session.world, ActionKind.LOOK, 0)
    anchor = _signature(session)
    frames = [session.recorder.frames[-1].copy()]
    signatures = []
    for _ in range(maximum_steps):
        _turn(session, right)
        signatures.append(_signature(session))
        frames.append(session.recorder.frames[-1].copy())
    period, distances = infer_visual_rotation_period(
        anchor,
        signatures,
        minimum_steps=minimum_steps,
    )
    if period % 2:
        raise RuntimeError("RGB-inferred camera period is not even")
    for _ in range(maximum_steps - period):
        _turn(session, left)
    session.records.append(
        session.world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2))
    )
    restored_distance = math.dist(anchor, _signature(session))
    scan_signatures = (anchor, *signatures[: period - 1])
    scan_frames = tuple(frames[:period])
    encoder = RGBPanoramaEncoder()
    panorama = encoder.encode_signatures(scan_signatures)
    metadata = _save_scan(output, scan_frames, panorama)
    metadata.update(
        {
            "inferred_period": period,
            "probe_steps": maximum_steps,
            "anchor_distances": distances,
            "restored_distance": restored_distance,
        }
    )
    return period, panorama, metadata


def _capture_scan(
    session: Session,
    output: Path,
    *,
    period: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    right = _target(session.world, ActionKind.LOOK, 2)
    signatures = [_signature(session)]
    frames = [session.recorder.frames[-1].copy()]
    for step in range(1, period + 1):
        _turn(session, right)
        if step < period:
            signatures.append(_signature(session))
            frames.append(session.recorder.frames[-1].copy())
    session.records.append(
        session.world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2))
    )
    encoder = RGBPanoramaEncoder()
    panorama = encoder.encode_signatures(signatures)
    return panorama, _save_scan(output, tuple(frames), panorama)


def _move(session: Session, *, moves: int, move_frames: int) -> None:
    handle = _target(session.world, ActionKind.MOVE)
    for _ in range(moves):
        session.records.append(
            session.world.step(
                PrimitiveAction(
                    kind=ActionKind.MOVE,
                    target_id=handle,
                    ticks=move_frames,
                )
            )
        )
        session.records.append(
            session.world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=1))
        )


def _half_turn(session: Session, period: int) -> None:
    right = _target(session.world, ActionKind.LOOK, 2)
    for _ in range(period // 2):
        _turn(session, right)
    session.records.append(
        session.world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2))
    )


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


def _orientation_return_error(positions: list[dict[str, float]]) -> float | None:
    compatible = [item for item in positions if "yaw" in item]
    if not compatible:
        return None
    delta = compatible[-1]["yaw"] - compatible[0]["yaw"]
    return abs((delta + 180) % 360 - 180)


def _write_events(path: Path, sessions: list[Session]) -> None:
    records = [record for session in sessions for record in session.records]
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def _copy_sources(output: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    sources = {
        "natural_visual_loop_benchmark.py": Path(__file__).resolve(),
        "perception.py": root / "src/minecraft_scientist/exploration/perception.py",
    }
    target = output / "source"
    target.mkdir()
    hashes = {}
    for name, source in sources.items():
        destination = target / name
        shutil.copy2(source, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--moves", type=int, default=4)
    parser.add_argument("--move-frames", type=int, default=4)
    parser.add_argument("--minimum-turn-steps", type=int, default=8)
    parser.add_argument("--maximum-turn-steps", type=int, default=16)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.moves < 1 or args.move_frames < 1:
        parser.error("movement budgets must be positive")
    if args.minimum_turn_steps < 4 or args.maximum_turn_steps < args.minimum_turn_steps:
        parser.error("invalid camera-period search interval")
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("margin must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)

    sessions: list[Session] = []
    scans: dict[str, dict[str, Any]] = {}
    try:
        discovery = _open_session(args.seed)
        sessions.append(discovery)
        period, anchor, scans["discovery"] = _discover_scan(
            discovery,
            args.output / "scans/discovery",
            minimum_steps=args.minimum_turn_steps,
            maximum_steps=args.maximum_turn_steps,
        )
        discovery.world.close()

        training = _open_session(args.seed)
        sessions.append(training)
        positive_one, scans["train_positive_one"] = _capture_scan(
            training,
            args.output / "scans/train-positive-one",
            period=period,
        )
        _move(training, moves=args.moves, move_frames=args.move_frames)
        training_negative, scans["train_negative"] = _capture_scan(
            training,
            args.output / "scans/train-negative",
            period=period,
        )
        training.world.close()

        second_positive_session = _open_session(args.seed)
        sessions.append(second_positive_session)
        positive_two, scans["train_positive_two"] = _capture_scan(
            second_positive_session,
            args.output / "scans/train-positive-two",
            period=period,
        )
        second_positive_session.world.close()

        training_loop = _open_session(args.seed)
        sessions.append(training_loop)
        _move(training_loop, moves=args.moves, move_frames=args.move_frames)
        _half_turn(training_loop, period)
        _move(training_loop, moves=args.moves, move_frames=args.move_frames)
        _half_turn(training_loop, period)
        training_return, scans["train_loop_return"] = _capture_scan(
            training_loop,
            args.output / "scans/train-loop-return",
            period=period,
        )
        training_loop.world.close()

        memory_path = args.output / "exploration.sqlite3"
        encoder = RGBPanoramaEncoder()
        with ExplorationMemory(memory_path, max_exemplars=4) as memory:
            place = memory.observe_place((0,), episode_id=0, step=0)
            memory.record_modality_view(
                encoder.VERSION,
                place.place_id,
                anchor,
                episode_id=0,
                step=0,
            )
            calibrated = memory.calibrate_modality_place(
                encoder.VERSION,
                place.place_id,
                (positive_one, positive_two, training_return),
                (training_negative,),
                episode_id=3,
                step=1,
                margin=args.margin,
            )

        loop = _open_session(args.seed)
        sessions.append(loop)
        loop_start, scans["heldout_start"] = _capture_scan(
            loop,
            args.output / "scans/heldout-start",
            period=period,
        )
        _move(loop, moves=args.moves, move_frames=args.move_frames)
        loop_outbound, scans["heldout_outbound"] = _capture_scan(
            loop,
            args.output / "scans/heldout-outbound",
            period=period,
        )
        _half_turn(loop, period)
        _move(loop, moves=args.moves, move_frames=args.move_frames)
        _half_turn(loop, period)
        loop_return, scans["heldout_return"] = _capture_scan(
            loop,
            args.output / "scans/heldout-return",
            period=period,
        )
        loop.world.close()

        with ExplorationMemory.open(memory_path) as memory:
            start_match = memory.match_modality_place(encoder.VERSION, loop_start)
            outbound_match = memory.match_modality_place(encoder.VERSION, loop_outbound)
            return_match = memory.match_modality_place(encoder.VERSION, loop_return)
            if return_match == place.place_id:
                memory.record_modality_view(
                    encoder.VERSION,
                    place.place_id,
                    loop_return,
                    episode_id=4,
                    step=1,
                )

        restart = _open_session(args.seed)
        sessions.append(restart)
        restart_signature, scans["heldout_restart"] = _capture_scan(
            restart,
            args.output / "scans/heldout-restart",
            period=period,
        )
        restart.world.close()

        with ExplorationMemory.open(memory_path) as memory:
            restart_match = memory.match_modality_place(
                encoder.VERSION, restart_signature
            )
            if restart_match == place.place_id:
                memory.record_modality_view(
                    encoder.VERSION,
                    place.place_id,
                    restart_signature,
                    episode_id=5,
                    step=0,
                )
            memory_summary = memory.summary()

        _write_events(args.output / "events.jsonl", sessions)
        public_text = (args.output / "events.jsonl").read_text(encoding="utf-8")
        public_telemetry_clean = not any(
            token in public_text for token in PRIVILEGED_ARTIFACT_TOKENS
        )
        action_records = [record for session in sessions for record in session.records]
        actions_compliant = bool(action_records) and all(
            record.intervention_compliance and record.event.success
            for record in action_records
        )
        loop_metrics = _position_metrics(loop.audit.positions)
        orientation_error = _orientation_return_error(discovery.audit.positions)
        source_hashes = _copy_sources(args.output)
        assert calibrated.calibration is not None
        gate = {
            "camera_period_even": period % 2 == 0,
            "camera_orientation_restored": (
                orientation_error is not None and orientation_error <= 1.0
            ),
            "actions_compliant": actions_compliant,
            "public_telemetry_clean": public_telemetry_clean,
            "heldout_start_recognized": start_match == place.place_id,
            "heldout_outbound_rejected": outbound_match is None,
            "heldout_return_recognized": return_match == place.place_id,
            "persistent_restart_recognized": restart_match == place.place_id,
            "physical_excursion": (
                loop_metrics["maximum_excursion"] is not None
                and loop_metrics["maximum_excursion"] >= 2.0
            ),
            "physical_return": (
                loop_metrics["return_error"] is not None
                and loop_metrics["return_error"] <= 1.5
            ),
            "memory_integrity": memory_summary["integrity_valid"] is True,
        }
        report = {
            "suite": "natural_visual_loop",
            "backend": "real-minestudio",
            "seed": args.seed,
            "terrain_modified": False,
            "controlled_conditions": list(NATURAL_WORLD_COMMANDS),
            "encoder": encoder.VERSION,
            "descriptor_dimensions": encoder.dimensions,
            "inferred_camera_period": period,
            "evaluator_orientation_return_error": orientation_error,
            "moves_each_leg": args.moves,
            "move_frames": args.move_frames,
            "calibration": calibrated.calibration.model_dump(mode="json"),
            "loop_evaluator_metrics": loop_metrics,
            "memory": memory_summary,
            "scans": scans,
            "source_sha256": source_hashes,
            "gate": {**gate, "passed": all(gate.values())},
            "scope": (
                "Natural generated terrain under fixed noon/clear/peaceful conditions; "
                "pixel-only panoramic place retrieval with evaluator-only coordinates. "
                "This is a bounded out-and-back loop, not open-world survival or SLAM."
            ),
        }
        evaluator = {
            "sessions": [
                {
                    "positions": session.audit.positions,
                    "raw_metadata_keys": sorted(session.audit.metadata_keys),
                }
                for session in sessions
            ],
            "loop_session_index": 4,
            "loop_metrics": loop_metrics,
        }
        (args.output / "config.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "moves": args.moves,
                    "move_frames": args.move_frames,
                    "minimum_turn_steps": args.minimum_turn_steps,
                    "maximum_turn_steps": args.maximum_turn_steps,
                    "margin": args.margin,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (args.output / "evaluator-audit.json").write_text(
            json.dumps(evaluator, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True), flush=True)
        if not report["gate"]["passed"]:
            raise SystemExit(1)
    finally:
        for session in sessions:
            session.world.close()


if __name__ == "__main__":
    main()

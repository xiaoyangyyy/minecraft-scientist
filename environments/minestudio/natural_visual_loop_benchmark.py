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
import time
from contextlib import suppress
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
    RGBEdgePanoramaEncoder,
    RGBPanoramaEncoder,
    infer_visual_rotation_period,
    rgb_signature_from_observation,
    temporal_median_rgb,
)
from minecraft_scientist.minecraft import GridVisualController, MineStudioVisualAdapter
from minecraft_scientist.protocols import ActionKind, PrimitiveAction, TransitionRecord

BASE_NATURAL_WORLD_COMMANDS = (
    "/gamerule sendCommandFeedback false",
    "/gamerule announceAdvancements false",
    "/gamerule doTileDrops false",
    "/gamemode creative",
    "/difficulty peaceful",
    "/gamerule doDaylightCycle false",
    "/gamerule doWeatherCycle false",
    "/clear @p",
)
TIME_CHOICES = ("day", "noon", "night", "midnight")
WEATHER_CHOICES = ("clear", "rain", "thunder")


def natural_world_commands(*, time_of_day: str, weather: str) -> tuple[str, ...]:
    if time_of_day not in TIME_CHOICES:
        raise ValueError(f"unsupported time of day: {time_of_day}")
    if weather not in WEATHER_CHOICES:
        raise ValueError(f"unsupported weather: {weather}")
    return (
        *BASE_NATURAL_WORLD_COMMANDS[:5],
        f"/weather {weather}",
        f"/time set {time_of_day}",
        *BASE_NATURAL_WORLD_COMMANDS[5:],
    )


NATURAL_WORLD_COMMANDS = natural_world_commands(time_of_day="noon", weather="clear")
PANORAMA_ENCODERS = {
    "lab": RGBPanoramaEncoder,
    "edge": RGBEdgePanoramaEncoder,
}
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

    def __init__(self, movement_programs: tuple[tuple[str, ...], ...]) -> None:
        self.grid = GridVisualController(
            rows=3,
            columns=3,
            hotbar_slots=1,
            movement_programs=movement_programs,
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
    startup_attempts: int = 1


def _factory(config, audit: PositionAudit, commands: tuple[str, ...]):
    simulator = MinecraftSim(
        action_type="env",
        seed=config.seed,
        callbacks=[CommandsCallback(list(commands))],
    )
    return PositionAuditSimulator(simulator, audit)


def _open_session(
    seed: int,
    commands: tuple[str, ...],
    movement_programs: tuple[tuple[str, ...], ...],
) -> Session:
    maximum_attempts = 3
    for attempt in range(1, maximum_attempts + 1):
        recorder = NaturalRecorder(movement_programs)
        audit = PositionAudit()
        world = MineStudioVisualAdapter(
            recorder,
            simulator_factory=lambda config, audit=audit, commands=commands: _factory(
                config, audit, commands
            ),
        )
        try:
            world.reset(seed)
            records = [
                world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=20))
            ]
        except Exception:
            with suppress(Exception):
                world.close()
            if attempt == maximum_attempts:
                raise
            time.sleep(2)
            continue
        return Session(
            world=world,
            recorder=recorder,
            audit=audit,
            records=records,
            startup_attempts=attempt,
        )
    raise AssertionError("unreachable session startup state")


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


def _movement_targets(world: MineStudioVisualAdapter) -> tuple[str, ...]:
    return tuple(
        sorted(
            item.object_id
            for item in world.observe().objects
            if item.affordances == (ActionKind.MOVE,)
        )
    )


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
    temporal_samples: tuple[tuple[np.ndarray, ...], ...] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for index, frame in enumerate(frames):
        path = output / f"view-{index:02d}.png"
        Image.fromarray(frame).save(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata: dict[str, Any] = {
        "view_count": len(frames),
        "panorama_sha256": hashlib.sha256(
            json.dumps(panorama, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "image_sha256": hashes,
    }
    if temporal_samples is not None:
        if len(temporal_samples) != len(frames):
            raise ValueError("temporal samples do not match panorama views")
        sample_hashes: dict[str, dict[str, str]] = {}
        for view_index, samples in enumerate(temporal_samples):
            view_name = f"view-{view_index:02d}"
            view_hashes = {}
            for sample_index, sample in enumerate(samples):
                name = f"sample-{sample_index:02d}.png"
                path = output / "samples" / view_name / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(sample).save(path)
                view_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            sample_hashes[view_name] = view_hashes
        metadata["temporal_sample_count"] = len(temporal_samples[0])
        metadata["temporal_sample_sha256"] = sample_hashes
    return metadata


def _temporal_frame(
    session: Session, sample_count: int
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if sample_count < 1 or sample_count % 2 == 0:
        raise ValueError("temporal RGB sample count must be a positive odd integer")
    samples = [session.recorder.frames[-1].copy()]
    for _ in range(sample_count - 1):
        session.records.append(
            session.world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=1))
        )
        samples.append(session.recorder.frames[-1].copy())
    median = temporal_median_rgb(samples)
    return median, tuple(samples)


def view_openness(frame: np.ndarray) -> float:
    """Label-free bright-horizon score used to choose a traversable heading."""

    array = np.asarray(frame, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("openness scoring requires an RGB frame")
    height, width = array.shape[:2]
    crop = array[: int(0.55 * height), int(0.25 * width) : int(0.75 * width), :3]
    luminance = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    return float(luminance.mean())


def _discover_scan(
    session: Session,
    output: Path,
    *,
    encoder: RGBPanoramaEncoder,
    minimum_steps: int,
    maximum_steps: int,
) -> tuple[
    int,
    tuple[int, ...],
    dict[str, Any],
    int,
    tuple[tuple[int, ...], ...],
]:
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
    openness_scores = tuple(view_openness(frame) for frame in scan_frames)
    selected_heading_steps = max(
        range(period), key=lambda index: (openness_scores[index], -index)
    )
    panorama = encoder.encode(scan_frames)
    metadata = _save_scan(output, scan_frames, panorama)
    metadata.update(
        {
            "inferred_period": period,
            "probe_steps": maximum_steps,
            "anchor_distances": distances,
            "restored_distance": restored_distance,
            "openness_scores": openness_scores,
            "selected_heading_steps": selected_heading_steps,
        }
    )
    return period, panorama, metadata, selected_heading_steps, scan_signatures


def _capture_scan(
    session: Session,
    output: Path,
    *,
    encoder: RGBPanoramaEncoder,
    period: int,
    temporal_sample_count: int = 1,
) -> tuple[tuple[int, ...], dict[str, Any], tuple[tuple[int, ...], ...]]:
    right = _target(session.world, ActionKind.LOOK, 2)
    signatures = []
    frames = []
    raw_samples = []
    for step in range(period):
        if step:
            _turn(session, right)
        frame, samples = _temporal_frame(session, temporal_sample_count)
        frames.append(frame)
        raw_samples.append(samples)
        signatures.append(encoder.scene_encoder.encode(frame))
    _turn(session, right)
    session.records.append(
        session.world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2))
    )
    panorama = encoder.encode(frames)
    return (
        panorama,
        _save_scan(
            output,
            tuple(frames),
            panorama,
            tuple(raw_samples) if temporal_sample_count > 1 else None,
        ),
        tuple(signatures),
    )


def _move(
    session: Session,
    *,
    moves: int,
    move_frames: int,
    target_id: str,
) -> None:
    if target_id not in _movement_targets(session.world):
        raise ValueError("selected opaque movement handle is unavailable")
    for _ in range(moves):
        session.records.append(
            session.world.step(
                PrimitiveAction(
                    kind=ActionKind.MOVE,
                    target_id=target_id,
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


def _orient(session: Session, steps: int) -> None:
    if steps < 0:
        raise ValueError("orientation steps must be nonnegative")
    right = _target(session.world, ActionKind.LOOK, 2)
    for _ in range(steps):
        _turn(session, right)
    if steps:
        session.records.append(
            session.world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=2))
        )


def _save_probe_frame(path: Path, frame: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_heading(
    session: Session,
    *,
    encoder: RGBPanoramaEncoder,
    heading_steps: int,
    program_index: int,
    movement_target_id: str,
    period: int,
    moves: int,
    move_frames: int,
    output: Path,
) -> dict[str, Any]:
    """Measure a heading's visual out-and-back reversibility without coordinates."""

    _orient(session, heading_steps)
    start_frame = session.recorder.frames[-1].copy()
    start = encoder.scene_encoder.encode(start_frame)
    _move(
        session,
        moves=moves,
        move_frames=move_frames,
        target_id=movement_target_id,
    )
    outbound_frame = session.recorder.frames[-1].copy()
    outbound = encoder.scene_encoder.encode(outbound_frame)
    _half_turn(session, period)
    _move(
        session,
        moves=moves,
        move_frames=move_frames,
        target_id=movement_target_id,
    )
    _half_turn(session, period)
    return_frame = session.recorder.frames[-1].copy()
    returned = encoder.scene_encoder.encode(return_frame)
    outbound_distance = math.dist(start, outbound)
    return_distance = math.dist(start, returned)
    root = output / f"heading-{heading_steps:02d}-program-{program_index:02d}"
    hashes = {
        "start.png": _save_probe_frame(root / "start.png", start_frame),
        "outbound.png": _save_probe_frame(root / "outbound.png", outbound_frame),
        "return.png": _save_probe_frame(root / "return.png", return_frame),
    }
    return {
        "heading_steps": heading_steps,
        "program_index": program_index,
        "movement_target_id": movement_target_id,
        "outbound_distance": outbound_distance,
        "return_distance": return_distance,
        "reversibility_score": outbound_distance / (return_distance + 1.0),
        "image_sha256": hashes,
        "signature_sha256": {
            name: hashlib.sha256(
                json.dumps(signature, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for name, signature in (
                ("start", start),
                ("outbound", outbound),
                ("return", returned),
            )
        },
    }


def _candidate_headings(
    openness_scores: tuple[float, ...], count: int
) -> tuple[int, ...]:
    if not 1 <= count <= len(openness_scores):
        raise ValueError("heading candidate count is outside the camera period")
    return tuple(
        sorted(
            range(len(openness_scores)),
            key=lambda index: (-openness_scores[index], index),
        )[:count]
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
        "navigation_probe.py": Path(__file__).with_name("navigation_probe.py"),
        "memory.py": root / "src/minecraft_scientist/exploration/memory.py",
        "perception.py": root / "src/minecraft_scientist/exploration/perception.py",
        "minestudio.py": root / "src/minecraft_scientist/minecraft/minestudio.py",
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
    parser.add_argument(
        "--auto-margin",
        action="store_true",
        help="derive the maximum-gap place radius from training positives/negatives",
    )
    parser.add_argument(
        "--temporal-samples",
        type=int,
        default=1,
        help="odd number of RGB frames median-aggregated at every panorama heading",
    )
    parser.add_argument("--training-time", choices=TIME_CHOICES, default="noon")
    parser.add_argument("--training-weather", choices=WEATHER_CHOICES, default="clear")
    parser.add_argument("--heldout-time", choices=TIME_CHOICES, default="noon")
    parser.add_argument("--heldout-weather", choices=WEATHER_CHOICES, default="clear")
    parser.add_argument(
        "--require-condition-shift",
        action="store_true",
        help="fail unless held-out lighting/weather differs from training",
    )
    parser.add_argument(
        "--panorama-encoder",
        choices=tuple(PANORAMA_ENCODERS),
        default="lab",
    )
    parser.add_argument(
        "--select-open-heading",
        action="store_true",
        help="select a persistent movement heading from the training RGB scan",
    )
    parser.add_argument(
        "--select-reversible-heading",
        action="store_true",
        help="probe visually open headings and select one by RGB out-and-back recovery",
    )
    parser.add_argument(
        "--heading-probe-candidates",
        type=int,
        default=6,
        help="number of visually open headings tested for RGB reversibility",
    )
    parser.add_argument(
        "--enable-jump-program",
        action="store_true",
        help="add an anonymous forward-plus-jump candidate to reversible probing",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.moves < 1 or args.move_frames < 1:
        parser.error("movement budgets must be positive")
    if args.minimum_turn_steps < 4 or args.maximum_turn_steps < args.minimum_turn_steps:
        parser.error("invalid camera-period search interval")
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("margin must be finite and nonnegative")
    if args.temporal_samples < 1 or args.temporal_samples % 2 == 0:
        parser.error("temporal samples must be a positive odd integer")
    if args.select_open_heading and args.select_reversible_heading:
        parser.error("open and reversible heading selection are mutually exclusive")
    if args.enable_jump_program and not args.select_reversible_heading:
        parser.error("the jump program requires reversible heading selection")
    args.output.mkdir(parents=True, exist_ok=False)
    training_commands = natural_world_commands(
        time_of_day=args.training_time,
        weather=args.training_weather,
    )
    heldout_commands = natural_world_commands(
        time_of_day=args.heldout_time,
        weather=args.heldout_weather,
    )
    condition_shifted = training_commands != heldout_commands
    if args.require_condition_shift and not condition_shifted:
        parser.error("held-out conditions must differ from training conditions")
    encoder = PANORAMA_ENCODERS[args.panorama_encoder]()
    movement_programs = (
        (("forward",), ("forward", "jump"))
        if args.enable_jump_program
        else (("forward",),)
    )
    source_hashes = _copy_sources(args.output)

    sessions: list[Session] = []
    scans: dict[str, dict[str, Any]] = {}
    heading_probe: dict[str, Any] | None = None
    selected_program_index = 0
    selected_probe_metrics: dict[str, float | None] | None = None
    try:
        discovery = _open_session(args.seed, training_commands, movement_programs)
        sessions.append(discovery)
        (
            period,
            anchor,
            scans["discovery"],
            discovered_heading,
            _discovery_views,
        ) = _discover_scan(
            discovery,
            args.output / "scans/discovery",
            encoder=encoder,
            minimum_steps=args.minimum_turn_steps,
            maximum_steps=args.maximum_turn_steps,
        )
        movement_target_ids = _movement_targets(discovery.world)
        discovery.world.close()
        (args.output / "discovery-evaluator-checkpoint.json").write_text(
            json.dumps(
                {
                    "positions": discovery.audit.positions,
                    "orientation_return_error": _orientation_return_error(
                        discovery.audit.positions
                    ),
                    "startup_attempts": discovery.startup_attempts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if args.select_reversible_heading:
            scores = tuple(float(item) for item in scans["discovery"]["openness_scores"])
            try:
                candidates = _candidate_headings(
                    scores,
                    args.heading_probe_candidates,
                )
            except ValueError as error:
                parser.error(str(error))
            probe_results = []
            for candidate in candidates:
                for program_index, movement_target_id in enumerate(
                    movement_target_ids
                ):
                    probe = _open_session(
                        args.seed, training_commands, movement_programs
                    )
                    sessions.append(probe)
                    result = _probe_heading(
                        probe,
                        encoder=encoder,
                        heading_steps=candidate,
                        program_index=program_index,
                        movement_target_id=movement_target_id,
                        period=period,
                        moves=args.moves,
                        move_frames=args.move_frames,
                        output=args.output / "heading-probes",
                    )
                    result["session_index"] = len(sessions) - 1
                    probe_results.append(result)
                    probe.world.close()
            selected = max(
                probe_results,
                key=lambda item: (
                    float(item["reversibility_score"]),
                    float(item["outbound_distance"]),
                    -int(item["heading_steps"]),
                    -int(item["program_index"]),
                ),
            )
            heading_steps = int(selected["heading_steps"])
            selected_program_index = int(selected["program_index"])
            selected_movement_target_id = str(selected["movement_target_id"])
            selected_probe_metrics = _position_metrics(
                sessions[int(selected["session_index"])].audit.positions
            )
            heading_probe = {
                "selector": "rgb-reversibility-v2",
                "candidate_count": len(candidates),
                "candidate_headings": list(candidates),
                "movement_target_ids": list(movement_target_ids),
                "results": probe_results,
                "selected_heading_steps": heading_steps,
                "selected_program_index": selected_program_index,
                "selected_movement_target_id": selected_movement_target_id,
                "selected_evaluator_metrics": selected_probe_metrics,
            }
            (args.output / "heading-probe.json").write_text(
                json.dumps(heading_probe, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (args.output / "heading-probe-evaluator.json").write_text(
                json.dumps(
                    {
                        "sessions": [
                            {
                                "positions": session.audit.positions,
                                "startup_attempts": session.startup_attempts,
                            }
                            for session in sessions
                        ],
                        "selected_session_index": int(selected["session_index"]),
                        "selected_metrics": selected_probe_metrics,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            heading_steps = discovered_heading if args.select_open_heading else 0
            selected_movement_target_id = movement_target_ids[0]

        training = _open_session(args.seed, training_commands, movement_programs)
        sessions.append(training)
        _orient(training, heading_steps)
        positive_one, scans["train_positive_one"], _positive_one_views = _capture_scan(
            training,
            args.output / "scans/train-positive-one",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
        )
        _move(
            training,
            moves=args.moves,
            move_frames=args.move_frames,
            target_id=selected_movement_target_id,
        )
        training_negative, scans["train_negative"], _negative_views = _capture_scan(
            training,
            args.output / "scans/train-negative",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
        )
        training.world.close()

        second_positive_session = _open_session(
            args.seed, training_commands, movement_programs
        )
        sessions.append(second_positive_session)
        _orient(second_positive_session, heading_steps)
        positive_two, scans["train_positive_two"], _positive_two_views = _capture_scan(
            second_positive_session,
            args.output / "scans/train-positive-two",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
        )
        second_positive_session.world.close()

        training_loop = _open_session(
            args.seed, training_commands, movement_programs
        )
        sessions.append(training_loop)
        _orient(training_loop, heading_steps)
        _move(
            training_loop,
            moves=args.moves,
            move_frames=args.move_frames,
            target_id=selected_movement_target_id,
        )
        _half_turn(training_loop, period)
        _move(
            training_loop,
            moves=args.moves,
            move_frames=args.move_frames,
            target_id=selected_movement_target_id,
        )
        _half_turn(training_loop, period)
        training_return, scans["train_loop_return"], _return_views = _capture_scan(
            training_loop,
            args.output / "scans/train-loop-return",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
        )
        training_loop.world.close()

        memory_path = args.output / "exploration.sqlite3"
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
                margin=None if args.auto_margin else args.margin,
            )

        continual_adaptation = {
            "policy": "accepted-view-bounded-coreset-v1",
            "heldout_start_consolidated": False,
            "heldout_return_consolidated": False,
        }
        loop_session_index = len(sessions)
        loop = _open_session(args.seed, heldout_commands, movement_programs)
        sessions.append(loop)
        _orient(loop, heading_steps)
        loop_start, scans["heldout_start"], _loop_start_views = _capture_scan(
            loop,
            args.output / "scans/heldout-start",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
        )
        with ExplorationMemory.open(memory_path) as memory:
            start_match = memory.match_modality_place(encoder.VERSION, loop_start)
            if start_match == place.place_id:
                memory.consolidate_modality_view(
                    encoder.VERSION,
                    place.place_id,
                    loop_start,
                    episode_id=4,
                    step=0,
                )
                continual_adaptation["heldout_start_consolidated"] = True
        _move(
            loop,
            moves=args.moves,
            move_frames=args.move_frames,
            target_id=selected_movement_target_id,
        )
        loop_outbound, scans["heldout_outbound"], _outbound_views = _capture_scan(
            loop,
            args.output / "scans/heldout-outbound",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
        )
        with ExplorationMemory.open(memory_path) as memory:
            outbound_match = memory.match_modality_place(
                encoder.VERSION, loop_outbound
            )
        _half_turn(loop, period)
        _move(
            loop,
            moves=args.moves,
            move_frames=args.move_frames,
            target_id=selected_movement_target_id,
        )
        _half_turn(loop, period)
        loop_return, scans["heldout_return"], _return_views = _capture_scan(
            loop,
            args.output / "scans/heldout-return",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
        )
        with ExplorationMemory.open(memory_path) as memory:
            return_match = memory.match_modality_place(encoder.VERSION, loop_return)
            if return_match == place.place_id:
                memory.consolidate_modality_view(
                    encoder.VERSION,
                    place.place_id,
                    loop_return,
                    episode_id=4,
                    step=1,
                )
                continual_adaptation["heldout_return_consolidated"] = True
        loop.world.close()
        loop_metrics = _position_metrics(loop.audit.positions)
        (args.output / "loop-evaluator-checkpoint.json").write_text(
            json.dumps(
                {
                    "positions": loop.audit.positions,
                    "metrics": loop_metrics,
                    "startup_attempts": loop.startup_attempts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_events(args.output / "events-checkpoint.jsonl", sessions)

        restart = _open_session(args.seed, heldout_commands, movement_programs)
        sessions.append(restart)
        _orient(restart, heading_steps)
        restart_signature, scans["heldout_restart"], _restart_views = _capture_scan(
            restart,
            args.output / "scans/heldout-restart",
            encoder=encoder,
            period=period,
            temporal_sample_count=args.temporal_samples,
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
        orientation_error = _orientation_return_error(discovery.audit.positions)
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
            "selected_probe_physical_reversibility": (
                selected_probe_metrics is None
                or (
                    selected_probe_metrics["maximum_excursion"] is not None
                    and selected_probe_metrics["maximum_excursion"] >= 2.0
                    and selected_probe_metrics["return_error"] is not None
                    and selected_probe_metrics["return_error"] <= 1.5
                )
            ),
            "memory_integrity": memory_summary["integrity_valid"] is True,
            "required_condition_shift": (
                condition_shifted or not args.require_condition_shift
            ),
        }
        report = {
            "suite": "natural_visual_loop",
            "backend": "real-minestudio",
            "seed": args.seed,
            "terrain_modified": False,
            "controlled_conditions": {
                "training": list(training_commands),
                "heldout": list(heldout_commands),
            },
            "condition_split": {
                "training": {
                    "time": args.training_time,
                    "weather": args.training_weather,
                },
                "heldout": {
                    "time": args.heldout_time,
                    "weather": args.heldout_weather,
                },
                "shifted": condition_shifted,
                "required": args.require_condition_shift,
            },
            "encoder": encoder.VERSION,
            "descriptor_dimensions": encoder.dimensions,
            "temporal_samples": args.temporal_samples,
            "calibration_strategy": (
                "maximum-gap-training-only" if args.auto_margin else "fixed-margin"
            ),
            "inferred_camera_period": period,
            "evaluator_orientation_return_error": orientation_error,
            "moves_each_leg": args.moves,
            "move_frames": args.move_frames,
            "selected_heading_steps": heading_steps,
            "selected_program_index": selected_program_index,
            "selected_movement_target_id": selected_movement_target_id,
            "open_heading_selection_enabled": args.select_open_heading,
            "reversible_heading_selection_enabled": args.select_reversible_heading,
            "jump_program_enabled": args.enable_jump_program,
            "session_startup_attempts": [
                session.startup_attempts for session in sessions
            ],
            "heading_probe": heading_probe,
            "continual_adaptation": continual_adaptation,
            "calibration": calibrated.calibration.model_dump(mode="json"),
            "loop_evaluator_metrics": loop_metrics,
            "memory": memory_summary,
            "scans": scans,
            "source_sha256": source_hashes,
            "gate": {**gate, "passed": all(gate.values())},
            "scope": (
                "Natural generated terrain with preregistered training/held-out "
                "lighting and weather; pixel-only panoramic place retrieval with "
                "evaluator-only coordinates. This is a bounded out-and-back loop, "
                "not open-world survival or SLAM."
            ),
        }
        evaluator = {
            "sessions": [
                {
                    "positions": session.audit.positions,
                    "raw_metadata_keys": sorted(session.audit.metadata_keys),
                    "startup_attempts": session.startup_attempts,
                }
                for session in sessions
            ],
            "loop_session_index": loop_session_index,
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
                    "auto_margin": args.auto_margin,
                    "temporal_samples": args.temporal_samples,
                    "training_time": args.training_time,
                    "training_weather": args.training_weather,
                    "heldout_time": args.heldout_time,
                    "heldout_weather": args.heldout_weather,
                    "require_condition_shift": args.require_condition_shift,
                    "panorama_encoder": args.panorama_encoder,
                    "select_open_heading": args.select_open_heading,
                    "select_reversible_heading": args.select_reversible_heading,
                    "heading_probe_candidates": args.heading_probe_candidates,
                    "enable_jump_program": args.enable_jump_program,
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

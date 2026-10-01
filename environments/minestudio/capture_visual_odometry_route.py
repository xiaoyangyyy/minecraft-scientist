"""Capture dense public RGB along an audited natural route for visual odometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from natural_visual_goal_search_benchmark import _prepare_source, _replay, _source_paths
from natural_visual_loop_benchmark import (
    PANORAMA_ENCODERS,
    PRIVILEGED_ARTIFACT_TOKENS,
    CommandsCallback,
    MinecraftSim,
    NaturalRecorder,
    Session,
    _factory,
    _write_events,
    natural_world_commands,
)
from navigation_probe import PositionAudit, PositionAuditSimulator
from PIL import Image

from minecraft_scientist.exploration.resolution import ResolutionNormalizedSceneEncoder
from minecraft_scientist.minecraft import MineStudioConfig, MineStudioVisualAdapter
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


class DenseRGBSimulator:
    """Capture only public RGB after the privileged telemetry interceptor."""

    def __init__(self, simulator: Any, frames: list[np.ndarray]) -> None:
        self.simulator = simulator
        self.frames = frames

    def _capture(self, frame: Any) -> None:
        array = np.asarray(frame["image"] if isinstance(frame, dict) else frame)
        if len(self.frames) >= 5000:
            raise RuntimeError("dense RGB capture budget exhausted")
        self.frames.append(array.copy())

    def reset(self) -> Any:
        frame, info = self.simulator.reset()
        self._capture(frame)
        return frame, info

    def step(self, action: Any) -> Any:
        frame, reward, terminated, truncated, info = self.simulator.step(action)
        self._capture(frame)
        return frame, reward, terminated, truncated, info

    def noop_action(self) -> Any:
        return self.simulator.noop_action()

    def close(self) -> None:
        self.simulator.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-size", type=int, choices=(224, 448), default=224)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = _prepare_source(args.source_run.resolve(), args.output)
    source_paths = _source_paths() | {Path(__file__).name: Path(__file__).resolve()}
    source_paths["resolution.py"] = Path(
        ResolutionNormalizedSceneEncoder.encode.__code__.co_filename
    )
    source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir()
    for name, data in source_bytes.items():
        (snapshot / name).write_bytes(data)
    source_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()
    }
    source_memory_hash = hashlib.sha256(
        (args.output / "exploration.sqlite3").read_bytes()
    ).hexdigest()
    frames: list[np.ndarray] = []
    recorder = NaturalRecorder((("forward",), ("forward", "jump")))
    audit = PositionAudit()
    commands = natural_world_commands(
        time_of_day=spec["condition"]["time"], weather=spec["condition"]["weather"]
    )

    def factory(config):
        if args.observation_size == 224:
            simulator = _factory(config, audit, commands)
        else:
            simulator = PositionAuditSimulator(
                MinecraftSim(
                    action_type="env",
                    seed=config.seed,
                    obs_size=config.obs_size,
                    render_size=config.render_size,
                    callbacks=[CommandsCallback(list(commands))],
                ),
                audit,
            )
        return DenseRGBSimulator(simulator, frames)

    factor = args.observation_size // 224
    encoder = PANORAMA_ENCODERS["edge"]()
    with Image.open(args.output / "goal-reference/view-00.png") as reference:
        policy_size = reference.size
    if policy_size != (args.observation_size, args.observation_size):
        encoder.scene_encoder = ResolutionNormalizedSceneEncoder(
            encoder.scene_encoder, size=policy_size
        )
    world = MineStudioVisualAdapter(
        recorder,
        MineStudioConfig(
            obs_size=(args.observation_size, args.observation_size),
            render_size=(640 * factor, 360 * factor),
        ),
        simulator_factory=factory,
    )
    scans = {}
    session = Session(world=world, recorder=recorder, audit=audit, records=[])
    replay = None
    failure = None
    try:
        world.reset(spec["seed"])
        session.records.append(
            world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=20))
        )
        replay = _replay(
            session,
            output=args.output,
            scans=scans,
            prefix="route",
            spec=spec,
            encoder=encoder,
            samples=3,
            route=spec["route"],
            memory_path=args.output / "source-evidence/exploration.sqlite3",
        )
    except Exception as error:  # noqa: BLE001 -- preserve acquisition failures and exit nonzero
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        world.close()
    directory = args.output / "frames"
    directory.mkdir()
    image_hashes = {}
    rgb_hashes = []
    for index, frame in enumerate(frames):
        path = directory / f"frame-{index:05d}.png"
        Image.fromarray(frame).save(path)
        image_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rgb_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
    timeline = []
    frame_index = 0
    alignment_valid = True
    for record in session.records:
        before_index = frame_index
        frame_index += int(record.event.duration)
        alignment_valid = alignment_valid and frame_index < len(frames)
        if frame_index < len(frames):
            alignment_valid = alignment_valid and (
                record.state_before.agent_state["frame_sha256"]
                == rgb_hashes[before_index]
                and record.state_after.agent_state["frame_sha256"]
                == rgb_hashes[frame_index]
            )
        timeline.append(
            {
                "start_frame": before_index,
                "end_frame": frame_index,
                "action": record.executed_action.model_dump(mode="json"),
                "transition_id": record.transition_id,
            }
        )
    _write_events(args.output / "events.jsonl", [session])
    public = (args.output / "events.jsonl").read_text(encoding="utf-8")
    report = {
        "suite": "dense_visual_route_acquisition",
        "seed": spec["seed"],
        "source_spec": spec,
        "source_sha256": source_hashes,
        "observation_size": args.observation_size,
        "policy_reference_size": policy_size,
        "render_size": [640 * factor, 360 * factor],
        "replay": replay,
        "scans": scans,
        "timeline": timeline,
        "frame_count": len(frames),
        "frame_sha256": image_hashes,
        "rgb_sha256": rgb_hashes,
        "failure": failure,
        "gate": {
            "route_recognized": replay is not None,
            "public_actions_clean": not any(
                token in public for token in PRIVILEGED_ARTIFACT_TOKENS
            ),
            "image_action_alignment": alignment_valid
            and frame_index == len(frames) - 1,
            "actions_compliant": bool(session.records)
            and all(
                item.intervention_compliance and item.event.success
                for item in session.records
            ),
            "frame_position_alignment": len(frames) == len(audit.positions),
            "runtime_sources_unchanged": all(
                path.read_bytes() == source_bytes[name]
                for name, path in source_paths.items()
            ),
            "accepted_memory_unchanged": hashlib.sha256(
                (args.output / "exploration.sqlite3").read_bytes()
            ).hexdigest()
            == source_memory_hash,
        },
    }
    report["gate"]["passed"] = all(report["gate"].values())
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "evaluator-audit.json").write_text(
        json.dumps({"positions": audit.positions}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"frame_count": len(frames), "gate": report["gate"], "failure": failure}
        )
    )
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

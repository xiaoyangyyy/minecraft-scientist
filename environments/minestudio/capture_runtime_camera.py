"""Fresh evaluator-only image/camera timing diagnostic; no shared engine edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

ROOT = Path(__file__).resolve().parents[2]
JAVA = Path("/usr/lib/jvm/java-8-openjdk-amd64/bin")
ENGINE = Path("/opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar")
# All actions are fixed before observation, including two turns and inertia waits.
SCHEDULE = (
    [("wait", None)] * 12
    + [("look", None), ("wait", None), ("wait", None)] * 12
    + [("move", 0)] * 6
    + [("move", 2)] * 4
    + [("move", 1)] * 4
    + [("wait", None)] * 12
    + [("look", None)]
    + [("wait", None)] * 4
    + [("move", 0)] * 6
    + [("move", 3)] * 4
    + [("move", 2)] * 4
    + [("wait", None)] * 12
)
REFERENCE_INDICES = tuple(range(15, 49, 3))
TRIAL_INTERVALS = ((48, 58, 62), (79, 89, 93))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@contextmanager
def isolated_agent_launch(instance_class, jar, output):
    """Only a fresh JVM launched in this Python process receives instrumentation."""
    jar, output = Path(jar).resolve(strict=True), Path(output).resolve()
    if output.exists() or any(char.isspace() for char in str(jar) + str(output)):
        raise ValueError("agent requires new output and whitespace-free paths")
    # Preserve existing JVM controls; do not inspect or persist their contents.
    previous = os.environ.get("JAVA_TOOL_OPTIONS")
    if previous and "-javaagent:" in previous:
        raise ValueError("another Java agent makes timing instrumentation ambiguous")
    launcher = instance_class._launch_minecraft
    launches = []

    def launch(instance, *args, **kwargs):
        if launches:
            raise RuntimeError(
                "runtime timing diagnostic permits exactly one fresh JVM"
            )
        launches.append(True)
        os.environ["JAVA_TOOL_OPTIONS"] = (
            previous + " " if previous else ""
        ) + f"-javaagent:{jar}={output}"
        try:
            return launcher(instance, *args, **kwargs)
        finally:
            if previous is None:
                os.environ.pop("JAVA_TOOL_OPTIONS", None)
            else:
                os.environ["JAVA_TOOL_OPTIONS"] = previous

    instance_class._launch_minecraft = launch
    try:
        yield launches
    finally:
        instance_class._launch_minecraft = launcher
        if os.environ.get("JAVA_TOOL_OPTIONS") != previous:
            raise RuntimeError("JVM environment not restored")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--active-causal",
        action="store_true",
        help="dispatch public-only first-eligible map acquisition after calibration",
    )
    parser.add_argument("--acquisition-strategy", choices=("past_support", "fixed_cap"))
    parser.add_argument("--settled-controls", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.acquisition_strategy is not None and not args.active_causal:
        parser.error("acquisition strategy requires --active-causal")
    if args.settled_controls and (
        not args.active_causal or args.acquisition_strategy == "fixed_cap"
    ):
        parser.error("settled controls require active past-support acquisition")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    build = directory / "agent-build"
    build.mkdir()
    source = Path(__file__).with_name("camera_audit_agent")
    subprocess.run(
        [
            str(JAVA / "javac"),
            "-XDignore.symbol.file",
            "-d",
            str(build),
            str(source / "CameraAudit.java"),
        ],
        check=True,
        timeout=60,
    )
    jar = directory / "camera-audit.jar"
    subprocess.run(
        [
            str(JAVA / "jar"),
            "cfm",
            str(jar),
            str(source / "MANIFEST.MF"),
            "-C",
            str(build),
            ".",
        ],
        check=True,
        timeout=30,
    )
    from capture_controlled_rgb_motion import ProbeRecorder
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks.commands import CommandsCallback
    from minestudio.simulator.minerl.env.malmo import InstanceManager, MinecraftInstance
    from navigation_probe import PositionAudit, PositionAuditSimulator

    from minecraft_scientist.minecraft import MineStudioConfig, MineStudioVisualAdapter
    from minecraft_scientist.minecraft.client_camera_profile import runtime_ledger
    from minecraft_scientist.protocols import ActionKind, PrimitiveAction

    if InstanceManager._instance_pool:
        raise RuntimeError("timing diagnostic requires a fresh instance process")
    runtime_dir = Path(InstanceManager.RUNTIME_DIR)
    runtime_before = runtime_ledger(runtime_dir)
    sources = {
        "capture_runtime_camera.py": Path(__file__),
        "CameraAudit.java": source / "CameraAudit.java",
        "MANIFEST.MF": source / "MANIFEST.MF",
        "probe-recorder.py": Path(__import__(ProbeRecorder.__module__).__file__),
        "position-audit.py": Path(__import__(PositionAudit.__module__).__file__),
        "sdk-entry.py": Path(
            __import__(MinecraftSim.__module__, fromlist=["__file__"]).__file__
        ),
        "sdk-launcher.py": Path(
            MinecraftInstance._launch_minecraft.__code__.co_filename
        ),
    } | {
        "package/" + path.relative_to(ROOT / "src/minecraft_scientist").as_posix(): path
        for path in sorted((ROOT / "src/minecraft_scientist").rglob("*.py"))
    }
    for name, path in sources.items():
        target = directory / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    config = {
        "schema": "runtime-camera-projection-v2",
        "seed": args.seed,
        "schedule": SCHEDULE,
        "reference_indices": REFERENCE_INDICES,
        "trial_intervals": TRIAL_INTERVALS,
        "render_size": [1280, 720],
        "observation_size": [448, 448],
        "engine_sha256": digest(ENGINE),
        "agent_sha256": digest(jar),
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "sdk_runtime_sha256": runtime_before,
        "scope": "fresh full world-render projection diagnostic; all matrix telemetry evaluator-only; preregistered distinct-control trials; no navigation promotion",
    }
    if args.active_causal:
        from minecraft_scientist.exploration.active_causal_map import ActiveMapPolicy

        config.update(
            schema="active-causal-camera-projection-v2",
            schedule=SCHEDULE[:48],
            trial_intervals=[],
            active_policy=ActiveMapPolicy(
                control_strategy=args.acquisition_strategy or "past_support"
            ).model_dump(mode="json"),
            active_seed=2000,
            scope="live public-only first-eligible acquisition and distinct two-tick validation; one development trajectory; no navigation promotion",
        )
        if args.settled_controls:
            from minecraft_scientist.exploration.settled_map_control import (
                SettledControlPolicy,
            )

            config["settled_policy"] = SettledControlPolicy().model_dump(mode="json")
            config["schema"] = "settled-causal-camera-projection-v1"
            config["scope"] = (
                "live public frozen-map MOVE/WAIT controls and returned-image clock; no player timestamp inference, goal navigation or held-out claim"
            )
    write_json(directory / "config.json", config)
    config_hash = digest(directory / "config.json")
    audit, images, native_images, native_hashes, records = (
        PositionAudit(),
        [],
        [],
        [],
        [],
    )

    class Capture(PositionAuditSimulator):
        def _record(self, frame, info):
            super()._record(frame, info)
            images.append(np.array(frame["image"], copy=True))
            # SDK flips GL's bottom-up buffer; reverse it to match the JVM digest.
            native = np.asarray(info["pov"])
            native_images.append(np.array(native, copy=True))
            native_hashes.append(hashlib.sha256(native[::-1].tobytes()).hexdigest())

    commands = [
        "/gamerule sendCommandFeedback false",
        "/gamemode creative",
        "/difficulty peaceful",
        "/weather clear",
        "/time set noon",
        "/gamerule doDaylightCycle false",
        "/gamerule doWeatherCycle false",
        "/clear @p",
    ]

    def factory(settings):
        return Capture(
            MinecraftSim(
                action_type="env",
                seed=settings.seed,
                obs_size=settings.obs_size,
                render_size=settings.render_size,
                callbacks=[CommandsCallback(commands)],
            ),
            audit,
        )

    world = MineStudioVisualAdapter(
        ProbeRecorder(),
        MineStudioConfig(obs_size=(448, 448), render_size=(1280, 720)),
        simulator_factory=factory,
    )
    error = None
    active = None
    launches = []
    try:
        with isolated_agent_launch(
            MinecraftInstance, jar, directory / "evaluator-camera.jsonl"
        ) as launches:
            world.reset(args.seed)
            for kind, index in config["schedule"]:
                target = None
                if kind == "move":
                    target = [
                        obj.object_id
                        for obj in world.observe().objects
                        if obj.affordances == (ActionKind.MOVE,)
                    ][index]
                elif kind == "look":
                    target = next(
                        obj.object_id
                        for obj in world.observe().objects
                        if ActionKind.LOOK in obj.affordances
                        and obj.observable_features.get("screen_row") == 1
                        and obj.observable_features.get("screen_column") == 2
                    )
                records.append(
                    world.step(
                        PrimitiveAction(
                            kind=ActionKind(kind), target_id=target, ticks=1
                        )
                    )
                )
            if args.active_causal:
                from minecraft_scientist.exploration.active_causal_map import (
                    acquire_causal_map,
                )
                from minecraft_scientist.exploration.online_landmark_probe import (
                    calibrate_reference_camera,
                )
                from minecraft_scientist.exploration.visibility_acquisition import (
                    PublicProbeStep,
                )

                calibration = calibrate_reference_camera(
                    [images[slot] for slot in REFERENCE_INDICES], (1280, 720)
                )

                def execute(handle):
                    targets = [
                        obj.object_id
                        for obj in world.observe().objects
                        if obj.affordances == (ActionKind.MOVE,)
                    ]
                    record = world.step(
                        PrimitiveAction(
                            kind=ActionKind.MOVE, target_id=targets[handle], ticks=1
                        )
                    )
                    records.append(record)
                    actual = (
                        targets.index(record.executed_action.target_id)
                        if record.executed_action.target_id in targets
                        else -1
                    )
                    compliant = (
                        record.executed_action.kind is ActionKind.MOVE
                        and record.intervention_compliance is True
                        and record.event.success is True
                        and record.controller_failure is None
                        and record.event.duration == 1
                    )
                    return PublicProbeStep(
                        images[-1].copy(),
                        actual,
                        compliant,
                        record.executed_action.ticks,
                    )

                active = {
                    "calibration": calibration,
                    "acquisition": acquire_causal_map(
                        images[-1].copy(),
                        execute,
                        calibration["camera_intrinsics"],
                        anchor_frame=48,
                        seed=config["active_seed"],
                        policy=config["active_policy"],
                    ),
                }
                if active["acquisition"]["failure"] is not None:
                    raise RuntimeError(
                        "active causal acquisition failed; partial evidence retained"
                    )
                if args.settled_controls:
                    from minecraft_scientist.exploration.settled_map_control import (
                        PublicControlStep,
                        run_settled_controls,
                    )

                    def execute_control(kind, handle):
                        targets = [
                            obj.object_id
                            for obj in world.observe().objects
                            if obj.affordances == (ActionKind.MOVE,)
                        ]
                        record = world.step(
                            PrimitiveAction(
                                kind=ActionKind(kind),
                                ticks=1,
                                target_id=targets[handle] if kind == "move" else None,
                            )
                        )
                        records.append(record)
                        actual = (
                            targets.index(record.executed_action.target_id)
                            if record.executed_action.target_id in targets
                            else None
                        )
                        compliant = (
                            record.intervention_compliance is True
                            and record.event.success is True
                            and record.controller_failure is None
                            and record.event.duration == 1
                        )
                        return PublicControlStep(
                            images[-1].copy(),
                            record.executed_action.kind.value,
                            actual,
                            compliant,
                            record.executed_action.ticks,
                        )

                    active["settled"] = run_settled_controls(
                        active["acquisition"],
                        images[48:],
                        execute_control,
                        policy=config["settled_policy"],
                    )
                    if active["settled"]["failure"] is not None:
                        raise RuntimeError(
                            "settled controls failed; partial public evidence retained"
                        )
    except Exception as failure:  # noqa: BLE001 -- preserve real launch/capture failures
        error = {"type": type(failure).__name__, "message": str(failure)}
    finally:
        world.close()
    frame_dir = directory / "frames"
    frame_dir.mkdir()
    native_dir = directory / "evaluator-native"
    native_dir.mkdir()
    for slot, frame in enumerate(images):
        Image.fromarray(frame).save(frame_dir / f"frame-{slot:05d}.png")
        Image.fromarray(native_images[slot]).save(native_dir / f"frame-{slot:05d}.png")
    (directory / "events.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in records), encoding="utf-8"
    )
    write_json(
        directory / "evaluator-positions.json",
        {"positions": audit.positions, "native_rgb_sha256": native_hashes},
    )
    checks = {
        "one_fresh_instrumented_launch": len(launches) == 1,
        "complete_frames_actions_positions": (
            args.active_causal or len(records) == len(SCHEDULE)
        )
        and len(images)
        == len(native_images)
        == len(native_hashes)
        == len(audit.positions)
        == len(records) + 1,
        "sdk_runtime_unchanged": runtime_ledger(runtime_dir) == runtime_before,
        "engine_agent_unchanged": digest(ENGINE) == config["engine_sha256"]
        and digest(jar) == config["agent_sha256"],
        "sources_unchanged": all(
            digest(path) == config["source_sha256"][name]
            for name, path in sources.items()
        ),
        "config_unchanged": digest(directory / "config.json") == config_hash,
    }
    if args.active_causal:
        checks["active_policy_completed"] = (
            active is not None
            and active["acquisition"]["failure"] is None
            and active["acquisition"]["terminal"]
            in ("validated", "rejected", "exhausted", "support_aborted")
            and len(records)
            == 48
            + len(active["acquisition"]["steps"])
            + len(active.get("settled", {}).get("steps", []))
        )
        if args.settled_controls:
            checks["settled_policy_completed"] = (
                active is not None
                and "settled" in active
                and active["settled"]["failure"] is None
                and active["settled"]["terminal"]
                in ("completed", "not_started", "support_aborted", "settle_exhausted")
            )
        write_json(directory / "active-acquisition.json", active)
    report = {
        "completed": error is None and all(checks.values()),
        "failure": error,
        "checks": checks,
        "frame_count": len(images),
        "config_sha256": config_hash,
        "navigation_accepted": False,
        "artifact_sha256": {
            path.relative_to(directory).as_posix(): digest(path)
            for path in [
                directory / "events.jsonl",
                directory / "evaluator-positions.json",
                directory / "evaluator-camera.jsonl",
            ]
            + sorted(frame_dir.glob("*.png"))
            + sorted(native_dir.glob("*.png"))
            + ([directory / "active-acquisition.json"] if args.active_causal else [])
            if path.exists()
        },
    }
    write_json(directory / "report.json", report)
    print(
        json.dumps(
            {"completed": report["completed"], "failure": error, "frames": len(images)}
        )
    )
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

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
    + [("move", 0)] * 6
    + [("move", 2)] * 6
    + [("wait", None)] * 12
    + [("look", None)]
    + [("wait", None)] * 4
    + [("move", 0)] * 6
    + [("move", 3)] * 6
    + [("wait", None)] * 12
)


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
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
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
        "schema": "runtime-camera-timing-v1",
        "seed": args.seed,
        "schedule": SCHEDULE,
        "render_size": [1280, 720],
        "observation_size": [448, 448],
        "engine_sha256": digest(ENGINE),
        "agent_sha256": digest(jar),
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "sdk_runtime_sha256": runtime_before,
        "scope": "fresh development timing diagnostic; base camera excludes bobbing matrix; evaluator-only; no navigation promotion",
    }
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
    launches = []
    try:
        with isolated_agent_launch(
            MinecraftInstance, jar, directory / "evaluator-camera.jsonl"
        ) as launches:
            world.reset(args.seed)
            for kind, index in SCHEDULE:
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
        "complete_frames_actions_positions": len(records) == len(SCHEDULE)
        and len(images)
        == len(native_images)
        == len(native_hashes)
        == len(audit.positions)
        == len(SCHEDULE) + 1,
        "sdk_runtime_unchanged": runtime_ledger(runtime_dir) == runtime_before,
        "engine_agent_unchanged": digest(ENGINE) == config["engine_sha256"]
        and digest(jar) == config["agent_sha256"],
        "sources_unchanged": all(
            digest(path) == config["source_sha256"][name]
            for name, path in sources.items()
        ),
        "config_unchanged": digest(directory / "config.json") == config_hash,
    }
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

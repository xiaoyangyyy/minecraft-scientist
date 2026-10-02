"""Public-only replay of adaptive acquisition and frozen pre-switch maps."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments/minestudio"))

import capture_visibility_rgb as capture
import controlled_motion_common as canonical
from audit_dense_correspondence import _numeric_equal

from minecraft_scientist.exploration.action_switch_motion import (
    ActionSwitchConfig,
    ActionSwitchPlan,
    inspect_action_switch,
)
from minecraft_scientist.exploration.keyframe_motion import (
    KeyframeMotionConfig,
    infer_keyframe_prefixes,
)
from minecraft_scientist.exploration.pose_direction_search import (
    inspect_direction_search,
)
from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTracks,
    track_frame_chain,
)
from minecraft_scientist.exploration.visibility_acquisition import (
    VisibilityController,
    VisibilityPolicy,
)
from minecraft_scientist.protocols import ActionKind, PrimitiveAction, TransitionRecord

RAW_CHECKS = {
    "execution_completed",
    "action_frame_alignment",
    "bounded_frames",
    "telemetry_alignment",
    "preregistration_unchanged",
    "input_sources_unchanged",
    "public_log_clean",
}


def capture_sources():
    paths = (
        capture.planner.sources()
        | capture.base.source_paths()
        | {
            "capture_visibility_rgb.py": Path(capture.__file__),
            "visibility_acquisition.py": Path(
                capture.acquire_visibility_probe.__code__.co_filename
            ),
        }
    )
    package = (
        Path(capture.MineStudioVisualAdapter.step.__code__.co_filename)
        .resolve()
        .parents[1]
    )
    return paths | {
        f"package/{path.relative_to(package).as_posix()}": path
        for path in sorted(package.rglob("*.py"))
    }


def parent_inputs(route):
    manifest = json.loads((route / "config.json").read_bytes())
    plan = ROOT / manifest["plan_directory"]
    prior = json.loads((plan / "config.json").read_bytes())
    return capture.planner.inputs(
        ROOT / prior["source_acquisition"], ROOT / prior["source_predictions"]
    ) | {
        "plan_config": plan / "config.json",
        "plan_result": plan / "plan.json",
    }


def inputs(route):
    paths = list(canonical.public_inputs(route).values()) + list(
        parent_inputs(route).values()
    )
    paths.extend((route / "source-snapshot").rglob("*.py"))
    return {path.resolve().relative_to(ROOT).as_posix(): path for path in paths}


def sources():
    return capture_sources() | {
        Path(__file__).name: Path(__file__),
        "probe_visibility_maps.py": Path(__file__).with_name(
            "probe_visibility_maps.py"
        ),
        "audit_visibility_maps.py": Path(__file__).with_name(
            "audit_visibility_maps.py"
        ),
        "audit_action_switch.py": Path(__file__).with_name("audit_action_switch.py"),
        "audit_dense_correspondence.py": Path(_numeric_equal.__code__.co_filename),
    }


def validate_capture(route):
    """Reconstruct all one-tick requests/receipts before any evaluator file read."""
    manifest = json.loads((route / "config.json").read_bytes())
    raw = json.loads((route / "report.json").read_bytes())
    runtime, parent = capture_sources(), parent_inputs(route)
    if (
        manifest["schema"] != "visibility-rgb-acquisition-v1"
        or raw["passed"] is not True
        or raw["failure"] is not None
        or raw["navigation_accepted"] is not False
        or set(raw["checks"]) != RAW_CHECKS
        or not all(value is True for value in raw["checks"].values())
        or raw["config_sha256"] != canonical.digest(route / "config.json")
        or set(manifest["source_sha256"]) != set(runtime) | {"sdk-entry.py"}
        or manifest["input_sha256"]
        != {name: canonical.digest(path) for name, path in parent.items()}
        or any(
            canonical.digest(path) != manifest["source_sha256"][name]
            for name, path in runtime.items()
        )
        or {
            path.relative_to(route / "source-snapshot").as_posix()
            for path in (route / "source-snapshot").rglob("*.py")
        }
        != set(manifest["source_sha256"])
        or any(
            canonical.digest(route / "source-snapshot" / name) != value
            for name, value in manifest["source_sha256"].items()
        )
    ):
        raise ValueError(
            "adaptive capture declarations, current sources or input/snapshot bytes invalid"
        )
    # The SDK entry is immutable provenance, not a dependency of public replay.
    # Current project package/control code is verified on portable audit hosts.
    bound = {
        path: canonical.digest(path)
        for path in list(inputs(route).values()) + list(runtime.values())
    }
    config = capture.planner.validate_plan(ROOT / manifest["plan_directory"])
    policy = VisibilityPolicy()
    expected = {
        "schema": "visibility-rgb-acquisition-v1",
        "policy": policy.model_dump(mode="json"),
        "world": {
            "seed": config.seed,
            "observation_size": config.observation_size,
            "render_size": list(config.render_size),
            "time": config.time_of_day,
            "weather": config.weather,
        },
        "startup": [
            item.model_dump(mode="json")
            for item in capture.motion_probe_schedule(config)[:25]
        ],
        "handle_pairs": config.model_dump(mode="json")["handle_pairs"],
        "repetitions": 2,
        "plan_directory": manifest["plan_directory"],
        "maximum_frames": 57
        + 2
        * len(config.handle_pairs)
        * (policy.maximum_first_ticks + policy.maximum_second_ticks + 8)
        + 8,
        "input_sha256": manifest["input_sha256"],
        "source_sha256": manifest["source_sha256"],
        "scope": "fresh visibility-aware RGB probe on development seed; no hidden telemetry in policy, no terrain edits, no pose validation or navigation promotion",
    }
    if (
        manifest != expected
        or not 1 <= raw["frame_count"] <= expected["maximum_frames"] <= 225
    ):
        raise ValueError(
            "capture differs from publicly replayed plan or fixed policy/budget"
        )
    names = {f"frame-{slot:05d}.png" for slot in range(raw["frame_count"])}
    if (
        set(raw["frame_sha256"]) != names
        or {path.name for path in (route / "frames").glob("*.png")} != names
        or len(raw["rgb_sha256"]) != len(names)
    ):
        raise ValueError(
            "adaptive capture must retain its complete contiguous frame ledger"
        )
    images = []
    for slot in range(raw["frame_count"]):
        path = route / "frames" / f"frame-{slot:05d}.png"
        image = canonical.rgb(path)
        if (
            canonical.digest(path) != raw["frame_sha256"][path.name]
            or capture.base.hashlib.sha256(image.tobytes()).hexdigest()
            != raw["rgb_sha256"][slot]
            or image.shape != (config.observation_size, config.observation_size, 3)
        ):
            raise ValueError("public frame bytes, RGB digest or dimensions differ")
        images.append(image)
    text = (route / "events.jsonl").read_text(encoding="utf-8")
    if any(
        token in text
        for token in ("location_stats", "xpos", "ypos", "zpos", '"yaw"', '"pitch"')
    ):
        raise ValueError("private telemetry appears in public transitions")
    events = [TransitionRecord.model_validate_json(line) for line in text.splitlines()]
    timeline, cursor = [], 0
    for record in events:
        end = cursor + record.event.duration
        if (
            record.event.duration != record.executed_action.ticks
            or end >= len(images)
            or record.intervention_compliance is not True
            or record.event.success is not True
            or record.controller_failure is not None
            or record.state_before.agent_state["frame_sha256"]
            != raw["rgb_sha256"][cursor]
            or record.state_after.agent_state["frame_sha256"] != raw["rgb_sha256"][end]
        ):
            raise ValueError(
                "failed/misaligned public control cannot become acquisition evidence"
            )
        timeline.append(
            {
                "start_frame": cursor,
                "end_frame": end,
                "action": record.executed_action.model_dump(mode="json"),
            }
        )
        cursor = end
    if cursor + 1 != len(images) or timeline != raw["timeline"]:
        raise ValueError("executed public timeline must cover every frame exactly")
    event_slot = 0

    def consume(kind, ticks, handle=None):
        nonlocal event_slot
        record = events[event_slot]
        expected_target = capture.target(
            SimpleNamespace(observe=lambda: record.state_before), kind, handle
        )
        action = PrimitiveAction(
            kind=ActionKind(kind), ticks=ticks, target_id=expected_target
        )
        if record.executed_action != action:
            raise ValueError(
                "public executed action differs from causal requested handle/target/ticks"
            )
        event_slot += 1
        return timeline[event_slot - 1]

    references = []
    for scheduled in capture.motion_probe_schedule(config)[:25]:
        interval = consume(
            scheduled.kind, scheduled.ticks, scheduled.motion_handle_index
        )
        if scheduled.phase == "reference_settle":
            references.append(interval["end_frame"])
    reference_names = {f"view-{number:02d}.png" for number in range(12)}
    if (
        raw["reference_indices"] != references
        or {path.name for path in (route / "goal-reference").glob("*.png")}
        != reference_names
        or any(
            canonical.digest(route / "goal-reference" / f"view-{number:02d}.png")
            != raw["frame_sha256"][f"frame-{slot:05d}.png"]
            for number, slot in enumerate(references)
        )
    ):
        raise ValueError(
            "camera reference frames must be the exact declared startup views"
        )
    if len(raw["trials"]) != 2 * len(config.handle_pairs):
        raise ValueError("adaptive acquisition cannot drop failed or aborted trials")
    reconstructed = []
    for number, trial in enumerate(raw["trials"]):
        pair = config.handle_pairs[number % len(config.handle_pairs)]
        controller = VisibilityController(pair, policy=policy)
        anchor = timeline[event_slot]["start_frame"]
        prefixes, steps = [], []
        for _ in range(policy.maximum_first_ticks + policy.maximum_second_ticks):
            handle = controller.next_handle
            interval = consume("move", 1, handle)
            steps.append(
                {
                    "requested_handle": handle,
                    "executed_handle": handle,
                    "ticks": 1,
                    "compliant": True,
                }
            )
            tracked = track_frame_chain(
                images[anchor : interval["end_frame"] + 1], config=policy.temporal
            )
            decision = controller.observe(tracked)
            prefixes.append(tracked.model_dump(mode="json"))
            if decision.instruction in ("finish", "abort"):
                break
        else:
            raise ValueError("adaptive trial exceeds its action budget")
        replayed = {
            "repetition": number // len(config.handle_pairs),
            "handles": list(pair),
            "anchor_frame": anchor,
            "end_frame": interval["end_frame"],
            "terminal": "finished" if decision.instruction == "finish" else "aborted",
            "failure": None,
            "steps": steps,
            "decisions": [
                item.model_dump(mode="json") for item in controller.decisions
            ],
            "prefix_tracks": prefixes,
        }
        if not _numeric_equal(replayed, trial):
            raise ValueError(
                "causal prefix tracking, decisions or receipts differ from independent replay"
            )
        reconstructed.append(replayed)
        consume("wait", 8)
    consume("wait", 8)
    if event_slot != len(events) or not all(
        canonical.digest(path) == value for path, value in bound.items()
    ):
        raise ValueError(
            "trailing/omitted controls or changed public bytes during replay"
        )
    checks = {
        name: True
        for name in (
            "raw_declarations_and_gates",
            "parent_public_plan_replayed",
            "current_project_capture_sources",
            "complete_capture_snapshots",
            "complete_frame_and_rgb_ledger",
            "public_log_clean",
            "all_actions_and_frame_alignment",
            "complete_reference_views",
            "complete_trials",
            "all_causal_prefixes_and_receipts_replayed",
            "no_lost_identity_resurrection",
            "all_public_bytes_stable",
        )
    }
    return config, raw, reconstructed, checks


def pairs(raw):
    result = []
    for number, trial in enumerate(raw["trials"]):
        first, end = trial["anchor_frame"], trial["end_frame"]
        switch = next(
            (item for item in trial["decisions"] if item["instruction"] == "switch"),
            None,
        )
        split = None if switch is None else first + switch["tick"]
        for group, begin, finish in (
            ("compound_moving", first, end),
            ("compound_settled", first, end + 8),
            ("leg_one", first, split),
            ("leg_two", split, end),
            ("settle_control", end + 4, end + 8),
        ):
            result.append(
                {
                    "id": f"{group}-{number:02d}",
                    "group": group,
                    "first": begin,
                    "second": finish,
                }
            )
    startup = raw["timeline"][:25]
    for number, interval in enumerate(
        [item for item in startup if item["action"]["kind"] == "look"][:2]
    ):
        result.append(
            {
                "id": f"rotation-{number:02d}",
                "group": "rotation_control",
                "first": interval["start_frame"],
                "second": interval["end_frame"],
            }
        )
    for number, interval in enumerate((raw["timeline"][0], raw["timeline"][-1])):
        result.append(
            {
                "id": f"wait-{number:02d}",
                "group": "wait_control",
                "first": interval["end_frame"] - 4,
                "second": interval["end_frame"],
            }
        )
    for slot in (startup[-1]["end_frame"], raw["frame_count"] - 1):
        result.append(
            {
                "id": f"identity-{slot:03d}",
                "group": "identity_control",
                "first": slot,
                "second": slot,
            }
        )
    return result


def configuration(route, acquisition, raw, seed):
    baseline = canonical.inference_config(
        route, acquisition, raw | {"pairs": pairs(raw)}, seed
    )
    return baseline | {
        "schema": "adaptive-visibility-frozen-maps-v1",
        "keyframe": KeyframeMotionConfig().model_dump(mode="json"),
        "action_switch": ActionSwitchConfig().model_dump(mode="json"),
        "scope": "all adaptive trial/control prefixes; leg-one maps frozen before second-control pixels; no pose replacement or navigation; single development world, unmatched trajectories",
    }


def infer(route, raw, config, *, results=None):
    results = [] if results is None else results
    camera = np.asarray(config["camera_intrinsics"])
    keyframe = KeyframeMotionConfig.model_validate(config["keyframe"])
    future_policy = ActionSwitchConfig.model_validate(config["action_switch"])
    for index, pair in enumerate(config["pairs"]):
        first, second = pair["first"], pair["second"]
        selection, evidence = None, None
        if first is not None and second is not None:
            indices = (first, first) if first == second else range(first, second + 1)
            images = [
                canonical.rgb(route / "frames" / f"frame-{slot:05d}.png")
                for slot in indices
            ]
            # Identity controls use one actual image and infer's own repeat, so
            # their terminal offset is zero, not a synthetic one-tick movement.
            selection = infer_keyframe_prefixes(
                images[:1] if first == second else images,
                camera,
                config=keyframe,
                seed=config["seed"] + index,
            )
        if pair["group"] == "leg_one":
            trial = raw["trials"][int(pair["id"].split("-")[-1])]
            selected = None if selection is None else selection.selected()
            last = trial["decisions"][-1]
            available = selected is not None and trial["terminal"] == "finished"
            targets = tuple(
                trial["anchor_frame"] + tick for tick in last["holdout_ticks"]
            )
            plan = ActionSwitchPlan(
                anchor_frame=trial["anchor_frame"],
                candidate_end_frame=None
                if selected is None
                else first + selected.end_offset,
                available=available,
                reason="actual causal controller first/end second-leg views; physical diversity unknown"
                if available
                else "no frozen candidate or incomplete acquisition",
                current_handle=trial["handles"][0],
                next_handle=trial["handles"][1],
                action_start_frame=second,
                action_end_frame=trial["end_frame"],
                target_frames=targets if available else (),
                requested_frames=targets,
            )
            prefit = inspect_direction_search(
                selected,
                camera,
                seed=config["seed"] + index,
                config=future_policy.search,
            )
            extended = None
            if available and prefit.reference_matches_original_checks:
                images = [
                    canonical.rgb(route / "frames" / f"frame-{slot:05d}.png")
                    for slot in range(first, trial["end_frame"] + 1)
                ]
                tracked = track_frame_chain(images, config=future_policy.temporal)
                # Do not replenish identities rejected earlier by online control.
                keep = [
                    slot
                    for slot, identity in enumerate(tracked.track_ids)
                    if identity in last["retained_ids"]
                ]
                extended = TemporalTracks.model_validate(
                    tracked.model_dump()
                    | {
                        "track_ids": tuple(tracked.track_ids[slot] for slot in keep),
                        "pixels": tuple(
                            tuple(frame[slot] for slot in keep)
                            for frame in tracked.pixels
                        ),
                        "minimum_patch_correlations": tuple(
                            tracked.minimum_patch_correlations[slot] for slot in keep
                        ),
                        "chain_cycle_errors": tuple(
                            tracked.chain_cycle_errors[slot] for slot in keep
                        ),
                    }
                )
            evidence = inspect_action_switch(
                selected,
                extended,
                plan,
                camera,
                seed=config["seed"] + index,
                config=future_policy,
                frozen_search=prefit,
            )
        results.append(
            {
                "pair": pair,
                "selection": None
                if selection is None
                else selection.model_dump(mode="json"),
                "action_switch": None
                if evidence is None
                else evidence.model_dump(mode="json"),
            }
        )
    return results

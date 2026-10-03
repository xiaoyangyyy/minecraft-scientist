import hashlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image", reason="optional public RGB pipeline")

from minecraft_scientist.exploration.compound_motion import CompoundProbeConfig
from minecraft_scientist.exploration.keyframe_motion import (
    KeyframeAttempt,
    KeyframeMotionConfig,
    KeyframeSelection,
)
from minecraft_scientist.exploration.visibility_acquisition import (
    VisibilityController,
)
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def adaptive_capture(tmp_path, monkeypatch):
    from test_keyframe_motion import _motion, _tracks

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    common = importlib.import_module("visibility_capture_common")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    config = CompoundProbeConfig(seed=201, observation_size=224)
    policy = common.capture.online_policy()
    camera = {"camera_intrinsics": [[180, 0, 112], [0, 180, 112], [0, 0, 1]]}
    monkeypatch.setattr(
        common.capture, "calibrate_reference_camera", lambda images, size: camera
    )
    route, plan = tmp_path / "runs/capture", tmp_path / "runs/plan"
    parent = {"plan_config": plan / "config.json", "plan_result": plan / "plan.json"}
    for path in parent.values():
        save(path, {"fixture": True})
    monkeypatch.setattr(common, "parent_inputs", lambda route: parent)
    monkeypatch.setattr(common.capture.planner, "validate_plan", lambda path: config)
    monkeypatch.setattr(
        common, "track_frame_chain", lambda frames, config: _tracks(0, len(frames))
    )
    runtime = common.capture_sources()
    hashes = {name: common.canonical.digest(path) for name, path in runtime.items()}
    hashes["sdk-entry.py"] = hashlib.sha256(b"fixture SDK provenance\n").hexdigest()
    for name, path in runtime.items():
        target = route / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (route / "source-snapshot/sdk-entry.py").write_bytes(b"fixture SDK provenance\n")
    manifest = {
        "schema": "visibility-rgb-acquisition-v2",
        "policy": policy.model_dump(mode="json"),
        "online_landmarks": common.capture.online_settings(),
        "world": {
            "seed": 201,
            "observation_size": 224,
            "render_size": [640, 360],
            "time": "noon",
            "weather": "clear",
        },
        "startup": [
            item.model_dump(mode="json")
            for item in common.capture.motion_probe_schedule(config)[:25]
        ],
        "handle_pairs": config.model_dump(mode="json")["handle_pairs"],
        "repetitions": 2,
        "plan_directory": "runs/plan",
        "maximum_frames": 209,
        "input_sha256": {
            name: common.canonical.digest(path) for name, path in parent.items()
        },
        "source_sha256": hashes,
        "scope": "fresh visibility-aware RGB probe on development seed; no hidden telemetry in policy, no terrain edits, no pose validation or navigation promotion",
    }
    save(route / "config.json", manifest)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    png = stream.getvalue()
    file_hash = hashlib.sha256(png).hexdigest()
    objects = tuple(
        ObjectObservation(object_id=f"opaque-{index}", affordances=(ActionKind.MOVE,))
        for index in range(4)
    ) + (
        ObjectObservation(
            object_id="opaque-camera",
            affordances=(ActionKind.LOOK,),
            observable_features={"screen_row": 1, "screen_column": 2},
        ),
    )
    state = Observation(
        episode_id=0,
        step=0,
        inventory={},
        agent_state={"frame_sha256": rgb_hash},
        objects=objects,
    )
    events, timeline, trials, references = [], [], [], []
    cursor = 0

    def action(kind, ticks, handle=None):
        nonlocal cursor, state
        target = (
            f"opaque-{handle}"
            if kind == "move"
            else "opaque-camera"
            if kind == "look"
            else None
        )
        executed = PrimitiveAction(kind=ActionKind(kind), target_id=target, ticks=ticks)
        event = WorldEvent(event_type="fixture", target_id=target, duration=ticks)
        after = state.model_copy(update={"step": state.step + 1, "last_event": event})
        events.append(
            TransitionRecord(
                transition_id=f"fixture-{len(events)}",
                context={},
                state_before=state,
                executed_action=executed,
                intervention_compliance=True,
                state_after=after,
                observed_delta={},
                event=event,
            )
        )
        timeline.append(
            {
                "start_frame": cursor,
                "end_frame": cursor + ticks,
                "action": executed.model_dump(mode="json"),
            }
        )
        cursor += ticks
        state = after

    for scheduled in common.capture.motion_probe_schedule(config)[:25]:
        action(scheduled.kind, scheduled.ticks, scheduled.motion_handle_index)
        if scheduled.phase == "reference_settle":
            references.append(cursor)
    for repetition in range(2):
        for pair in config.handle_pairs:
            anchor = cursor
            controller = VisibilityController(pair, policy=policy)
            tracked = _tracks(0, 2)
            decision = controller.observe(tracked)
            action("move", 1, pair[0])
            trials.append(
                {
                    "repetition": repetition,
                    "handles": list(pair),
                    "anchor_frame": anchor,
                    "end_frame": cursor,
                    "terminal": "aborted",
                    "failure": None,
                    "landmark_seed": 5 * len(trials) + 2,
                    "landmark_attempts": [],
                    "steps": [
                        {
                            "requested_handle": pair[0],
                            "executed_handle": pair[0],
                            "ticks": 1,
                            "compliant": True,
                        }
                    ],
                    "decisions": [decision.model_dump(mode="json")],
                    "prefix_tracks": [tracked.model_dump(mode="json")],
                }
            )
            action("wait", 8)
    action("wait", 8)
    (route / "frames").mkdir()
    for slot in range(cursor + 1):
        (route / "frames" / f"frame-{slot:05d}.png").write_bytes(png)
    (route / "goal-reference").mkdir()
    for number in range(12):
        (route / "goal-reference" / f"view-{number:02d}.png").write_bytes(png)
    (route / "events.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in events), encoding="utf-8"
    )
    raw = {
        "passed": True,
        "checks": dict.fromkeys(common.RAW_CHECKS, True),
        "config_sha256": common.canonical.digest(route / "config.json"),
        "failure": None,
        "frame_count": cursor + 1,
        "frame_sha256": {
            f"frame-{slot:05d}.png": file_hash for slot in range(cursor + 1)
        },
        "rgb_sha256": [rgb_hash] * (cursor + 1),
        "reference_indices": references,
        "landmark_calibration": camera,
        "timeline": timeline,
        "trials": trials,
        "navigation_accepted": False,
    }
    save(route / "report.json", raw)
    save(
        route / "evaluator-audit.json",
        {
            "positions": [{"xpos": 0, "ypos": 0, "zpos": 0, "yaw": 0, "pitch": 0}]
            * (cursor + 1)
        },
    )

    def calibration(route, acquisition, report, seed):
        return {
            "seed": seed,
            "pairs": report["pairs"],
            "camera_intrinsics": [[180, 0, 112], [0, 180, 112], [0, 0, 1]],
            "evaluation_thresholds": {
                "minimum_translation_blocks": 0.15,
                "maximum_direction_error_degrees": 20,
                "maximum_rotation_error_degrees": 5,
            },
        }

    monkeypatch.setattr(common.canonical, "inference_config", calibration)

    def selection(frames, camera, *, config, seed):
        terminal = len(frames) - 1
        attempts = tuple(
            KeyframeAttempt(
                end_offset=offset,
                tracks=_tracks(0, max(2, offset + 1)),
                motion=_motion(0, supported=False),
            )
            for offset in KeyframeMotionConfig().offsets(terminal)
        )
        return KeyframeSelection(terminal_offset=terminal, attempts=attempts)

    monkeypatch.setattr(common, "infer_keyframe_prefixes", selection)
    return common, route


@pytest.fixture
def inference(adaptive_capture):
    common, route = adaptive_capture
    output = route.parent / "predictions"
    result = importlib.import_module("probe_visibility_maps").run(route, output)
    assert result["completed"] and result["input_sources_unchanged"]
    return common, route, output


def test_public_capture_and_map_inference_never_reads_positions(
    adaptive_capture, monkeypatch
):
    _common, route = adaptive_capture
    reading = Path.read_bytes

    def public(path):
        assert path.name != "evaluator-audit.json"
        return reading(path)

    monkeypatch.setattr(Path, "read_bytes", public)
    result = importlib.import_module("probe_visibility_maps").run(
        route, route.parent / "public"
    )
    assert result["completed"] and len(result["results"]) == 46
    assert not result["navigation_accepted"]
    assert len([item for item in result["results"] if item["selection"] is None]) == 16


def test_all_public_controls_and_maps_replay_before_private_read(
    inference, monkeypatch
):
    common, route, output = inference
    reading, infer = Path.read_bytes, common.infer
    completed = []

    def replay(*args, **kwargs):
        results = infer(*args, **kwargs)
        completed.extend(results)
        return results

    def guarded(path):
        if path.name == "evaluator-audit.json":
            assert len(completed) == 46
        return reading(path)

    monkeypatch.setattr(common, "infer", replay)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = importlib.import_module("audit_visibility_maps").audit(route, output)
    assert result["valid"] and all(result["checks"].values())
    assert not result["evaluation"]["navigation_accepted"]
    assert all(
        result["evaluation"]["groups"][group]["trials"] == 8
        for group in (
            "leg_one",
            "leg_two",
            "compound_moving",
            "compound_settled",
            "settle_control",
        )
    )


@pytest.mark.parametrize(
    "change",
    [
        "decision",
        "drop_trial",
        "wrong_handle",
        "failed_control",
        "drop_frame",
        "snapshot",
        "landmark_seed",
        "landmark_calibration",
        "missing_landmark_attempts",
    ],
)
def test_raw_acquisition_tampering_rejects_before_private_read(
    adaptive_capture, monkeypatch, change
):
    common, route = adaptive_capture
    raw = json.loads((route / "report.json").read_bytes())
    if change == "decision":
        raw["trials"][0]["decisions"][0]["reason"] = "forged"
    elif change == "drop_trial":
        raw["trials"].pop()
    elif change == "wrong_handle":
        raw["trials"][0]["steps"][0]["requested_handle"] = 99
    elif change == "failed_control":
        text = (route / "events.jsonl").read_text(encoding="utf-8")
        (route / "events.jsonl").write_text(
            text.replace(
                '"intervention_compliance":true', '"intervention_compliance":false', 1
            ),
            encoding="utf-8",
        )
    elif change == "drop_frame":
        (route / "frames/frame-00003.png").unlink()
    elif change == "landmark_seed":
        raw["trials"][0]["landmark_seed"] += 1
    elif change == "landmark_calibration":
        raw["landmark_calibration"]["camera_intrinsics"][0][0] += 1
    elif change == "missing_landmark_attempts":
        del raw["trials"][0]["landmark_attempts"]
    else:
        (route / "source-snapshot/visibility_acquisition.py").write_bytes(b"forged\n")
    save(route / "report.json", raw)
    reading = Path.read_bytes

    def public(path):
        assert path.name != "evaluator-audit.json"
        return reading(path)

    monkeypatch.setattr(Path, "read_bytes", public)
    with pytest.raises((ValueError, OSError)):
        common.validate_capture(route)


@pytest.mark.parametrize(
    "change", ["configuration", "prediction", "drop_pair", "snapshot"]
)
def test_prediction_tampering_prevents_private_scoring(inference, monkeypatch, change):
    _common, route, output = inference
    if change == "configuration":
        config = json.loads((output / "config.json").read_bytes())
        config["keyframe"]["candidate_offsets"] = [2]
        save(output / "config.json", config)
    elif change == "snapshot":
        (output / "source-snapshot/visibility_capture_common.py").write_bytes(
            b"forged\n"
        )
    else:
        report = json.loads((output / "report.json").read_bytes())
        if change == "drop_pair":
            report["results"].pop()
        else:
            report["results"][0]["selection"]["attempts"][0]["motion"]["reason"] = (
                "forged"
            )
        save(output / "report.json", report)
    reading = Path.read_bytes

    def guarded(path):
        assert path.name != "evaluator-audit.json"
        return reading(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = importlib.import_module("audit_visibility_maps").audit(route, output)
    assert not result["valid"] and result["evaluation"] is None


def test_partial_failure_preserves_completed_pairs_and_overwrite_refuses(
    adaptive_capture, monkeypatch
):
    common, route = adaptive_capture
    inference = common.infer

    def partial(route, raw, config, *, results):
        saved = inference(route, raw, config)
        results.extend(saved[:2])
        raise RuntimeError("interrupted fit")

    monkeypatch.setattr(common, "infer", partial)
    output = route.parent / "partial"
    producer = importlib.import_module("probe_visibility_maps")
    result = producer.run(route, output)
    assert not result["completed"] and len(result["results"]) == 2
    assert result["failure"]["type"] == "RuntimeError"
    monkeypatch.setattr(
        common, "inputs", lambda *a: pytest.fail("read inputs before overwrite refusal")
    )
    with pytest.raises(FileExistsError):
        producer.run(route, output)


def test_pre_switch_search_is_frozen_before_future_pixels_and_lost_ids_stay_lost(
    adaptive_capture, monkeypatch
):
    from types import SimpleNamespace

    from test_keyframe_motion import _motion, _tracks

    common, route = adaptive_capture
    frozen, seen = [], []

    def rgb(path):
        slot = int(path.stem.split("-")[-1])
        if slot > 4:
            assert frozen, "new-control image reached prefit search"
        seen.append(slot)
        return np.full((224, 224, 3), slot, dtype=np.uint8)

    def selection(frames, camera, *, config, seed):
        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(5))
        attempts = tuple(
            KeyframeAttempt(
                end_offset=tick,
                tracks=_tracks(64, tick + 1),
                motion=_motion(64, supported=True),
            )
            for tick in (2, 4)
        )
        return KeyframeSelection(
            terminal_offset=4, attempts=attempts, selected_offset=2
        )

    prefit = SimpleNamespace(reference_matches_original_checks=True)

    def search(candidate, camera, **kwargs):
        assert candidate.end_offset == 2 and max(seen) == 4
        frozen.append(candidate.model_dump_json())
        return prefit

    def inspect(candidate, extended, plan, camera, **kwargs):
        assert candidate.model_dump_json() == frozen[0]
        assert extended.track_ids == tuple(range(50))
        assert extended.frame_count == 7 and plan.target_frames == (5, 6)
        assert kwargs["frozen_search"] is prefit
        return SimpleNamespace(model_dump=lambda **kwargs: {"fixture": "dataflow only"})

    monkeypatch.setattr(common.canonical, "rgb", rgb)
    monkeypatch.setattr(common, "infer_keyframe_prefixes", selection)
    monkeypatch.setattr(common, "inspect_direction_search", search)
    monkeypatch.setattr(common, "inspect_action_switch", inspect)
    monkeypatch.setattr(
        common, "track_frame_chain", lambda frames, **kwargs: _tracks(64, len(frames))
    )
    config = {
        "seed": 0,
        "camera_intrinsics": np.eye(3).tolist(),
        "keyframe": KeyframeMotionConfig().model_dump(mode="json"),
        "action_switch": common.ActionSwitchConfig().model_dump(mode="json"),
        "pairs": [{"id": "leg_one-00", "group": "leg_one", "first": 0, "second": 4}],
    }
    raw = {
        "trials": [
            {
                "anchor_frame": 0,
                "end_frame": 6,
                "handles": [0, 1],
                "terminal": "finished",
                "landmark_seed": 2,
                "landmark_attempts": [],
                "decisions": [
                    {"holdout_ticks": [5, 6], "retained_ids": list(range(50))}
                ],
            }
        ]
    }
    result = common.infer(route, raw, config)
    assert result[0]["selection"]["selected_offset"] == 2
    assert seen == list(range(5)) + list(range(7))


def test_online_frozen_candidate_is_not_retrospectively_refitted(
    adaptive_capture, monkeypatch
):
    from types import SimpleNamespace

    from test_keyframe_motion import _motion, _tracks

    common, route = adaptive_capture
    seen, frozen = [], []
    first = KeyframeAttempt(
        end_offset=2, tracks=_tracks(64, 3), motion=_motion(64, supported=False)
    )
    selected = KeyframeAttempt(
        end_offset=3, tracks=_tracks(64, 4), motion=_motion(64, supported=True)
    )

    def rgb(path):
        slot = int(path.stem.split("-")[-1])
        if slot > 3:
            assert frozen
        seen.append(slot)
        return np.full((224, 224, 3), slot, dtype=np.uint8)

    def search(candidate, camera, *, seed, **kwargs):
        assert candidate == selected and seed == 17 and max(seen) == 3
        frozen.append(candidate.model_dump_json())
        return SimpleNamespace(reference_matches_original_checks=True)

    def inspect(candidate, extended, plan, camera, *, seed, **kwargs):
        assert candidate == selected and seed == 17
        assert plan.target_frames == (4, 5) and extended.track_ids == tuple(range(50))
        return SimpleNamespace(
            model_dump=lambda **kwargs: {"fixture": "original online map"}
        )

    monkeypatch.setattr(common.canonical, "rgb", rgb)
    monkeypatch.setattr(
        common,
        "infer_keyframe_prefixes",
        lambda *a, **k: pytest.fail("online map replaced by retrospective fit"),
    )
    monkeypatch.setattr(common, "inspect_direction_search", search)
    monkeypatch.setattr(common, "inspect_action_switch", inspect)
    monkeypatch.setattr(
        common, "track_frame_chain", lambda frames, **kwargs: _tracks(64, len(frames))
    )
    config = {
        "seed": 90,
        "camera_intrinsics": np.eye(3).tolist(),
        "keyframe": KeyframeMotionConfig().model_dump(mode="json"),
        "action_switch": common.ActionSwitchConfig().model_dump(mode="json"),
        "pairs": [{"id": "leg_one-00", "group": "leg_one", "first": 0, "second": 3}],
    }
    raw = {
        "trials": [
            {
                "anchor_frame": 0,
                "end_frame": 5,
                "handles": [0, 1],
                "terminal": "finished",
                "landmark_seed": 17,
                "landmark_attempts": [
                    first.model_dump(mode="json"),
                    selected.model_dump(mode="json"),
                ],
                "decisions": [
                    {"holdout_ticks": [4, 5], "retained_ids": list(range(50))}
                ],
            }
        ]
    }
    result = common.infer(route, raw, config)
    assert result[0]["selection"]["selected_offset"] == 3
    assert seen == list(range(4)) + list(range(6))
    assert common.expected_offsets(config["pairs"][0], raw) == (2, 3)


@pytest.mark.parametrize("target", ["public", "private"])
def test_bytes_changed_during_scoring_invalidate_the_evaluation(
    inference, monkeypatch, target
):
    _common, route, output = inference
    auditor = importlib.import_module("audit_visibility_maps")
    score = auditor.score_relative_motion
    mutated = []

    def changing(*args, **kwargs):
        result = score(*args, **kwargs)
        if not mutated:
            path = route / (
                "events.jsonl" if target == "public" else "evaluator-audit.json"
            )
            path.write_bytes(path.read_bytes() + b"\n")
            mutated.append(path)
        return result

    monkeypatch.setattr(auditor, "score_relative_motion", changing)
    result = auditor.audit(route, output)
    assert mutated and not result["valid"] and result["evaluation"] is None


@pytest.fixture
def shadow_capture(adaptive_capture):
    common, route = adaptive_capture
    manifest = json.loads((route / "config.json").read_bytes())
    manifest.update(
        schema="visibility-rgb-acquisition-v3",
        second_leg_shadow_policy=common.SecondLegShadowPolicy().model_dump(mode="json"),
        maximum_frames=manifest["maximum_frames"] + 32,
        scope="fresh development RGB probe with second-leg maps/search frozen before bounded third-control views; physical diversity unknown, no navigation promotion",
    )
    save(route / "config.json", manifest)
    raw = json.loads((route / "report.json").read_bytes())
    raw["config_sha256"] = common.canonical.digest(route / "config.json")
    for trial in raw["trials"]:
        trial["second_leg_shadow"] = {
            "terminal": "source_aborted",
            "anchor_frame": None,
            "freeze_frame": trial["end_frame"],
            "end_frame": trial["end_frame"],
        }
    save(route / "report.json", raw)
    return common, route


def test_shadow_schema_retains_all_skipped_trials_and_replays_before_private_scoring(
    shadow_capture, monkeypatch
):
    common, route = shadow_capture
    output = route.parent / "shadow-predictions"
    result = importlib.import_module("probe_visibility_maps").run(route, output)
    assert result["completed"] and len(result["results"]) == 54
    assert all(
        item["selection"] is None
        for item in result["results"]
        if item["pair"]["group"] == "third_shadow"
    )
    reading, infer = Path.read_bytes, common.infer
    completed = []

    def replay(*args, **kwargs):
        records = infer(*args, **kwargs)
        completed.extend(records)
        return records

    def guarded(path):
        if path.name == "evaluator-audit.json":
            assert len(completed) == 54
        return reading(path)

    monkeypatch.setattr(common, "infer", replay)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    audit = importlib.import_module("audit_visibility_maps").audit(route, output)
    assert audit["valid"] and not audit["evaluation"]["navigation_accepted"]
    assert audit["evaluation"]["groups"]["third_shadow"]["trials"] == 8


@pytest.mark.parametrize(
    "change", ("omit", "invent_freeze", "end_frame", "policy", "budget")
)
def test_forged_shadow_metadata_fails_before_private_inputs(
    shadow_capture, monkeypatch, change
):
    common, route = shadow_capture
    manifest = json.loads((route / "config.json").read_bytes())
    raw = json.loads((route / "report.json").read_bytes())
    if change == "omit":
        raw["trials"][0].pop("second_leg_shadow")
    elif change == "invent_freeze":
        raw["trials"][0]["second_leg_shadow"]["freeze"] = {"invented": True}
    elif change == "end_frame":
        raw["trials"][0]["second_leg_shadow"]["end_frame"] += 1
    elif change == "policy":
        manifest["second_leg_shadow_policy"]["ticks"] = 5
    elif change == "budget":
        manifest["maximum_frames"] -= 1
    save(route / "config.json", manifest)
    raw["config_sha256"] = common.canonical.digest(route / "config.json")
    save(route / "report.json", raw)
    reading = Path.read_bytes

    def public(path):
        assert path.name != "evaluator-audit.json"
        return reading(path)

    monkeypatch.setattr(Path, "read_bytes", public)
    result = importlib.import_module("probe_visibility_maps").run(
        route, route.parent / "forged-shadow"
    )
    assert not result["completed"] and result["failure"] is not None


def test_second_leg_inference_uses_frozen_selection_partitions_and_seed(
    adaptive_capture, monkeypatch
):
    from types import SimpleNamespace

    from test_keyframe_motion import _motion, _tracks

    common, route = adaptive_capture
    attempts = tuple(
        KeyframeAttempt(
            end_offset=tick,
            tracks=_tracks(64, tick + 1),
            motion=_motion(64, supported=tick == 3),
        )
        for tick in (2, 3, 4)
    )
    selection = KeyframeSelection(
        terminal_offset=4, attempts=attempts, selected_offset=3
    )
    candidate = selection.selected()
    seen = []

    def inspect(selected, extended, plan, camera, *, seed, frozen_search, **kwargs):
        assert selected == candidate and seed == 1007
        assert plan.anchor_frame == 10 and plan.candidate_end_frame == 13
        assert plan.action_start_frame == 14 and plan.target_frames == (16, 18)
        assert extended.track_ids == tuple(range(40))
        seen.append(selected.model_dump_json())
        return SimpleNamespace(
            model_dump=lambda **kwargs: {"fixture": "original second-leg map"}
        )

    monkeypatch.setattr(common, "inspect_action_switch", inspect)
    monkeypatch.setattr(
        common,
        "infer_keyframe_prefixes",
        lambda *a, **k: pytest.fail("frozen second-leg map was refitted"),
    )
    monkeypatch.setattr(
        common.canonical, "rgb", lambda path: np.zeros((224, 224, 3), dtype=np.uint8)
    )
    search = common.inspect_direction_search(None, np.eye(3), seed=1007).model_dump(
        mode="json"
    )
    data = {
        "anchor_frame": 10,
        "freeze": {
            "selection": selection.model_dump(mode="json"),
            "seed": 1007,
            "search": search,
        },
        "prefix_tracks": [_tracks(64, 9).model_dump(mode="json")],
        "retained_ids": [list(range(40))],
        "plan": common.ActionSwitchPlan(
            anchor_frame=0,
            candidate_end_frame=3,
            available=True,
            reason="fixture",
            current_handle=1,
            next_handle=2,
            action_start_frame=4,
            action_end_frame=8,
            target_frames=(6, 8),
            requested_frames=(6, 8),
        ).model_dump(mode="json"),
    }
    raw = {"trials": [{"second_leg_shadow": data}]}
    config = {
        "seed": 999,
        "camera_intrinsics": np.eye(3).tolist(),
        "keyframe": KeyframeMotionConfig().model_dump(mode="json"),
        "action_switch": common.ActionSwitchConfig().model_dump(mode="json"),
        "pairs": [{"id": "leg_two-00", "group": "leg_two", "first": 10, "second": 14}],
    }
    result = common.infer(route, raw, config)
    assert seen == [candidate.model_dump_json()]
    assert result[0]["selection"]["selected_offset"] == 3
    assert common.expected_offsets(config["pairs"][0], raw) == (2, 3, 4)

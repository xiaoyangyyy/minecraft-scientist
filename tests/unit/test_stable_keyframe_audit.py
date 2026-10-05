import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="optional image replay")
pytest.importorskip("PIL.Image", reason="optional image replay")

from test_runtime_camera_audit import (
    diagnostic as diagnostic,  # noqa: PLC0414 -- fixture re-export
)
from test_runtime_camera_audit import (
    private_guard,
    save,
)
from test_runtime_camera_audit import (
    scripts as scripts,  # noqa: PLC0414 -- fixture re-export
)

from minecraft_scientist.protocols import ActionKind, PrimitiveAction, TransitionRecord


@pytest.fixture
def keyframe_capture(diagnostic, scripts, monkeypatch, request):
    runtime, route, engine = diagnostic
    _, definition = scripts
    auditor = importlib.import_module("audit_stable_keyframes")
    mode = getattr(request, "param", False)
    safe_seeds = mode == "safe-seeds"
    recovery = safe_seeds or mode == "recovering"
    responsive = recovery or mode == "responsive"
    micro = responsive or mode is True
    policy_type = (
        auditor.StableSafeSeedKeyframePolicy
        if safe_seeds
        else auditor.StableRecoveringKeyframePolicy
        if recovery
        else auditor.StableResponsiveKeyframePolicy
        if responsive
        else auditor.StableMicroKeyframePolicy
        if micro
        else auditor.StableKeyframePolicy
    )
    calibration = {
        "camera_intrinsics": [[180, 0, 224], [0, 320, 224], [0, 0, 1]],
        "scope": "test-only calibration",
    }
    monkeypatch.setattr(
        auditor, "calibrate_reference_camera", lambda *args: calibration
    )
    config = json.loads((route / "config.json").read_bytes())
    config.update(
        schema="safe-seed-stable-keyframe-camera-projection-v1"
        if safe_seeds
        else "recovering-stable-keyframe-camera-projection-v1"
        if recovery
        else "responsive-stable-keyframe-camera-projection-v1"
        if responsive
        else "micro-stable-keyframe-camera-projection-v1"
        if micro
        else "stable-keyframe-camera-projection-v1",
        schedule=[list(item) for item in definition.SCHEDULE[:48]],
        trial_intervals=[],
        active_seed=2000,
        keyframe_policy=policy_type().model_dump(mode="json"),
    )
    save(route / "config.json", config)
    action_count = 96 if recovery else 64
    events = [
        TransitionRecord.model_validate_json(line)
        for line in (route / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ][:action_count]
    for index in range(48, action_count):
        events[index] = events[index].model_copy(
            update={"executed_action": PrimitiveAction(kind=ActionKind.WAIT, ticks=1)}
        )
    (route / "events.jsonl").write_text(
        "".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8"
    )
    for folder in ("frames", "evaluator-native"):
        for path in (route / folder).glob("*.png"):
            if int(path.stem.split("-")[-1]) >= action_count + 1:
                path.unlink()  # only newly created pytest-owned mock data
    private = json.loads((route / "evaluator-positions.json").read_bytes())
    save(
        route / "evaluator-positions.json",
        {key: value[: action_count + 1] for key, value in private.items()},
    )
    rows = (route / "evaluator-camera.jsonl").read_text(encoding="utf-8").splitlines()
    (route / "evaluator-camera.jsonl").write_text(
        "\n".join(rows[: action_count + 3]) + "\n", encoding="utf-8"
    )
    cursor = 48

    def execute(kind, handle):
        nonlocal cursor
        assert kind == "wait" and handle is None
        cursor += 1
        return auditor.PublicControlStep(
            runtime.load_image(route / "frames" / f"frame-{cursor:05d}.png"),
            kind,
            handle,
            True,
        )

    result = auditor.acquire_stable_keyframe_map(
        runtime.load_image(route / "frames/frame-00048.png"),
        execute,
        calibration["camera_intrinsics"],
        anchor_frame=48,
        micro_controls=micro,
        responsive_controls=responsive,
        recovery_controls=recovery,
        safe_seed_controls=safe_seeds,
    )
    assert result["terminal"] == "initial_wait_exhausted"
    save(
        route / "keyframe-acquisition.json",
        {"calibration": calibration, "acquisition": result},
    )
    report = json.loads((route / "report.json").read_bytes())
    report.update(
        frame_count=action_count + 1,
        config_sha256=runtime.digest(route / "config.json"),
    )
    report["checks"]["keyframe_policy_completed"] = True
    report["artifact_sha256"] = {
        name: runtime.digest(route / name)
        for name in report["artifact_sha256"]
        if (route / name).exists()
    }
    report["artifact_sha256"]["keyframe-acquisition.json"] = runtime.digest(
        route / "keyframe-acquisition.json"
    )
    save(route / "report.json", report)
    return auditor, route, engine


def test_nonempty_public_replay_precedes_any_private_reads(
    keyframe_capture, monkeypatch
):
    auditor, route, engine = keyframe_capture
    with monkeypatch.context() as guards:
        private_guard(guards)
        result = auditor.runtime.public_data(
            route, engine, registration=auditor.registration
        )
        assert result["valid"] and result["count"] == 65
    result = auditor.audit(route, engine)
    assert result["valid"] and all(result["checks"].values())
    assert result["evaluation"] is None and result["selected_keyframes"] == 0
    assert result["policy_terminal"] == "initial_wait_exhausted"


@pytest.mark.parametrize(
    "case", ["policy", "schema", "metric", "motion", "clock", "stop"]
)
def test_forged_keyframe_claim_cannot_reach_private_evaluator(
    keyframe_capture, monkeypatch, case
):
    auditor, route, engine = keyframe_capture
    config = json.loads((route / "config.json").read_bytes())
    saved = json.loads((route / "keyframe-acquisition.json").read_bytes())
    if case == "policy":
        config["keyframe_policy"]["novelty_median_pixels"] = 0.1
    elif case == "schema":
        config["schema"] = "shared-endpoint-camera-projection-v1"
    elif case == "metric":
        saved["acquisition"]["metric_scale_known"] = True
    elif case == "motion":
        saved["acquisition"]["motion"] = {"unit_translation": [1, 0, 0]}
    elif case == "clock":
        saved["acquisition"]["samples"] = [{"selected": True, "absolute_frame": 49}]
    else:
        saved["acquisition"]["terminal"] = "validated"
    save(route / "config.json", config)
    save(route / "keyframe-acquisition.json", saved)
    report = json.loads((route / "report.json").read_bytes())
    report["config_sha256"] = auditor.runtime.digest(route / "config.json")
    report["artifact_sha256"]["keyframe-acquisition.json"] = auditor.runtime.digest(
        route / "keyframe-acquisition.json"
    )
    save(route / "report.json", report)
    private_guard(monkeypatch)
    result = auditor.audit(route, engine)
    assert not result["valid"] and result["evaluation"] is None


@pytest.mark.parametrize(
    "name", ["keyframe-acquisition.json", "evaluator-positions.json", "report.json"]
)
def test_late_mutation_discards_every_score(keyframe_capture, monkeypatch, name):
    auditor, route, engine = keyframe_capture
    audit = auditor.runtime.audit

    def mutate(*args, **kwargs):
        result = audit(*args, **kwargs)
        path = route / name
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(auditor.runtime, "audit", mutate)
    result = auditor.audit(route, engine)
    assert not result["valid"] and result["evaluation"] is None
    assert result["stability_diagnostics"] is None


def test_capture_flag_is_exclusive_before_creating_output(
    scripts, tmp_path, monkeypatch
):
    _, capture = scripts
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            capture.__file__,
            "--stable-keyframes",
            "--active-causal",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        capture.main()
    assert not output.exists()


def test_cli_keeps_source_and_refuses_overwrite(scripts, tmp_path, monkeypatch):
    auditor = importlib.import_module("audit_stable_keyframes")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(auditor, "audit", lambda *args: {"valid": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            auditor.__file__,
            str(tmp_path),
            "--engine-jar",
            str(tmp_path / "mock.jar"),
            "--output",
            str(output),
        ],
    )
    auditor.main()
    assert (
        output.with_suffix(".source.py").read_bytes()
        == Path(auditor.__file__).read_bytes()
    )
    with pytest.raises(FileExistsError):
        auditor.main()


def test_nonempty_stationary_and_moving_diagnostics_are_json_serializable(scripts):
    auditor = importlib.import_module("audit_stable_keyframes")
    for direction, distance in ((None, 0.0), (np.array((1.0, 0.0, 0.0)), 0.4)):
        truth = {
            "relative_rotation": np.eye(3),
            "unit_translation": direction,
            "distance": distance,
        }
        result = json.loads(
            json.dumps(auditor.serializable_truth(truth), allow_nan=False)
        )
        assert result["distance"] == distance
        assert result["relative_rotation"] == np.eye(3).tolist()
        assert result["unit_translation"] == (
            None if direction is None else direction.tolist()
        )
    with pytest.raises(ValueError, match="nonfinite"):
        auditor.serializable_truth(
            {
                "relative_rotation": np.eye(3),
                "unit_translation": None,
                "distance": float("nan"),
            }
        )


@pytest.mark.parametrize("keyframe_capture", [True], indirect=True)
def test_micro_registration_replays_before_private_reads(keyframe_capture, monkeypatch):
    auditor, route, engine = keyframe_capture
    with monkeypatch.context() as guards:
        private_guard(guards)
        result = auditor.runtime.public_data(
            route, engine, registration=auditor.registration
        )
        assert result["valid"] and result["count"] == 65
    result = auditor.audit(route, engine)
    assert result["valid"] and all(result["checks"].values())
    assert result["policy_terminal"] == "initial_wait_exhausted"
    assert not result["navigation_accepted"]


@pytest.mark.parametrize("keyframe_capture", [True], indirect=True)
@pytest.mark.parametrize("case", ["schema", "bool_tick", "future_tick", "marker"])
def test_micro_disguise_rejects_before_private_reads(
    keyframe_capture, monkeypatch, case
):
    auditor, route, engine = keyframe_capture
    config = json.loads((route / "config.json").read_bytes())
    saved = json.loads((route / "keyframe-acquisition.json").read_bytes())
    if case == "schema":
        config["schema"] = "stable-keyframe-camera-projection-v1"
    elif case == "bool_tick":
        config["keyframe_policy"]["move_ticks"] = True
    elif case == "future_tick":
        config["keyframe_policy"]["validation_move_ticks"] = 1
    else:
        saved["acquisition"].pop("micro_control_registration")
    save(route / "config.json", config)
    save(route / "keyframe-acquisition.json", saved)
    report = json.loads((route / "report.json").read_bytes())
    report["config_sha256"] = auditor.runtime.digest(route / "config.json")
    report["artifact_sha256"]["keyframe-acquisition.json"] = auditor.runtime.digest(
        route / "keyframe-acquisition.json"
    )
    save(route / "report.json", report)
    private_guard(monkeypatch)
    result = auditor.audit(route, engine)
    assert not result["valid"] and result["evaluation"] is None


def test_micro_flag_requires_stable_protocol_before_output(
    scripts, tmp_path, monkeypatch
):
    _, capture = scripts
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys, "argv", [capture.__file__, "--micro-keyframes", "--output", str(output)]
    )
    with pytest.raises(SystemExit):
        capture.main()
    assert not output.exists()


@pytest.mark.parametrize("keyframe_capture", ["responsive"], indirect=True)
def test_response_registration_replays_before_private_reads(
    keyframe_capture, monkeypatch
):
    auditor, route, engine = keyframe_capture
    with monkeypatch.context() as guards:
        private_guard(guards)
        result = auditor.runtime.public_data(
            route, engine, registration=auditor.registration
        )
        assert result["valid"] and result["count"] == 65
    result = auditor.audit(route, engine)
    assert result["valid"] and all(result["checks"].values())
    assert result["policy_terminal"] == "initial_wait_exhausted"
    assert not result["navigation_accepted"]


@pytest.mark.parametrize("keyframe_capture", ["responsive"], indirect=True)
@pytest.mark.parametrize(
    "case", ["schema", "selector", "marker", "choice", "future_choice"]
)
def test_forged_response_history_rejects_before_private_reads(
    keyframe_capture, monkeypatch, case
):
    auditor, route, engine = keyframe_capture
    config = json.loads((route / "config.json").read_bytes())
    saved = json.loads((route / "keyframe-acquisition.json").read_bytes())
    if case == "schema":
        config["schema"] = "micro-stable-keyframe-camera-projection-v1"
    elif case == "selector":
        config["keyframe_policy"]["action_selection"] = "private-heading"
    elif case == "marker":
        saved["acquisition"].pop("response_control_registration")
    elif case == "choice":
        saved["acquisition"]["control_choices"] = [{"handle": 2, "observed_trials": 0}]
    else:
        saved["acquisition"]["future_control_choice"] = {"handle": 0}
    save(route / "config.json", config)
    save(route / "keyframe-acquisition.json", saved)
    report = json.loads((route / "report.json").read_bytes())
    report["config_sha256"] = auditor.runtime.digest(route / "config.json")
    report["artifact_sha256"]["keyframe-acquisition.json"] = auditor.runtime.digest(
        route / "keyframe-acquisition.json"
    )
    save(route / "report.json", report)
    private_guard(monkeypatch)
    result = auditor.audit(route, engine)
    assert not result["valid"] and result["evaluation"] is None


@pytest.mark.parametrize(
    "flags",
    [["--responsive-keyframes"], ["--responsive-keyframes", "--stable-keyframes"]],
)
def test_response_flag_requires_both_explicit_protocol_flags(
    scripts, tmp_path, monkeypatch, flags
):
    _, capture = scripts
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys, "argv", [capture.__file__, *flags, "--output", str(output)]
    )
    with pytest.raises(SystemExit):
        capture.main()
    assert not output.exists()


@pytest.mark.parametrize("keyframe_capture", ["recovering"], indirect=True)
def test_recovery_ledger_independently_replays_before_private_reads(
    keyframe_capture, monkeypatch
):
    auditor, route, engine = keyframe_capture
    with monkeypatch.context() as guards:
        private_guard(guards)
        result = auditor.runtime.public_data(
            route, engine, registration=auditor.registration
        )
        assert result["valid"] and result["count"] == 97
    result = auditor.audit(route, engine)
    assert result["valid"] and all(result["checks"].values())
    assert result["policy_terminal"] == "initial_wait_exhausted"
    assert result["evaluation"] is None and not result["navigation_accepted"]


@pytest.mark.parametrize("keyframe_capture", ["recovering"], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "window_budget",
        "epoch_budget",
        "trial_budget",
        "drop_failure",
        "window_clock",
        "window_status",
        "parent_hash",
        "namespace",
        "scoped_history",
    ],
)
def test_forged_recovery_or_epoch_ledger_cannot_reach_private_evaluator(
    keyframe_capture, monkeypatch, case
):
    auditor, route, engine = keyframe_capture
    config = json.loads((route / "config.json").read_bytes())
    saved = json.loads((route / "keyframe-acquisition.json").read_bytes())
    acquisition = saved["acquisition"]
    if case == "schema":
        config["schema"] = "responsive-stable-keyframe-camera-projection-v1"
    elif case in ("window_budget", "epoch_budget", "trial_budget"):
        field = {
            "window_budget": "maximum_recovery_windows",
            "epoch_budget": "maximum_local_epochs",
            "trial_budget": "maximum_total_acquisition_trials",
        }[case]
        config["keyframe_policy"][field] += 1
    elif case == "drop_failure":
        acquisition["failed_wait_windows"].pop(0)
    elif case == "window_clock":
        acquisition["recovery_wait_windows"][0]["interval"][0] += 1
    elif case == "window_status":
        acquisition["recovery_wait_windows"][0]["status"] = "stable"
    elif case == "parent_hash":
        acquisition["epoch_links"] = [{"parent_epoch": 0, "parent_sha256": "0" * 64}]
    elif case == "namespace":
        acquisition["epochs"] = [{"namespace": "invented-stitched-map"}]
    else:
        acquisition["control_choices"] = [
            {"handle": 0, "epoch_id": 1, "scoring_trial_ids": [0]}
        ]
    save(route / "config.json", config)
    save(route / "keyframe-acquisition.json", saved)
    report = json.loads((route / "report.json").read_bytes())
    report["config_sha256"] = auditor.runtime.digest(route / "config.json")
    report["artifact_sha256"]["keyframe-acquisition.json"] = auditor.runtime.digest(
        route / "keyframe-acquisition.json"
    )
    save(route / "report.json", report)
    private_guard(monkeypatch)
    result = auditor.audit(route, engine)
    assert not result["valid"] and result["evaluation"] is None
    assert result["stability_diagnostics"] is None


@pytest.mark.parametrize(
    "flags",
    [
        ["--recovering-keyframes"],
        ["--recovering-keyframes", "--stable-keyframes"],
        ["--recovering-keyframes", "--stable-keyframes", "--micro-keyframes"],
        ["--recovering-keyframes", "--responsive-keyframes", "--micro-keyframes"],
    ],
)
def test_recovery_flags_require_full_explicit_protocol_before_output(
    scripts, tmp_path, monkeypatch, flags
):
    _, capture = scripts
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys, "argv", [capture.__file__, *flags, "--output", str(output)]
    )
    with pytest.raises(SystemExit):
        capture.main()
    assert not output.exists()


@pytest.mark.parametrize("keyframe_capture", ["safe-seeds"], indirect=True)
def test_safe_seed_registration_replays_full_detector_ledger_before_private_reads(
    keyframe_capture, monkeypatch
):
    auditor, route, engine = keyframe_capture
    with monkeypatch.context() as guards:
        private_guard(guards)
        public = auditor.runtime.public_data(
            route, engine, registration=auditor.registration
        )
        assert public["valid"] and public["count"] == 97
    scored = auditor.audit(route, engine)
    assert scored["valid"] and all(scored["checks"].values())
    assert scored["policy_terminal"] == "initial_wait_exhausted"
    assert scored["evaluation"] is None and not scored["navigation_accepted"]


@pytest.mark.parametrize("keyframe_capture", ["safe-seeds"], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "detector",
        "gates",
        "marker",
        "seed_pixels",
        "seed_image_hash",
        "namespace",
        "future_seeds",
    ],
)
def test_forged_safe_detector_data_cannot_reach_private_scoring(
    keyframe_capture, monkeypatch, case
):
    auditor, route, engine = keyframe_capture
    config = json.loads((route / "config.json").read_bytes())
    saved = json.loads((route / "keyframe-acquisition.json").read_bytes())
    acquisition = saved["acquisition"]
    if case == "schema":
        config["schema"] = "recovering-stable-keyframe-camera-projection-v1"
    elif case == "detector":
        config["keyframe_policy"]["seed_detection"] = "future-surviving-pixels"
    elif case == "gates":
        config["keyframe_policy"]["temporal"]["minimum_patch_correlation"] = 0.5
    elif case == "marker":
        acquisition.pop("seed_detection_registration")
    elif case == "seed_pixels":
        acquisition["flow_prefixes"][0]["seed_pixels"] = [[224, 224]]
    elif case == "seed_image_hash":
        acquisition["flow_prefixes"][0]["seed_image_sha256"] = "0" * 64
    elif case == "namespace":
        acquisition["epochs"] = [{"namespace": "sparse-keyframes-56", "detector": {}}]
    else:
        acquisition["frozen"] = {"detector": {"seed_image_frame": 96}}
    save(route / "config.json", config)
    save(route / "keyframe-acquisition.json", saved)
    report = json.loads((route / "report.json").read_bytes())
    report["config_sha256"] = auditor.runtime.digest(route / "config.json")
    report["artifact_sha256"]["keyframe-acquisition.json"] = auditor.runtime.digest(
        route / "keyframe-acquisition.json"
    )
    save(route / "report.json", report)
    private_guard(monkeypatch)
    rejected = auditor.audit(route, engine)
    assert not rejected["valid"] and rejected["evaluation"] is None
    assert rejected["stability_diagnostics"] is None


@pytest.mark.parametrize(
    "flags",
    [
        ["--safe-seed-keyframes"],
        [
            "--safe-seed-keyframes",
            "--stable-keyframes",
            "--micro-keyframes",
            "--responsive-keyframes",
        ],
        [
            "--safe-seed-keyframes",
            "--stable-keyframes",
            "--micro-keyframes",
            "--recovering-keyframes",
        ],
    ],
)
def test_safe_seed_flags_require_all_explicit_protocols_before_output(
    scripts, tmp_path, monkeypatch, flags
):
    _, capture = scripts
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys, "argv", [capture.__file__, *flags, "--output", str(output)]
    )
    with pytest.raises(SystemExit):
        capture.main()
    assert not output.exists()

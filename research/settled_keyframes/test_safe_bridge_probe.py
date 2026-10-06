import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

import safe_bridge_probe as probe

from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.shared_map_endpoint import (
    SharedEndpointPolicy,
    _visible_flow,
)


def save(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def bind_public(route):
    report = json.loads((route / "report.json").read_bytes())
    report["config_sha256"] = probe.digest(route / "config.json")
    report["artifact_sha256"] = {
        path.relative_to(route).as_posix(): probe.digest(path)
        for path in [
            route / "keyframe-acquisition.json",
            route / "events.jsonl",
            *(route / "frames").glob("*.png"),
        ]
    }
    save(route / "report.json", report)


def images(count=5, size=448):
    gray = cv2.GaussianBlur(
        np.random.default_rng(27).integers(0, 256, (size, size), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return [
        cv2.warpAffine(rgb, np.array(((1.0, 0, index * 3), (0, 1, 0))), (size, size))
        for index in range(count)
    ]


@pytest.fixture
def public_capture(tmp_path):
    """Synthetic mock ledger only, not native runtime/audit evidence."""
    route = tmp_path / "mock-capture"
    (route / "frames").mkdir(parents=True)
    frames = images()
    frames[-1] = frames[-2].copy()
    for slot, frame in enumerate(frames):
        Image.fromarray(frame).save(route / "frames" / f"frame-{slot:05d}.png")
    samples = []
    for trial, (epoch, first, last) in enumerate(((0, 0, 1), (0, 1, 2), (1, 3, 4))):
        seeds = safe_seeds(frames[first])
        novelty = _visible_flow(
            frames[first], frames[last], SharedEndpointPolicy(), seed_pixels=seeds
        )
        novelty.update(
            seed_detection_registration=probe.REGISTRATION,
            seed_image_sha256=hashlib.sha256(frames[first].tobytes()).hexdigest(),
            seed_pixels=seeds.tolist(),
        )
        selected = last != 4
        samples.append(
            {
                "epoch_id": epoch,
                "trial": trial,
                "absolute_frame": last,
                "selected": selected,
                "correspondence_supported": True,
                "novelty": novelty,
            }
        )
    bindings = {}
    for relative, module in probe.DEPENDENCIES.values():
        key = "package/exploration/" + relative
        source = Path(module.__file__)
        target = route / "source-snapshot" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        bindings[key] = probe.digest(source)
    save(
        route / "config.json",
        {
            "schema": "safe-seed-stable-keyframe-camera-projection-v1",
            "source_sha256": bindings,
        },
    )
    save(
        route / "keyframe-acquisition.json",
        {
            "acquisition": {
                "seed_detection_registration": probe.REGISTRATION,
                "anchor_frame": 0,
                "steps": [{"absolute_frame": slot} for slot in range(1, 5)],
                "terminal": "map_unavailable",
                "epochs": [
                    {"epoch_id": 0, "anchor_frame": 0, "frame_ordinals": [0, 1, 2]},
                    {"epoch_id": 1, "anchor_frame": 3, "frame_ordinals": [3]},
                ],
                "samples": samples,
            },
        },
    )
    save(
        route / "report.json",
        {"completed": True, "failure": None, "checks": {"mock_only": True}},
    )
    (route / "events.jsonl").write_bytes(
        b"synthetic public ledger, not native evidence\n"
    )
    bind_public(route)
    (route / "audit.source.py").write_bytes(b"mock audit only, not native evidence")
    save(
        route / "audit.json",
        {
            "valid": True,
            "checks": {"mock_only": True},
            "audit_snapshot_sha256": probe.digest(route / "audit.source.py"),
        },
    )
    return route


def test_every_actual_pair_replays_across_epochs_without_private_reads(
    public_capture, tmp_path, monkeypatch
):
    old = {
        path: probe.digest(path) for path in public_capture.rglob("*") if path.is_file()
    }
    read_bytes = Path.read_bytes

    def guard(path):
        if path.name.startswith("evaluator-"):
            raise AssertionError("accessed evaluator input")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    output = tmp_path / "diagnosis"
    result = probe.probe(public_capture, output)
    assert all(result["checks"].values())
    assert [r["interval"] for r in result["records"]] == [[0, 1], [1, 2], [3, 4]]
    assert [r["original_selected"] for r in result["records"]] == [True, True, False]
    assert not result["navigation_accepted"]
    assert all(not r["original_sample_replaced"] for r in result["records"])
    assert all(probe.digest(path) == value for path, value in old.items())
    for name, digest in result["config"]["sources_sha256"].items():
        assert probe.digest(output / name) == digest
    with pytest.raises(FileExistsError):
        probe.probe(public_capture, output)


def test_continuous_bridge_preserves_seeds_and_cannot_skip_failed_middle():
    frames = images(size=224)
    seeds = safe_seeds(frames[0])
    before = seeds.copy()
    frames[2] = np.zeros_like(frames[2])
    result = probe.compare_seeded(frames, seeds)
    assert result["direct"]["surviving_count"] > 100
    assert result["continuous"]["surviving_count"] == 0
    assert result["shared_anchor_id_count"] == 0
    assert not result["is_ground_truth_error"]
    assert np.array_equal(before, seeds)
    for mode in ("direct", "continuous"):
        record = result[mode]
        assert record["seed_count"] == len(seeds)
        rejected = [
            identity
            for event in record["diagnosis"]["rejections"]
            for identity in event["track_ids"]
        ]
        survivors = record["diagnosis"]["tracks"]["track_ids"]
        assert len(rejected) == len(set(rejected))
        assert not set(rejected) & set(survivors)
        assert set(rejected) | set(survivors) == set(range(len(seeds)))


@pytest.mark.parametrize("count", [1, 129])
def test_raw_bridge_cannot_be_truncated_to_fit_budget(count):
    with pytest.raises(ValueError, match="complete bounded"):
        probe.compare_seeded([np.zeros((224, 224, 3), dtype=np.uint8)] * count, [])


@pytest.mark.parametrize(
    "case",
    [
        "invalid_audit",
        "empty_audit_checks",
        "audit_source",
        "public_hash",
        "config_hash",
        "schema",
    ],
)
def test_bad_attestation_rejects_before_allocating_output(
    public_capture, tmp_path, case
):
    name = (
        "audit.json"
        if case in ("invalid_audit", "empty_audit_checks", "audit_source")
        else "report.json"
    )
    path = public_capture / name
    data = json.loads(path.read_bytes())
    if case == "invalid_audit":
        data["valid"] = False
    elif case == "empty_audit_checks":
        data["checks"] = {}
    elif case == "audit_source":
        data["audit_snapshot_sha256"] = "0" * 64
    elif case == "public_hash":
        data["artifact_sha256"]["keyframe-acquisition.json"] = "0" * 64
    elif case == "config_hash":
        data["config_sha256"] = "0" * 64
    else:
        path = public_capture / "config.json"
        data = {"schema": "old-detector-schema"}
    save(path, data)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="intact independently audited"):
        probe.probe(public_capture, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "case",
    [
        "seed_count",
        "seed_image",
        "seed_pixels",
        "selection",
        "epoch_frames",
        "missing_epoch",
    ],
)
def test_original_records_cannot_be_replaced(public_capture, tmp_path, case):
    path = public_capture / "keyframe-acquisition.json"
    public = json.loads(path.read_bytes())
    acquisition = public["acquisition"]
    sample = acquisition["samples"][0]
    if case == "seed_count":
        sample["novelty"]["tracks"]["seed_count"] += 1
    elif case == "seed_image":
        sample["novelty"]["seed_image_sha256"] = "0" * 64
    elif case == "seed_pixels":
        sample["novelty"]["seed_pixels"][0][0] += 1
    elif case == "selection":
        sample["selected"] = False
    elif case == "epoch_frames":
        acquisition["epochs"][0]["frame_ordinals"] = [0, 2]
    else:
        acquisition["epochs"] = acquisition["epochs"][:1]
    save(path, public)
    bind_public(public_capture)
    output = tmp_path / "failed-diagnosis"
    with pytest.raises(ValueError, match="original|omitted"):
        probe.probe(public_capture, output)
    assert not (output / "report.json").exists()


def test_late_mutation_preserves_partial_artifacts_but_no_success(
    public_capture, tmp_path, monkeypatch
):
    compare = probe.compare_seeded

    def mutate(frames, seeds):
        result = compare(frames, seeds)
        path = public_capture / "frames/frame-00001.png"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(probe, "compare_seeded", mutate)
    output = tmp_path / "failed-late"
    with pytest.raises(ValueError, match="input bytes changed"):
        probe.probe(public_capture, output)
    assert (output / "probe.source.py").exists()
    assert not (output / "report.json").exists()


def test_overbudget_pair_keeps_direct_failure_without_shorter_bridge(
    public_capture, tmp_path, monkeypatch
):
    path = public_capture / "keyframe-acquisition.json"
    public = json.loads(path.read_bytes())
    sample = public["acquisition"]["samples"][0]
    sample["absolute_frame"] = 129
    public["acquisition"]["samples"] = [sample]
    public["acquisition"]["steps"] = [
        {"absolute_frame": slot} for slot in range(1, 130)
    ]
    public["acquisition"]["epochs"] = [
        {"epoch_id": 0, "anchor_frame": 0, "frame_ordinals": [0, 129]}
    ]
    (public_capture / "frames/frame-00129.png").write_bytes(
        (public_capture / "frames/frame-00001.png").read_bytes()
    )
    save(path, public)
    bind_public(public_capture)

    def forbidden(*args):
        raise AssertionError("attempted raw truncation")

    monkeypatch.setattr(probe, "compare_seeded", forbidden)
    record = probe.probe(public_capture, tmp_path / "bounded")["records"][0]
    assert record["status"] == "raw_interval_exceeds_registered_tracking_budget"
    assert record["comparison"] is None and record["direct_tracks_replayed"]


@pytest.mark.parametrize("schema", list(probe.SCHEMAS))
def test_known_protocols_require_explicit_markers_but_do_not_change_tracking(
    public_capture, tmp_path, schema
):
    config = json.loads((public_capture / "config.json").read_bytes())
    config["schema"] = schema
    save(public_capture / "config.json", config)
    public = json.loads((public_capture / "keyframe-acquisition.json").read_bytes())
    public["acquisition"].update(probe.SCHEMAS[schema])
    save(public_capture / "keyframe-acquisition.json", public)
    bind_public(public_capture)
    result = probe.probe(public_capture, tmp_path / "registered")
    assert result["config"]["capture_schema"] == schema
    assert len(result["records"]) == 3
    assert all(result["checks"].values())


@pytest.mark.parametrize("case", ["source_hash", "snapshot", "marker", "events"])
def test_new_provenance_failures_reject_before_allocating_output(
    public_capture, tmp_path, case
):
    config = json.loads((public_capture / "config.json").read_bytes())
    key = next(iter(config["source_sha256"]))
    if case == "source_hash":
        config["source_sha256"][key] = "0" * 64
    elif case == "snapshot":
        (public_capture / "source-snapshot" / key).write_bytes(b"changed snapshot")
    elif case == "marker":
        config["schema"] = (
            "support-loss-long-wait-jump-safe-seed-stable-keyframe-camera-projection-v1"
        )
    else:
        (public_capture / "events.jsonl").write_bytes(b"changed events")
    save(public_capture / "config.json", config)
    if case != "events":
        bind_public(public_capture)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="dependency|intact independently"):
        probe.probe(public_capture, output)
    assert not output.exists()


def test_terminal_partial_actions_are_kept_without_inventing_a_completed_trial(
    public_capture, tmp_path
):
    path = public_capture / "keyframe-acquisition.json"
    public = json.loads(path.read_bytes())
    public["acquisition"]["steps"].append(
        {"absolute_frame": 5, "requested_kind": "wait"}
    )
    public["acquisition"]["terminal"] = "budget_exhausted"
    frame = public_capture / "frames/frame-00005.png"
    frame.write_bytes((public_capture / "frames/frame-00004.png").read_bytes())
    save(path, public)
    bind_public(public_capture)
    result = probe.probe(public_capture, tmp_path / "partial")
    assert len(result["records"]) == 3
    partial = result["trailing_partial_controls"]
    assert partial["interval"] == [4, 5]
    assert partial["completed_trial"] is False
    assert partial["steps"] == [{"absolute_frame": 5, "requested_kind": "wait"}]
    assert result["original_control_journal"]["terminal"] == "budget_exhausted"
    assert result["public_inputs_sha256"]["frames/frame-00005.png"] == probe.digest(
        frame
    )


@pytest.mark.parametrize("case", ["jump", "float", "bool", "episode_budget"])
def test_partial_or_complete_control_ordinals_cannot_be_retimed(
    public_capture, tmp_path, case
):
    path = public_capture / "keyframe-acquisition.json"
    public = json.loads(path.read_bytes())
    if case == "episode_budget":
        public["acquisition"]["steps"] = [
            {"absolute_frame": slot} for slot in range(1, 290)
        ]
    else:
        public["acquisition"]["steps"][1]["absolute_frame"] = {
            "jump": 5,
            "float": 2.0,
            "bool": True,
        }[case]
    save(path, public)
    bind_public(public_capture)
    with pytest.raises(ValueError, match="control journal"):
        probe.probe(public_capture, tmp_path / "bad-clock")

# ruff: noqa: F401, F811 -- pytest fixture re-exports
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image", reason="requires optional visual extra")
from test_controlled_motion_pipeline import acquisition, modules


@pytest.fixture
def profile(acquisition, modules):
    common, _, _ = modules
    auditor = importlib.import_module("audit_motion_duration")
    route, _ = acquisition
    manifest = json.loads((route / "config.json").read_bytes())
    directory = route.with_name(route.name + ".duration-profile")
    directory.mkdir()
    snapshot = directory / "source-snapshot"
    snapshot.mkdir()
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "environments/minestudio/capture_motion_duration.py"
    sources = common.capture_sources() | {wrapper.name: wrapper}
    hashes = {}
    for name, path in sources.items():
        (snapshot / name).write_bytes(path.read_bytes())
        hashes[name] = common.digest(path)
    config = {"capture_config": manifest["config"], "source_sha256": hashes}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    report = {
        "passed": True,
        "checks": {name: True for name in auditor.CHECK_NAMES},
        "config_sha256": common.digest(directory / "config.json"),
        "constructor_calls": [manifest["config"]],
        "failure": None,
        "capture_config_sha256": common.digest(route / "config.json"),
        "capture_report_sha256": common.digest(route / "report.json"),
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return route, directory, auditor


def test_complete_duration_metadata_and_public_actions_verify_without_telemetry(
    profile, monkeypatch
):
    route, _, auditor = profile
    original = Path.read_bytes

    def public_only(path):
        assert path.name != "evaluator-audit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", public_only)
    result = auditor.audit(route)
    assert result["valid"] and len(result["checks"]) == 8


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_checks",
        "repeated_calls",
        "changed_duration",
        "snapshot",
        "path_traversal",
        "capture_binding",
    ],
)
def test_duration_profile_tampering_fails_closed(profile, tamper):
    route, directory, auditor = profile
    config_path, report_path = directory / "config.json", directory / "report.json"
    config, report = (
        json.loads(config_path.read_bytes()),
        json.loads(report_path.read_bytes()),
    )
    if tamper == "missing_checks":
        report["checks"] = {}
    elif tamper == "repeated_calls":
        report["constructor_calls"] *= 2
    elif tamper == "changed_duration":
        config["capture_config"]["movement_ticks"] = [6, 10]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report["constructor_calls"] = [config["capture_config"]]
        report["config_sha256"] = auditor.digest(config_path)
    elif tamper == "snapshot":
        (directory / "source-snapshot" / "capture_motion_duration.py").write_text(
            "changed", encoding="utf-8"
        )
    elif tamper == "path_traversal":
        config["source_sha256"]["../../outside.py"] = "0" * 64
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report["config_sha256"] = auditor.digest(config_path)
    else:
        report["capture_report_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert not auditor.audit(route)["valid"]

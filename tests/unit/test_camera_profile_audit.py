import importlib
import json
import shutil
from pathlib import Path

import pytest
from test_client_camera_profile import fake_client
from test_controlled_motion_pipeline import (  # noqa: F401 -- pytest fixtures
    acquisition,
    modules,
)

from minecraft_scientist.minecraft.client_camera_profile import (
    file_digest,
    isolated_bobbing_profile,
)


@pytest.fixture
def profile(tmp_path, modules, acquisition):  # noqa: F811 -- imported pytest fixtures
    common, _, _ = modules
    route, _ = acquisition
    config, _, _ = common.validate_acquisition(route)
    directory = tmp_path / "profile"
    manager, instance, _ = fake_client(tmp_path)
    wrapper = (
        Path(__file__).resolve().parents[2]
        / "environments/minestudio/capture_camera_profile.py"
    )
    with isolated_bobbing_profile(
        manager,
        instance,
        directory=directory,
        enabled=False,
        capture_config=config.model_dump(mode="json"),
        wrapper_source=wrapper,
    ) as manifest:
        working = tmp_path / "instance"
        shutil.copytree(manager.RUNTIME_DIR, working)
        instance(working)._launch_minecraft()
    binding = {
        "wrapper_sha256": manifest["source_sha256"]["capture_camera_profile.py"],
        "wrapper_unchanged": True,
        "capture_config_sha256": file_digest(route / "config.json"),
        "capture_report_sha256": file_digest(route / "report.json"),
        "failure": None,
    }
    (directory / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    return importlib.import_module("audit_camera_profile"), directory, route


def test_profile_audit_uses_public_capture_and_no_evaluator(profile):
    auditor, directory, route = profile
    assert not (route / "evaluator-audit.json").exists()
    result = auditor.audit(directory, route)
    assert result["valid"] and len(result["checks"]) == 12
    assert result["enabled"] is False
    assert "supplied at launch" in result["scope"]


@pytest.mark.parametrize("file", ["original-options.txt", "supplied-options.txt"])
def test_changed_options_cannot_pass_identity(profile, file):
    auditor, directory, route = profile
    (directory / file).write_bytes(b"bobView:true\n")
    result = auditor.audit(directory, route)
    assert not result["valid"] and not result["checks"]["options_byte_identity"]


def test_empty_check_dictionary_is_not_a_successful_profile(profile):
    auditor, directory, route = profile
    report = json.loads((directory / "report.json").read_bytes())
    report["checks"] = {}
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    result = auditor.audit(directory, route)
    assert not result["valid"] and not result["checks"]["profile_passed"]


def test_launch_path_traversal_is_rejected_without_reading_external_file(
    profile, monkeypatch
):
    auditor, directory, route = profile
    report = json.loads((directory / "report.json").read_bytes())
    report["launches"][0]["file"] = "../outside.txt"
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    original_digest = auditor.file_digest

    def guarded_digest(path):
        assert path.name != "outside.txt"
        return original_digest(path)

    monkeypatch.setattr(auditor, "file_digest", guarded_digest)
    result = auditor.audit(directory, route)
    assert not result["valid"] and not result["checks"]["all_launches_verified"]


def test_missing_and_failed_binding_is_not_accepted(profile):
    auditor, directory, route = profile
    binding = json.loads((directory / "binding.json").read_bytes())
    binding["failure"] = {"type": "SystemExit", "message": "1"}
    (directory / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    result = auditor.audit(directory, route)
    assert not result["valid"] and not result["checks"]["capture_binding"]
    (directory / "binding.json").unlink()
    result = auditor.audit(directory, route)
    assert not result["valid"] and "failure" in result


def test_source_wrapper_was_frozen_before_capture(profile):
    auditor, directory, route = profile
    binding = json.loads((directory / "binding.json").read_bytes())
    binding["wrapper_unchanged"] = False
    (directory / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    result = auditor.audit(directory, route)
    assert not result["valid"] and not result["checks"]["wrapper_identity"]

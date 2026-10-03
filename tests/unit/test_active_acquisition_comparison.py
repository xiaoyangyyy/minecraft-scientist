import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def comparison(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("compare_active_acquisition")
    engine = tmp_path / "test-only-engine.jar"
    engine.write_bytes(b"unit fixture; not a real engine")
    source = tmp_path / "source.py"
    source.write_text("# frozen unit source\n", encoding="utf-8")
    monkeypatch.setattr(module, "sources", lambda: {"source.py": source})
    calls = []

    def runner(argv, logfile, seconds):
        calls.append((argv, seconds))
        logfile.write_text("unit stub; no Minecraft was executed\n", encoding="utf-8")
        if "--active-causal" in argv:
            raw = Path(argv[argv.index("--output") + 1])
            raw.mkdir()
            strategy = argv[argv.index("--acquisition-strategy") + 1]
            policy = module.ActiveMapPolicy(control_strategy=strategy).model_dump(
                mode="json"
            )
            module.write_json(
                raw / "config.json", {"seed": 201, "active_policy": policy}
            )
            module.write_json(raw / "report.json", {"completed": True, "failure": None})
            module.write_json(
                raw / "active-acquisition.json",
                {
                    "acquisition": {
                        "terminal": "rejected",
                        "failure": None,
                        "policy": policy,
                        "first_control_end_tick": 6 if strategy == "fixed_cap" else 3,
                        "fit_count": 35,
                        "acquisition_attempts": [1] * 7,
                        "steps": [1] * 11,
                    }
                },
            )
        else:
            raw = Path(argv[2])
            strategy = raw.parent.name
            module.write_json(
                raw / "audit.json",
                {
                    "valid": True,
                    "control_strategy": strategy,
                    "evaluation": {
                        "rendered": {"translated": True, "correct": False},
                        "body": {"translated": True, "correct": False},
                    },
                },
            )
        return 0

    return module, engine, source, runner, calls


def test_preregistered_conditions_retained_without_turning_integrity_into_accuracy(
    comparison, tmp_path
):
    module, engine, _, runner, calls = comparison
    output = tmp_path / "compare"
    result = module.run(output, engine, runner=runner)
    assert result["integrity_passed"] and all(result["checks"].values())
    assert len(calls) == 4 and result["registered_episode_count"] == 2
    assert [item["strategy"] for item in result["conditions"]] == [
        "fixed_cap",
        "past_support",
    ]
    assert all(not item["useful_rendered_motion"] for item in result["conditions"])
    assert not result["navigation_accepted"] and not result["statistical_claim"]
    assert (
        result["budget_caps_matched"]
        and not result["compute_matched"]
        and not result["images_matched"]
    )
    config = json.loads((output / "config.json").read_bytes())
    assert (
        config["conditions"][0]["policy"]["maximum_ticks"]
        == config["conditions"][1]["policy"]["maximum_ticks"]
    )
    assert (
        module.digest(output / "source-snapshot/source.py")
        == config["source_sha256"]["source.py"]
    )
    assert module.digest(output / "config.json") == result["config_sha256"]


def test_failed_capture_does_not_drop_condition_or_skip_other_strategy(
    comparison, tmp_path
):
    module, engine, _, runner, calls = comparison

    def fail_first(argv, logfile, seconds):
        if "fixed_cap" in argv:
            logfile.write_text("injected timeout\n", encoding="utf-8")
            return 124
        return runner(argv, logfile, seconds)

    result = module.run(tmp_path / "failed", engine, runner=fail_first)
    assert not result["integrity_passed"]
    assert result["checks"]["all_registered_conditions_retained"]
    assert not result["conditions"][0]["integrity_passed"]
    assert result["conditions"][1]["integrity_passed"]
    assert len(calls) == 2


def test_wrong_strategy_cannot_be_reported_as_matched_reference(comparison, tmp_path):
    module, engine, _, runner, _ = comparison

    def wrong(argv, logfile, seconds):
        code = runner(argv, logfile, seconds)
        if "fixed_cap" in argv:
            raw = Path(argv[argv.index("--output") + 1])
            path = raw / "config.json"
            config = json.loads(path.read_bytes())
            config["active_policy"]["control_strategy"] = "past_support"
            module.write_json(path, config)
        return code

    result = module.run(tmp_path / "wrong", engine, runner=wrong)
    assert not result["integrity_passed"]
    assert not result["conditions"][0]["policy_and_seed_matched"]
    assert result["conditions"][1]["integrity_passed"]


def test_source_change_stops_next_dispatch_but_retains_registered_outcome(
    comparison, tmp_path
):
    module, engine, source, runner, calls = comparison

    def mutate(argv, logfile, seconds):
        code = runner(argv, logfile, seconds)
        if "--active-causal" not in argv:
            source.write_text("# injected new source\n", encoding="utf-8")
        return code

    result = module.run(tmp_path / "changed", engine, runner=mutate)
    assert not result["integrity_passed"] and not result["checks"]["sources_unchanged"]
    assert len(calls) == 2
    assert result["conditions"][1]["dispatch_failure"]["type"] == "ValueError"
    assert result["checks"]["all_registered_conditions_retained"]


def test_existing_output_and_bad_seed_cannot_launch_any_process(comparison, tmp_path):
    module, engine, _, runner, calls = comparison
    directory = tmp_path / "existing"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        module.run(directory, engine, runner=runner)
    for seed in (-1, True, 2**31):
        with pytest.raises(ValueError):
            module.run(tmp_path / "new", engine, seed=seed, runner=runner)
    assert not calls

"""Pinned split isolation, common lifecycle and independent raw scoring."""

import importlib.util
import itertools
import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.structural_campaigns.audit import verify_output
from research.structural_campaigns.corpus import digest, load_development
from research.structural_campaigns.runtime import (
    MATCHED_METHODS,
    PROTOCOL,
    audit,
    learn,
    prelude,
    public_config,
    raw_score,
)
from research.structural_holdouts.generator import fixture, instantiate
from research.structural_holdouts.raw_tables import compose, trace

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def pinned_stub(directory, *, development=2):
    # Loader-unit fixture only; not a substitute for construction audit.
    directory.mkdir()
    generated = [
        fixture(base_index=i, states=2, case="unchanged", boundary=1000)
        for i in range(development + 2)
    ]
    assignments, groups = {}, {}
    for i, record in enumerate(generated):
        fid = record["metadata"]["fixture_id"]
        assignments[fid] = (
            "development"
            if i < development
            else ("tuning" if i == development else "evaluation")
        )
        groups[fid] = f"group-{i if i < development else i + 10}"
        for name, content in (
            ("public.json", record["public"]),
            ("metadata.json", record["metadata"]),
            ("evaluator/world.json", record["private"]),
        ):
            dump(directory / fid / name, content)
    dump(
        directory / "registration.json",
        {"schema": "new-structure-corpus-registration-v1"},
    )
    dump(directory / "partitions.json", {"assignments": assignments, "groups": groups})
    dump(
        directory / "report.json",
        {
            "schema": "new-structure-corpus-report-v1",
            "summary": {
                "development": {
                    "fixtures": development,
                    "independent_groups": development,
                }
            },
        },
    )
    manifest = {
        p.relative_to(directory).as_posix(): digest(p)
        for p in directory.rglob("*.json")
    }
    dump(
        directory / "manifest.json",
        {
            "schema": "new-structure-corpus-manifest-v1",
            "sha256": manifest,
            "nondeterministic_artifacts": ["timing.json"],
        },
    )
    return digest(directory / "manifest.json"), generated


def runner():
    spec = importlib.util.spec_from_file_location(
        "structural_runner", ROOT / "scripts/run_structural_development.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_never_reads_any_sealed_fixture_or_selected_truth(tmp_path, monkeypatch):
    directory = tmp_path / "corpus"
    pin, generated = pinned_stub(directory)
    sealed = {r["metadata"]["fixture_id"] for r in generated[2:]}
    original = Path.read_bytes

    def guarded(path):
        if sealed.intersection(path.parts) or path.name in (
            "world.json",
            "metadata.json",
        ):
            pytest.fail(f"premature/sealed read: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    selected = load_development(directory, manifest_sha256=pin, max_groups=1)
    assert len(selected.fixture_ids) == 1
    assert selected.available_groups == 2
    with pytest.raises(ValueError, match="sealed"):
        selected.read(f"{generated[-1]['metadata']['fixture_id']}/public.json")


@pytest.mark.parametrize("limit", (0, -1, True, 1.2, "1", 100))
def test_invalid_group_limit_rejected(tmp_path, limit):
    directory = tmp_path / "corpus"
    pin, _ = pinned_stub(directory)
    with pytest.raises(ValueError):
        load_development(directory, manifest_sha256=pin, max_groups=limit)


@pytest.mark.parametrize("pin", ("", "0" * 64, "A" * 64, True))
def test_missing_or_forged_manifest_pin_rejected(tmp_path, pin):
    directory = tmp_path / "corpus"
    pinned_stub(directory)
    with pytest.raises(ValueError):
        load_development(directory, manifest_sha256=pin)


def test_pinned_public_bytes_cannot_change_and_group_limit_is_complete(tmp_path):
    directory = tmp_path / "corpus"
    pin, generated = pinned_stub(directory)
    selected = load_development(directory, manifest_sha256=pin)
    assert len(selected.fixture_ids) == 2
    fid = generated[0]["metadata"]["fixture_id"]
    dump(directory / fid / "public.json", {"after": "leaked"})
    with pytest.raises(ValueError, match="drift"):
        selected.read(f"{fid}/public.json")


def test_group_limit_keeps_every_member_not_only_first_fixture(tmp_path):
    directory = tmp_path / "corpus"
    _, generated = pinned_stub(directory)
    partition = json.loads((directory / "partitions.json").read_text())
    for record in generated[:2]:
        partition["groups"][record["metadata"]["fixture_id"]] = "group-0"
    dump(directory / "partitions.json", partition)
    report = json.loads((directory / "report.json").read_text())
    report["summary"]["development"]["independent_groups"] = 1
    dump(directory / "report.json", report)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("partitions.json", "report.json"):
        manifest["sha256"][name] = digest(directory / name)
    dump(directory / "manifest.json", manifest)
    selected = load_development(
        directory, manifest_sha256=digest(directory / "manifest.json"), max_groups=1
    )
    assert len(selected.fixture_ids) == 2
    assert len(set(selected.groups.values())) == 1


@pytest.mark.parametrize(
    "field", ("seed", "case", "boundary", "after", "group", "metadata")
)
def test_private_context_cannot_enter_learner_config(field):
    public = fixture(states=2)["public"]
    public[field] = "private"
    with pytest.raises(ValueError, match="private/extra"):
        public_config(public, MATCHED_METHODS[0], PROTOCOL)


@pytest.mark.parametrize(
    "field,value",
    (
        ("budget", True),
        ("horizon", 6),
        ("monitor_count", 65),
        ("old_budget", 0),
        ("monitor_rounds", 17),
        ("seed", -1),
    ),
)
def test_invalid_protocol_rejected_before_execution(field, value):
    with pytest.raises(ValueError):
        public_config(
            fixture(states=2)["public"], MATCHED_METHODS[0], PROTOCOL | {field: value}
        )


def test_public_config_detaches_mutable_prior():
    public = fixture(states=2)["public"]
    config = public_config(public, MATCHED_METHODS[0], PROTOCOL)
    config["initial_family"]["hypotheses"].clear()
    assert len(public["initial_family"]["hypotheses"]) == 9


@pytest.fixture(scope="module")
def campaigns(tmp_path_factory):
    generated = fixture(base_index=0, states=2, case="unchanged", boundary=1000)
    path = tmp_path_factory.mktemp("common") / "world.json"
    dump(path, generated["private"])
    results = {}
    for method in MATCHED_METHODS:
        config = public_config(generated["public"], method, PROTOCOL)
        result = learn(instantiate(generated["private"]), config)
        results[method] = config, result
    return generated, path, results


def test_four_methods_receive_same_paid_prelude_and_no_private_config(campaigns):
    _, _, results = campaigns
    assert (
        len(
            {
                json.dumps(prelude(result), sort_keys=True)
                for _, result in results.values()
            }
        )
        == 1
    )
    assert all(result["spent"] <= PROTOCOL["budget"] for _, result in results.values())


@pytest.mark.parametrize("method", MATCHED_METHODS)
def test_real_generated_world_public_replay_and_raw_product_audit(campaigns, method):
    generated, path, results = campaigns
    config, result = results[method]
    verified = audit(config, result, path)
    assert verified["valid"], verified
    assert verified["raw_score"]["controls_match"]
    assert verified["raw_score"]["old_correct_at_H"]
    assert verified["raw_score"]["promoted_correct_at_H"] is None
    assert not verified["raw_score"]["behavior_change_at_H"]
    assert generated["metadata"]["inputs"]["boundary"] == 1000


def test_malformed_campaign_never_reads_private_truth(campaigns, monkeypatch):
    _, path, results = campaigns
    config, original = results[MATCHED_METHODS[0]]
    bad = deepcopy(original)
    bad["spent"] += 1

    def forbidden(*args, **kwargs):
        pytest.fail("invalid public ledger accessed private truth")

    monkeypatch.setattr(Path, "read_text", forbidden)
    verified = audit(config, bad, path)
    assert not verified["valid"]
    assert not verified["private_truth_read_attempted"]


def test_raw_scorer_does_not_compile_or_execute_ir(campaigns, monkeypatch):
    from minecraft_scientist.theory.finite_state import FiniteStateProgram

    def forbidden(*args, **kwargs):
        pytest.fail("raw scorer used TheoryIR execution")

    monkeypatch.setattr(FiniteStateProgram, "execute", forbidden)
    generated, _, results = campaigns
    for config, result in results.values():
        assert raw_score(config, result, generated["private"])["controls_match"]


def test_raw_candidate_score_matches_exhaustive_reset_words(campaigns):
    generated, _, results = campaigns
    config, result = results[MATCHED_METHODS[0]]
    program = result["old"]["run"]["candidate_ir"]["parameters"]
    truth = compose(**generated["private"]["before"])
    expected = all(
        trace(program, word) == trace(truth, word)
        for length in range(4)
        for word in itertools.product(config["actions"], repeat=length)
    )
    assert (
        raw_score(
            config | {"protocol": PROTOCOL | {"horizon": 3}},
            result,
            generated["private"],
        )["old_correct_at_H"]
        == expected
    )


def test_full_runner_reads_only_development_and_auditor_rejects_fake_denominators(
    tmp_path, monkeypatch
):
    directory = tmp_path / "corpus"
    pin, generated = pinned_stub(directory, development=1)
    sealed = {r["metadata"]["fixture_id"] for r in generated[1:]}
    read_bytes, read_text = Path.read_bytes, Path.read_text

    def guarded_bytes(path):
        if sealed.intersection(path.parts):
            pytest.fail("runner accessed a sealed split")
        return read_bytes(path)

    def guarded_text(path, *args, **kwargs):
        if sealed.intersection(path.parts):
            pytest.fail("runner accessed a sealed split")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    output = tmp_path / "run"
    report = runner().run(corpus=directory, manifest_sha256=pin, output=output)
    assert report["summary"]["campaigns"] == 4
    assert report["summary"]["independent_development_groups"] == 1
    assert verify_output(output, corpus=directory, manifest_sha256=pin)["valid"]
    report["summary"]["independent_development_groups"] = 4
    dump(output / "report.json", report)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["sha256"]["report.json"] = digest(output / "report.json")
    dump(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="denominators"):
        verify_output(output, corpus=directory, manifest_sha256=pin)


def test_existing_output_refused_and_invalid_protocol_creates_no_output(tmp_path):
    directory = tmp_path / "corpus"
    pin, _ = pinned_stub(directory, development=1)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        runner().run(corpus=directory, manifest_sha256=pin, output=output)
    new = tmp_path / "new"
    with pytest.raises(ValueError):
        runner().run(
            corpus=directory,
            manifest_sha256=pin,
            output=new,
            protocol=PROTOCOL | {"budget": False},
        )
    assert not new.exists()

"""New graph generator, raw factor audit and prior/target leakage negative cases."""

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from .audit_corpus import (
    audit_dataset,
    leakage_partition,
    validate_fixture,
    validate_public,
)
from .generator import CASES, fixture, instantiate, public_entry
from .identity import partition_records
from .raw_tables import apply_edit, compose, equivalent, in_one_edit_family, trace

ROOT = Path(__file__).resolve().parents[2]


def builder():
    spec = importlib.util.spec_from_file_location(
        "structural_corpus_builder", ROOT / "scripts" / "build_structural_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("states", (2, 5))
def test_new_graph_fixtures_have_valid_public_ir_and_private_edit_history(case, states):
    result = fixture(base_index=3, states=states, case=case)
    diagnostics = validate_fixture(
        result["public"], result["private"], result["metadata"]
    )
    assert diagnostics["mechanism_state_count_before"] == states
    assert len(result["public"]["initial_family"]["hypotheses"]) == 9
    if case in ("composed", "out_of_family"):
        assert not diagnostics["within_declared_mechanism_edit_family"]
    if case == "invisible_insert":
        assert diagnostics["within_declared_mechanism_edit_family"]
        assert not diagnostics["behavior_change_unbounded"]
        assert diagnostics["mechanism_state_count_after"] == states + 1


@pytest.mark.parametrize("case", CASES)
def test_names_do_not_change_structural_conditions_or_connected_split(case):
    left = fixture(base_index=2, case=case, renaming_index=0)
    right = fixture(base_index=2, case=case, renaming_index=1)
    assert left["public"] != right["public"]
    assert left["metadata"]["diagnostics"] == right["metadata"]["diagnostics"]
    partition = leakage_partition([left["metadata"], right["metadata"]], salt="frozen")
    assert partition["independent_group_count"] == 1


@pytest.mark.parametrize("base_index", range(4))
def test_before_public_family_shared_across_cases_cannot_cross_splits(base_index):
    records = [
        fixture(base_index=base_index, states=2 + base_index, case=case)["metadata"]
        for case in CASES
    ]
    partition = leakage_partition(records, salt="frozen")
    assert partition["independent_group_count"] == 1
    assert len(set(partition["assignments"].values())) == 1


def test_generation_changes_actual_topology_not_only_names():
    results = [
        fixture(base_index=i, states=2 + i % 4, case="redirect") for i in range(8)
    ]
    assert (
        len(
            {
                r["metadata"]["diagnostics"]["identities"]["pair-structure"][
                    "canonical"
                ]
                for r in results
            }
        )
        >= 6
    )


def test_public_entry_is_detached_and_never_contains_private_metadata():
    result = fixture()
    entry = public_entry(result)
    assert set(entry) == {"actions", "outputs", "costs", "initial_family"}
    entry["initial_family"]["hypotheses"][0]["mechanism_ir"]["parameters"][
        "states"
    ].append("changed")
    assert public_entry(result) != entry
    assert "boundary_episode" not in json.dumps(public_entry(result))
    assert "fixture_id" not in json.dumps(public_entry(result))
    assert "diagnostics" not in json.dumps(public_entry(result))


@pytest.mark.parametrize("case", CASES)
def test_actual_world_before_after_traces_match_independent_raw_factor_product(case):
    result = fixture(base_index=1, states=3, case=case, boundary=1)
    view = instantiate(result["private"])
    for epoch in ("before", "after"):
        observation = view.begin_trial()
        raw = compose(**result["private"][epoch])
        word = tuple(result["public"]["actions"] * 2)
        outputs = []
        for command in word:
            action = next(a for a in view.available_actions() if a.target_id == command)
            transition = view.step(action)
            outputs.append(transition.event.details["signal"])
            assert transition.context["treatment_verification"] == "unknown"
        assert tuple(outputs) == trace(raw, word)
        assert observation.episode_id == (0 if epoch == "before" else 1)


def test_raw_product_does_not_call_ir_compiler_or_interpreter(monkeypatch):
    from minecraft_scientist.discovery.diagnosis import JointHypothesis
    from minecraft_scientist.theory.finite_state import FiniteStateProgram

    def forbidden(*args, **kwargs):
        raise AssertionError("raw dataset audit must not use learner/IR execution")

    monkeypatch.setattr(JointHypothesis, "compile", forbidden)
    monkeypatch.setattr(FiniteStateProgram, "execute", forbidden)
    result = fixture(case="invisible_insert")
    assert (
        validate_fixture(result["public"], result["private"], result["metadata"])[
            "behavior_change_unbounded"
        ]
        is False
    )


@pytest.mark.parametrize("horizon", (0, 1, 3, 5))
def test_raw_bounded_equivalence_matches_exhaustive_word_reference(horizon):
    result = fixture(base_index=2, states=3, case="output")
    left, right = (compose(**result["private"][epoch]) for epoch in ("before", "after"))
    reference = all(
        trace(left, word) == trace(right, word)
        for length in range(horizon + 1)
        for word in itertools.product(left["actions"], repeat=length)
    )
    assert equivalent(left, right, horizon=horizon) is reference


def test_raw_one_edit_membership_rejects_two_changes():
    result = fixture(case="out_of_family")
    assert not in_one_edit_family(
        result["private"]["before"]["mechanism"],
        result["private"]["after"]["mechanism"],
    )
    before = result["private"]["before"]["mechanism"]
    row = before["transitions"][0]
    edit = {
        "operation": "change_output",
        "source": row["source"],
        "action": row["action"],
        "from": row["output"],
        "to": next(o for o in before["outputs"] if o != row["output"]),
    }
    after = apply_edit(before, edit)
    assert in_one_edit_family(before, after)
    assert before != after


@pytest.mark.parametrize(
    "field", ("private", "boundary_episode", "diagnostics", "seed", "case")
)
def test_private_fields_in_public_entry_rejected(field):
    public = fixture()["public"]
    public[field] = "leak"
    with pytest.raises(ValueError):
        validate_public(public)


@pytest.mark.parametrize("part", ("identity", "edit", "truth", "selection"))
def test_fixture_metadata_and_private_tampering_rejected(part):
    result = fixture(case="output")
    if part == "identity":
        result["metadata"]["diagnostics"]["identities"]["target-behavior"]["sha256"] = (
            "0" * 64
        )
    elif part == "edit":
        result["metadata"]["edits"][0]["from"] = "unknown"
    elif part == "truth":
        result["private"]["after"]["mechanism"]["transitions"][0]["output"] = "unknown"
    else:
        result["metadata"]["old_selection"] = [9, 9]
    with pytest.raises(ValueError):
        validate_fixture(result["public"], result["private"], result["metadata"])


@pytest.fixture
def small_corpus(tmp_path):
    module = builder()
    path = tmp_path / "corpus"
    result = module.build(output=path, base_count=1, renamings=2, states=(3,))
    assert result["valid"] and result["fixture_count"] == 16
    return path


def test_dataset_replay_and_manifest_are_deterministic(small_corpus, tmp_path):
    second = tmp_path / "repeat"
    builder().build(output=second, base_count=1, renamings=2, states=(3,))
    assert (small_corpus / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    audit = audit_dataset(small_corpus)
    assert audit["deterministic_artifacts"] == 51
    assert audit["independent_group_count"] == 1


def test_invalid_public_dataset_entry_rejected_before_any_private_read(
    small_corpus, monkeypatch
):
    public_path = next(small_corpus.glob("*/public.json"))
    public = json.loads(public_path.read_text())
    public["seed"] = "forbidden"
    public_path.write_text(json.dumps(public))
    original = Path.read_text
    private_reads = []

    def guarded(path, *args, **kwargs):
        if path.name in ("world.json", "metadata.json"):
            private_reads.append(path)
            raise AssertionError("private data opened before public entry rejection")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(ValueError, match="public entry"):
        audit_dataset(small_corpus)
    assert private_reads == []


@pytest.mark.parametrize("part", ("grid", "split", "source", "report", "extra_file"))
def test_dataset_omissions_and_partition_forgery_rejected(small_corpus, part):
    filename = {
        "grid": "registration.json",
        "source": "registration.json",
        "split": "partitions.json",
        "report": "report.json",
    }.get(part)
    if filename:
        path = small_corpus / filename
        data = json.loads(path.read_text())
        if part == "grid":
            data["fixture_inputs"].pop()
        elif part == "source":
            data["source_sha256"] = {}
        elif part == "split":
            data["independent_group_count"] += 1
        else:
            data["fixture_count"] += 1
        path.write_text(json.dumps(data))
    else:
        (small_corpus / "unregistered.txt").write_text("extra")
    with pytest.raises(ValueError):
        audit_dataset(small_corpus)


@pytest.mark.parametrize(
    "values", ({"seed": True}, {"states": (1,)}, {"renamings": 0}, {"salt": ""})
)
def test_invalid_corpus_protocol_does_not_create_directory(tmp_path, values):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        builder().build(output=destination, **values)
    assert not destination.exists()


def test_existing_output_refused_without_overwriting(small_corpus):
    before = (small_corpus / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        builder().build(output=small_corpus, base_count=1)
    assert (small_corpus / "manifest.json").read_bytes() == before


def test_source_drift_retains_failed_data_without_manifest(tmp_path, monkeypatch):
    module = builder()
    original = module.source_hashes
    count = 0

    def drift():
        nonlocal count
        count += 1
        return original() if count == 1 else {"fake": "0" * 64}

    monkeypatch.setattr(module, "source_hashes", drift)
    destination = tmp_path / "drift"
    with pytest.raises(RuntimeError, match="source drift"):
        module.build(output=destination, base_count=1, renamings=1, states=(2,))
    assert (destination / "registration.json").exists()
    assert list(destination.glob("*/public.json"))
    assert not (destination / "manifest.json").exists()


def test_full_prior_graph_leakage_key_capacity_is_explicit():
    result = partition_records(
        ({"id": "record", "leakage_keys": tuple(f"key-{i}" for i in range(64))},),
        salt="frozen",
    )
    assert result["independent_group_count"] == 1
    with pytest.raises(ValueError):
        partition_records(
            ({"id": "record", "leakage_keys": tuple(f"key-{i}" for i in range(65))},),
            salt="frozen",
        )


def test_failed_postconstruction_audit_retains_data_without_manifest(
    tmp_path, monkeypatch
):
    module = builder()

    def rejected(*args, **kwargs):
        raise ValueError("deliberate failed construction audit")

    monkeypatch.setattr(module, "audit_dataset", rejected)
    destination = tmp_path / "audit-failure"
    with pytest.raises(ValueError, match="failed construction audit"):
        module.build(output=destination, base_count=1, renamings=1, states=(2,))
    assert (destination / "report.json").exists()
    assert list(destination.glob("*/evaluator/world.json"))
    assert not (destination / "manifest.json").exists()

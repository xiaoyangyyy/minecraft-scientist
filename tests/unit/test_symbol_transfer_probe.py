import json

import pytest

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.evaluation import benchmark
from minecraft_scientist.protocols import RunConfig


@pytest.fixture(scope="module")
def primitive():
    result = ScientistAgent(RunConfig(seed=0)).run()
    return next(
        symbol
        for symbol in result.symbols
        if symbol.type_signature == "(processor, input) -> output"
    )


def test_transfer_executes_the_learned_definition_and_retains_events(primitive):
    library = benchmark._run_transfer_probe(0, primitive=primitive)
    scratch = benchmark._run_transfer_probe(0, primitive=None)
    assert library.symbol is primitive and scratch.symbol is None
    assert library.protocols == (benchmark._primitive_protocol(primitive),)
    assert len(library.experiments) == 3 and len(scratch.experiments) == 11
    assert all(experiment.compliant for experiment in library.experiments)
    assert library.experiments[-1].outcome["output_id"] is not None
    assert scratch.experiments[-1].outcome == library.experiments[-1].outcome


def test_symbol_name_does_not_select_or_repair_the_program(primitive):
    renamed = primitive.model_copy(
        update={"name": "opaque-library-name", "symbol_id": "opaque-symbol"}
    )
    assert len(benchmark._run_transfer_probe(0, primitive=renamed).experiments) == 3


def test_wrong_learned_order_is_executed_not_replaced_with_canonical_recipe(
    primitive, monkeypatch
):
    definition = dict(
        primitive.definition_ir, body=["activate", "place_input", "wait", "interact"]
    )
    changed = primitive.model_copy(update={"definition_ir": definition})
    observed = []
    original = benchmark.ExperimentExecutor.run_protocol_test

    def record(self, processor, item, protocol, **kwargs):
        observed.append(protocol)
        return original(self, processor, item, protocol, **kwargs)

    monkeypatch.setattr(benchmark.ExperimentExecutor, "run_protocol_test", record)
    with pytest.raises(RuntimeError, match="did not find"):
        benchmark._run_transfer_probe(0, primitive=changed)
    assert observed == [("activate", "place", "wait:1", "interact")] * 4


@pytest.mark.parametrize(
    "change", ["candidate", "unvalidated", "wrong-type", "unknown-op", "empty-body"]
)
def test_invalid_primitive_stops_before_world_creation(primitive, monkeypatch, change):
    updates = {
        "candidate": {"status": "candidate"},
        "unvalidated": {"validation_examples": ()},
        "wrong-type": {"type_signature": "(actor) -> bool"},
        "unknown-op": {
            "definition_ir": dict(primitive.definition_ir, body=["teleport"])
        },
        "empty-body": {"definition_ir": dict(primitive.definition_ir, body=[])},
    }
    monkeypatch.setattr(
        benchmark.WorldFactory,
        "create",
        lambda *args, **kwargs: pytest.fail("invalid primitive reached world creation"),
    )
    with pytest.raises(ValueError, match="validated executable"):
        benchmark._run_transfer_probe(
            0, primitive=primitive.model_copy(update=updates[change])
        )


def test_noncompliant_success_cannot_become_transfer_evidence(primitive, monkeypatch):
    original = benchmark.ExperimentExecutor.run_protocol_test

    def failed_control(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return result.model_copy(
            update={
                "compliant": False,
                "outcome": {"success": True, "output_id": "opaque-output"},
            }
        )

    monkeypatch.setattr(
        benchmark.ExperimentExecutor, "run_protocol_test", failed_control
    )
    with pytest.raises(RuntimeError, match="did not find"):
        benchmark._run_transfer_probe(0, primitive=primitive)


def test_actual_transfer_artifacts_reference_the_learned_primitive(tmp_path):
    payload = benchmark.run_mvp_benchmark(
        seed=0, intervention_budget=80, output=tmp_path, overwrite=False
    )
    evidence = json.loads((tmp_path / "transfer-evidence.json").read_bytes())
    symbols = json.loads((tmp_path / "full/symbols.json").read_bytes())
    assert evidence["library"]["symbol"] in symbols
    assert evidence["no-library"]["symbol"] is None
    assert (
        evidence["library"]["experiment_count"]
        == payload["ablation"]["full_transfer_experiments"]
    )
    assert (
        evidence["no-library"]["experiment_count"]
        == payload["ablation"]["no_library_transfer_experiments"]
    )
    assert (tmp_path / "library-transfer-events.jsonl").stat().st_size > 0
    assert (tmp_path / "no-library-transfer-events.jsonl").stat().st_size > 0


@pytest.mark.parametrize(
    "filename",
    [
        "library-transfer-events.jsonl",
        "no-library-transfer-events.jsonl",
        "transfer-evidence.json",
        "benchmark.json",
    ],
)
def test_partial_existing_artifact_is_preserved_before_experiments(
    tmp_path, monkeypatch, filename
):
    path = tmp_path / filename
    path.write_bytes(b"owned partial evidence")
    monkeypatch.setattr(
        benchmark.ScientistAgent,
        "run",
        lambda *args: pytest.fail("existing artifact started discovery"),
    )
    with pytest.raises(FileExistsError, match="already exist"):
        benchmark.run_mvp_benchmark(
            seed=0, intervention_budget=80, output=tmp_path, overwrite=False
        )
    assert path.read_bytes() == b"owned partial evidence"
    assert not (tmp_path / "full").exists()

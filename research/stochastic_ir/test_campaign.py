"""Actual capability rollout, source binding and incomplete-run safeguards."""

import gc
import hashlib
import weakref

import pytest

from minecraft_scientist.protocols import TheoryIR
from research.compact_campaigns.store import Reader, decoded

from . import campaign
from .audit import audit_trace
from .fixture import fixture
from .program import StochasticProgram


def test_real_four_fixture_campaign_retains_all_evidence_and_explicit_scope(tmp_path):
    output = tmp_path / "campaign"
    report = campaign.run(output=output, seed=7)
    reader = Reader(
        output,
        manifest_sha256=hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    stems = {f"seed-{s}-noise-{int(n)}" for s in (7, 8) for n in (False, True)}
    expected = {"registration.json", "report.json"} | {
        f"{stem}/{leaf}.json"
        for stem in stems
        for leaf in ("config", "events", "private", "raw-trace", "audit")
    }
    reader.coverage(expected, nondeterministic=())
    assert len(expected) == 22
    assert reader.read("report.json") == report
    assert sum(r["atomic_actions"] for r in report["rows"]) == 192
    assert sum(r["resets"] for r in report["rows"]) == 12
    assert sum(r["actual_unit_cost"] for r in report["rows"]) == 204
    assert report["unknown_controller_discovery_completed"] is False
    assert report["main_interpreter_integration_completed"] is False
    assert all(r["model_discovery_score"] is None for r in report["rows"])
    for stem in stems:
        config = reader.read(f"{stem}/config.json")
        dispatch = tuple(tuple(v) for v in config["frozen_dispatch"])
        audit = audit_trace(
            reader.read(f"{stem}/private.json"),
            reader.read(f"{stem}/raw-trace.json"),
            expected_dispatch=dispatch,
        )
        assert audit == reader.read(f"{stem}/audit.json")
        ledger = reader.read(f"{stem}/events.json")
        assert len(ledger) == len(dispatch) == 51
        model = StochasticProgram.from_ir(
            TheoryIR.model_validate(config["supplied_theory"])
        )
        belief = None
        for item in ledger:
            if item["kind"] == "reset":
                belief = model.initial_belief()
            else:
                predicted = model.predict_step(item["issued_command"], belief=belief)
                assert item["prediction_before_dispatch"] == {
                    o: campaign.mass_dict(sum(v.values())) for o, v in predicted.items()
                }
                _, belief = model.filter(
                    item["issued_command"],
                    item["transition"]["event"]["details"]["signal"],
                    belief=belief,
                )
                assert item["posterior_belief"] == {
                    s: campaign.mass_dict(p) for s, p in sorted(belief.items())
                }


def test_repeated_actual_campaign_has_identical_logical_payloads(tmp_path):
    readers = []
    for name in ("first", "second"):
        output = tmp_path / name
        campaign.run(output=output)
        readers.append(
            Reader(
                output,
                manifest_sha256=hashlib.sha256(
                    (output / "manifest.json").read_bytes()
                ).hexdigest(),
            )
        )
    assert readers[0].payload_hashes() == readers[1].payload_hashes()


def test_no_overwrite(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "user.txt"
    marker.write_bytes(b"untouched")
    with pytest.raises(FileExistsError):
        campaign.run(output=existing)
    assert marker.read_bytes() == b"untouched"


def test_filter_failure_retains_actual_receipt_and_private_trace_without_certificate(
    tmp_path, monkeypatch
):
    def fail(self, *args, **kwargs):
        raise ValueError("injected posterior failure after actual dispatch")

    monkeypatch.setattr(StochasticProgram, "filter", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected posterior"):
        campaign.run(output=output)
    assert not (output / "manifest.json").exists()
    failure = decoded((output / "failure.json.gz").read_bytes(), 1_000_000)[1]
    assert failure["completed_claim"] is False
    trace = decoded(
        (output / "seed-0-noise-0/raw-trace.json.gz").read_bytes(), 1_000_000
    )[1]
    events = decoded(
        (output / "seed-0-noise-0/events.json.gz").read_bytes(), 1_000_000
    )[1]
    assert len(trace) == len(events) == 2
    assert events[-1]["transition"] == trace[-1]["transition"]
    assert (output / "seed-0-noise-0/private.json.gz").exists()


def test_source_drift_retains_all_completed_actions_but_no_manifest(
    tmp_path, monkeypatch
):
    original, calls = campaign.hashes, []

    def drifting():
        calls.append(True)
        value = original()
        return value if len(calls) == 1 else {**value, "injected": "0" * 64}

    monkeypatch.setattr(campaign, "hashes", drifting)
    output = tmp_path / "drift"
    with pytest.raises(ValueError, match="source drift"):
        campaign.run(output=output)
    assert not (output / "manifest.json").exists()
    assert len(list(output.glob("seed-*/raw-trace.json.gz"))) == 4
    assert (output / "failure.json.gz").exists()


@pytest.mark.parametrize("seed", [True, -1, 2**64 - 1, 1.0])
def test_invalid_campaign_seed_rejected_before_any_output(tmp_path, seed):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        campaign.run(output=output, seed=seed)
    assert not output.exists()


def test_world_owner_and_views_release_after_capabilities_are_dropped():
    def create_refs():
        world, _, _, _ = fixture()
        view = world.agent_view()
        view.begin_trial()
        return weakref.ref(world), weakref.ref(view)

    refs = [create_refs() for _ in range(20)]
    gc.collect()
    assert all(world() is None and view() is None for world, view in refs)

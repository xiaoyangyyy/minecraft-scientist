"""Epoch ownership must retain live capabilities, not completed experiments."""

import gc
import weakref

import pytest

from recorded_epoch import RecordedEpoch
from minecraft_scientist.protocols.world import _CAPABILITIES
from minecraft_scientist.world.active_automata import automata_task
from minecraft_scientist.world.identification import _FiniteWorld


def world():
    return _FiniteWorld(automata_task(0, "cycle", 2).ground_truth)


@pytest.mark.parametrize("epochs", (1, 20, 200))
def test_completed_epochs_release_world_receipts_and_capabilities(epochs):
    gc.collect()
    baseline = len(_CAPABILITIES)
    references = []
    for _ in range(epochs):
        instance = world()
        recorder = RecordedEpoch(instance)
        view = recorder.view
        view.begin_trial()
        view.step(view.available_actions()[0])
        assert len(recorder.controls) == 2
        references.extend(weakref.ref(obj) for obj in (instance, recorder, view))
        del instance, recorder, view
    gc.collect()
    assert all(reference() is None for reference in references)
    assert len(_CAPABILITIES) == baseline


def test_live_view_keeps_recorder_and_world_alive_without_external_owner():
    instance = world()
    recorder = RecordedEpoch(instance)
    owner_ref, world_ref = weakref.ref(recorder), weakref.ref(instance)
    view = recorder.view
    del recorder, instance
    gc.collect()
    assert owner_ref() is not None and world_ref() is not None
    assert view.begin_trial().episode_id == 0
    record = view.step(view.available_actions()[0])
    assert record.state_after == view.observe()
    assert len(owner_ref().controls) == 2
    del record, view
    gc.collect()
    assert owner_ref() is None and world_ref() is None


def test_view_identity_stable_while_held_and_recreation_preserves_global_clock():
    instance = world()
    instance.begin_trial()  # Global time predates the rebased epoch.
    recorder = RecordedEpoch(instance)
    view = recorder.view
    assert view is recorder.view
    first = view.begin_trial()
    view.step(view.available_actions()[0])
    assert first.episode_id == 0
    old_ref = weakref.ref(view)
    del view
    gc.collect()
    assert old_ref() is None
    recreated = recorder.view
    assert recreated is recorder.view
    assert recreated.observe().episode_id == 0
    second = recreated.begin_trial()
    assert second.episode_id == 1
    global_resets = [row["observation"]["episode_id"] for row in recorder.controls if row["kind"] == "reset"]
    assert global_resets == [recorder.base, recorder.base + 1]


def test_view_still_has_no_hidden_owner_attributes_and_cannot_serialize():
    import pickle

    view = RecordedEpoch(world()).view
    assert not hasattr(view, "__dict__")
    assert not hasattr(view, "world")
    assert not hasattr(view, "_view_ref")
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(view)

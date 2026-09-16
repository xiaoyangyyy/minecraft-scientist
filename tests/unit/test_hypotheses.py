from minecraft_scientist.discovery import (
    MachineVersionSpace,
    ToolVersionSpace,
    TransformationLearner,
)
from minecraft_scientist.theory import TheoryInterpreter
from minecraft_scientist.world import WorldFactory


def test_tool_version_space_selects_informative_tests_and_resolves() -> None:
    bundle = WorldFactory.create(11, change_episode=None)
    evaluator = bundle.evaluator
    space = ToolVersionSpace(evaluator.tools, evaluator.blocks)
    evidence = 0
    while not space.resolved():
        pair = space.next_experiment()
        assert pair is not None
        evidence += 1
        space.observe(
            pair[0],
            pair[1],
            evaluator.tool_outcome(pair[0], pair[1], 0),
            f"e{evidence}",
        )
    assert evidence <= 9
    ir = space.to_ir()
    for tool in evaluator.tools:
        for target in evaluator.blocks:
            prediction = TheoryInterpreter.predict(ir, tool_id=tool, target_id=target)
            assert prediction["success"] is evaluator.tool_outcome(tool, target, 0)


def test_behaviorally_equivalent_latent_theories_are_resolved() -> None:
    space = ToolVersionSpace(("a", "b", "c"), ("x", "y", "z"))
    truth = space.candidates[0]
    evidence = 0
    while not space.resolved():
        pair = space.next_experiment()
        assert pair is not None
        evidence += 1
        space.observe(
            pair[0],
            pair[1],
            truth.predicts(*pair),
            f"behavior-{evidence}",
        )
    assert space.next_experiment() is not None or evidence == 9
    predictions = {
        tuple(
            candidate.predicts(tool, target)
            for tool in space.tools
            for target in space.targets
        )
        for candidate in space.candidates
    }
    assert len(predictions) == 1


def test_machine_space_exposes_shared_information_gain_features() -> None:
    space = MachineVersionSpace(("r0", "r1", "r2"))
    truth = space.candidates[7]
    steps = 0
    while not space.resolved():
        pair = space.next_experiment()
        assert pair is not None
        pairs, features = space.experiment_features()
        assert len(pairs) == len(features) == 15
        success = truth.predicts(*pair)
        steps += 1
        space.observe(*pair, success, f"machine-{steps}", "out" if success else None)
    assert steps <= 14
    assert space.best() == truth


def test_transformation_version_space_recovers_global_opaque_chain() -> None:
    space = TransformationLearner(
        ("processor_z", "processor_a", "processor_m"),
        ("item_9", "item_2", "item_7", "item_1"),
    )
    truth = space.candidates[83]
    steps = 0
    while not space.resolved():
        pair = space.next_experiment()
        assert pair is not None
        output = truth.predict(*pair)
        steps += 1
        space.observe(*pair, output is not None, output, f"transform-{steps}")
    assert steps < 12
    assert space.best() == truth
    assert space.to_ir().parameters["mapping"] == {
        processor: list(edge) for processor, edge in truth.mapping.items()
    }


def test_transformation_space_absorbs_unseen_terminal_output_and_distractor() -> None:
    space = TransformationLearner(
        ("p0", "p1", "p2"),
        ("raw", "wood", "plank", "stick"),
    )
    truth = {
        "p0": ("wood", "plank"),
        "p1": ("plank", "stick"),
        "p2": ("stick", "torch"),
    }
    steps = 0
    while not space.resolved():
        pair = space.next_experiment()
        assert pair is not None
        source, output = truth[pair[0]]
        observed = output if pair[1] == source else None
        steps += 1
        space.observe(*pair, observed is not None, observed, f"live-{steps}")
    assert space.best().mapping == truth
    assert "torch" in space.resources
    pairs, _ = space.experiment_features()
    assert all(item != "torch" for _, item in pairs)


def test_transformation_does_not_resolve_before_unseen_terminal_is_instantiated() -> None:
    # This opaque ordering reproduced the first real Fabric failure: two positive
    # edges plus one negative previously forced the distractor into the last edge.
    space = TransformationLearner(
        ("p2", "p0", "p1"),
        ("raw_copper", "stick", "plank", "wood"),
    )
    space.observe("p0", "plank", True, "stick", "e1")
    space.observe("p1", "wood", True, "plank", "e2")
    space.observe("p2", "raw_copper", False, None, "e3")

    assert space.resolved() is False
    assert space.next_experiment() == ("p2", "stick")

    space.observe("p2", "stick", True, "torch", "e4")
    assert space.resolved() is True
    assert space.best().mapping == {
        "p0": ("plank", "stick"),
        "p1": ("wood", "plank"),
        "p2": ("stick", "torch"),
    }

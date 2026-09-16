from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.discovery import AbstractionLearner
from minecraft_scientist.protocols import RunConfig


def test_library_learning_promotes_reusable_transform_symbol() -> None:
    full = ScientistAgent(RunConfig(seed=4, library_learning=True)).run()
    ablation = ScientistAgent(RunConfig(seed=4, library_learning=False)).run()
    assert len(full.symbols) == 3
    assert full.symbols[0].name == "transform"
    assert full.symbols[0].compression_gain > 0
    assert full.symbols[0].cross_world_reuse == 1
    assert full.symbols[0].validation_examples
    assert full.symbols[2].generation == 2
    assert full.symbols[2].parent_symbols == ("primitive-capability-threshold-v1",)
    assert all(symbol.validation_examples for symbol in full.symbols)
    assert any(
        "validation_world" in transition.provenance
        for experiment in full.experiments
        for transition in experiment.transitions
    )
    assert len({item.experiment_id for item in full.experiments}) == len(
        full.experiments
    )
    assert ablation.symbols == ()
    assert (
        AbstractionLearner().discover(
            tuple(ablation.theories), enabled=True, validation=None
        )
        == ()
    )
    assert full.report.description_length < ablation.report.description_length

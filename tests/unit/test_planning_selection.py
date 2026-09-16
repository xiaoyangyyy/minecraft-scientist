from minecraft_scientist.planning import (
    ExperimentUtilityWeights,
    experiment_utility,
    select_best_experiment,
)
from minecraft_scientist.discovery import ToolVersionSpace


def test_experiment_utility_combines_information_cost_and_failure_risk() -> None:
    weights = ExperimentUtilityWeights(
        information_gain=1.0,
        cost=0.2,
        control_failure=0.5,
    )
    safe = (0.5, 0.8, 0.0, 1.0, 0.2, 0.0)
    risky = (0.5, 0.9, 0.0, 1.0, 0.2, 0.5)
    assert experiment_utility(safe, weights) > experiment_utility(risky, weights)


def test_selector_prefers_lower_cost_when_information_is_equal() -> None:
    pairs = ("expensive", "cheap", "tested")
    features = (
        (0.5, 0.8, 0.0, 1.0, 1.0, 0.0),
        (0.5, 0.8, 0.0, 1.0, 0.2, 0.0),
        (0.5, 1.0, 1.0, 1.0, 0.0, 0.0),
    )
    assert select_best_experiment(pairs, features) == "cheap"


def test_control_failure_changes_priority_without_filtering_theory() -> None:
    space = ToolVersionSpace(("a", "b", "c"), ("x", "y", "z"))
    pair = space.next_experiment()
    assert pair is not None
    candidate_count = len(space.candidates)
    space.record_control_failure(*pair)
    pairs, features = space.experiment_features()
    failed_features = features[pairs.index(pair)]
    assert failed_features[2] == 0.0
    assert failed_features[5] == 0.5
    assert len(space.candidates) == candidate_count
    assert space.next_experiment() != pair

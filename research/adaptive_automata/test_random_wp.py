"""Independent raw suffix properties, distribution, bounds and paid freshness."""

import copy
import itertools
from dataclasses import dataclass

import pytest
from audit_random_wp import check_random_wp_plan
from random_wp import RandomWpSampler, random_wp_plan
from test_matching import counter, raw_outputs, view

from minecraft_scientist.discovery.active_automata import (
    LearnerStop,
    ObservationTree,
    OutputQueries,
)


@dataclass
class Hypothesis:
    program: object


@pytest.mark.parametrize("states", [1, 2, 3, 4])
def test_covers_and_suffixes_are_actual_raw_state_distinguishers(states):
    program = counter(states)
    raw = program.model_dump(mode="json")
    sampler = RandomWpSampler(program, seed=7)
    reached = set()
    for word in sampler.cover:
        state = raw["initial_state"]
        table = {(r["source"], r["action"]): r for r in raw["transitions"]}
        for a in word:
            state = table[state, a]["target"]
        reached.add(state)
    assert reached == set(raw["states"])
    assert len(sampler.cover) == states
    for left, right in itertools.combinations(raw["states"], 2):
        assert any(
            raw_outputs(raw, w, left) != raw_outputs(raw, w, right)
            for w in sampler.global_suffixes
        )
        assert any(
            raw_outputs(raw, w, left) != raw_outputs(raw, w, right)
            for w in sampler.local_suffixes[left]
        )
        assert any(
            raw_outputs(raw, w, left) != raw_outputs(raw, w, right)
            for w in sampler.local_suffixes[right]
        )


def test_minimum_geometric_middle_and_fair_local_global_coin():
    sampler = RandomWpSampler(counter(3), seed=23, operation_budget=200_000)
    items = [sampler.sample() for _ in range(10_000)]
    lengths = [len(i["middle"]) for i in items]
    assert min(lengths) == 3 and max(lengths) > 20
    assert abs(sum(lengths) / len(lengths) - 6) < 0.15
    assert 0.47 < sum(i["suffix_mode"] == "global" for i in items) / len(items) < 0.53
    # States, not individual input prefixes, receive uniform cover probability.
    counts = [sum(i["prefix"] == list(w) for i in items) for w in sampler.cover]
    assert all(0.31 < c / len(items) < 0.36 for c in counts)


def test_horizon_rejections_keep_full_samples_and_have_no_world_calls():
    plan = random_wp_plan(
        Hypothesis(counter(3)),
        {"a": 1, "b": 2},
        horizon=1,
        count=8,
        seed=0,
        max_samples=32,
    )
    assert plan["status"] == "insufficient_admissible_samples"
    assert plan["queries"] == [] and plan["total_cost"] == 0
    assert len(plan["samples"]) == 32
    assert all(
        len(s["middle"]) >= 3
        and len(s["actions"]) > 1
        and not s["admitted"]
        and s["rejection_reason"] == "query_horizon"
        for s in plan["samples"]
    )


def test_full_prediction_cost_and_deterministic_plan_match_independent_raw_tables():
    program = counter(4)
    kwargs = {"horizon": 5, "count": 32, "seed": 11, "max_samples": 512}
    costs = {"a": 1, "b": 3}
    plan = random_wp_plan(Hypothesis(program), costs, **kwargs)
    assert plan == random_wp_plan(Hypothesis(program), costs, **kwargs)
    assert plan["status"] == "planned" and len(plan["queries"]) == 32
    assert any(not s["admitted"] for s in plan["samples"])
    for item in plan["queries"]:
        word = tuple(item["actions"])
        assert word == tuple(item["prefix"] + item["middle"] + item["suffix"])
        assert (
            tuple(item["prediction"])
            == raw_outputs(
                program.model_dump(mode="json"), word, program.initial_state
            )[0]
        )
        assert item["cost"] == 1 + sum(costs[a] for a in word)
        assert plan["samples"][item["sample_index"]]["admitted"]


def test_duplicate_and_known_tests_are_reexecuted_and_charged_not_deduplicated():
    program = counter(1, actions=("a",))
    costs = {"a": 2}
    plan = random_wp_plan(Hypothesis(program), costs, horizon=3, count=8, seed=0)
    assert all(q["actions"] == ["a"] * 3 for q in plan["queries"])
    tree = ObservationTree(program.actions, program.outputs, epoch="fresh")
    query = OutputQueries(view(program), tree, costs, budget=100, horizon=3)
    query(("a",) * 3, "discovery")
    before = query.spent
    for item in plan["queries"]:
        query(tuple(item["actions"]), "conformance", fresh=True)
    assert query.spent - before == plan["total_cost"] == 56
    assert query.cache_hits == 0 and query.queries == 9


def test_compute_cap_inside_geometric_draw_does_not_truncate_or_pass():
    sampler = RandomWpSampler(counter(1), seed=0, operation_budget=4)
    # Cover construction has consumed 3 visits; starting a draw consumes the
    # fourth and its mandatory middle must stop, not return a shortened test.
    with pytest.raises(LearnerStop, match="random_wp_operation_budget_exhausted"):
        sampler.sample()
    assert sampler.visits == 4


def test_sampler_owns_declarative_model_even_when_executor_table_is_warmed():
    program = counter(3)
    program.execute(("a", "b"))
    sampler = RandomWpSampler(program, seed=0)
    assert sampler.program is not program
    assert sampler.program.model_dump(mode="json") == program.model_dump(mode="json")
    assert sampler.table is not program.table
    assert sampler.sample() == RandomWpSampler(counter(3), seed=0).sample()


@pytest.mark.parametrize("states", [1, 2, 3, 4])
@pytest.mark.parametrize("horizon", [1, 5, 8])
def test_independent_raw_sampler_replay_accepts_complete_and_unresolved_plans(
    states, horizon
):
    costs = {"a": 1, "b": 2}
    plan = random_wp_plan(
        Hypothesis(counter(states)), costs, horizon=horizon, count=8, seed=7
    )
    check_random_wp_plan(plan, costs)


@pytest.mark.parametrize(
    "mutation",
    [
        "seed",
        "prediction",
        "cost",
        "sample_order",
        "admission",
        "truncate",
        "bool_index",
        "extra",
        "visits",
    ],
)
def test_independent_sampler_rejects_mutated_public_plans(mutation):
    costs = {"a": 1, "b": 2}
    plan = copy.deepcopy(
        random_wp_plan(Hypothesis(counter(3)), costs, horizon=5, count=8, seed=7)
    )
    if mutation == "seed":
        plan["seed"] += 1
    elif mutation == "prediction":
        plan["queries"][0]["prediction"][0] = "wrong"
    elif mutation == "cost":
        plan["total_cost"] = float(plan["total_cost"])
    elif mutation == "sample_order":
        plan["samples"].reverse()
    elif mutation == "admission":
        plan["samples"][0]["admitted"] = not plan["samples"][0]["admitted"]
    elif mutation == "truncate":
        rejected = next(s for s in plan["samples"] if not s["admitted"])
        rejected["actions"] = rejected["actions"][:5]
    elif mutation == "bool_index":
        plan["samples"][0]["sample_index"] = False
    elif mutation == "extra":
        plan["private_oracle"] = True
    else:
        plan["sampler_visits"] += 1
    with pytest.raises(ValueError, match="independent random-Wp"):
        check_random_wp_plan(plan, costs)


def test_raw_checker_rejects_a_shared_sampler_prediction_bug(monkeypatch):
    original = RandomWpSampler.trace

    def bad_trace(self, word, state=None):
        outputs, reached = original(self, word, state)
        return (("wrong",) + outputs[1:] if outputs else outputs), reached

    monkeypatch.setattr(RandomWpSampler, "trace", bad_trace)
    costs = {"a": 1, "b": 2}
    plan = random_wp_plan(Hypothesis(counter(3)), costs, horizon=5, count=8, seed=7)
    assert (
        plan["status"] == "planned" and plan["queries"][0]["prediction"][0] == "wrong"
    )
    with pytest.raises(ValueError, match="raw replay/prediction"):
        check_random_wp_plan(plan, costs)

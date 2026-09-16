from __future__ import annotations

import pytest

from minecraft_scientist.causal import CausalSample, LocalCausalEngine


def _sample(
    sample_id: str,
    height: int,
    damage: float,
    *,
    armor: str = "opaque-a",
    confounders: tuple[str, ...] = (),
) -> CausalSample:
    return CausalSample(
        sample_id=sample_id,
        context={"armor": armor},
        intervention={"height": height},
        outcome={"damage": damage},
        possible_confounders=confounders,
    )


def test_causal_effect_uses_only_controlled_compliant_samples() -> None:
    engine = LocalCausalEngine(
        [
            _sample("a", 2, 0.0),
            _sample("b", 8, 5.0),
            _sample("confounded", 8, 100.0, confounders=("armor_changed",)),
            _sample("other-context", 8, 1.0, armor="opaque-b"),
        ]
    )
    effect = engine.controlled_effect(
        "height", "damage", fixed_context={"armor": "opaque-a"}
    )
    assert effect.absolute_effect == 5.0
    assert effect.sample_count == 2


def test_causal_graph_requires_a_measurable_intervention_effect() -> None:
    engine = LocalCausalEngine([_sample("a", 2, 1.0), _sample("b", 8, 1.0)])
    graph = engine.discover_graph(
        treatments=["height"], outcomes=["damage"], minimum_effect=0.1
    )
    assert not graph.has_edge("height", "damage")


def test_causal_effect_rejects_uncontrolled_data() -> None:
    engine = LocalCausalEngine([_sample("a", 2, 0.0)])
    with pytest.raises(ValueError, match="two treatment levels"):
        engine.controlled_effect("height", "damage")

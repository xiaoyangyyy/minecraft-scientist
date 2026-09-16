from pathlib import Path

from minecraft_scientist.cognitive_map import CognitiveMap


def _observation(step: int, x: float, *, block: str, health: float = 20) -> dict:
    return {
        "step": step,
        "position": {"x": x, "y": 64, "z": 0},
        "health": health,
        "food": 20,
        "blocks": [{"name": block}],
        "entities": [],
    }


def test_map_combines_space_knowledge_and_experiment_routes(tmp_path: Path) -> None:
    cognitive_map = CognitiveMap(cell_size=8)
    origin = cognitive_map.ingest(
        _observation(0, 0, block="stone"),
        theory_ids=("tool-capability",),
        uncertainty=0.1,
    )
    middle = cognitive_map.ingest(_observation(1, 9, block="dirt"), uncertainty=0.3)
    target = cognitive_map.ingest(
        _observation(2, 18, block="ore"),
        unresolved_hypotheses=("h-tool-tier",),
        available_experiments=("compare-tools",),
        uncertainty=0.9,
    )
    cognitive_map.current_node_id = origin
    assert cognitive_map.route_to_experiment("compare-tools") == (
        origin,
        middle,
        target,
    )
    assert cognitive_map.route_to_experiment("missing") is None

    path = tmp_path / "map.json"
    cognitive_map.save(path)
    restored = CognitiveMap.load(path)
    assert restored.route_to_experiment("compare-tools") == (origin, middle, target)
    assert restored.graph.nodes[target]["data"].block_histogram == {"ore": 1}


def test_map_merges_repeated_cell_observations() -> None:
    cognitive_map = CognitiveMap(cell_size=8)
    node_id = cognitive_map.ingest(_observation(0, 1, block="stone"))
    same_id = cognitive_map.ingest(
        _observation(1, 2, block="ore", health=10),
        theory_ids=("hazard",),
    )
    assert same_id == node_id
    node = cognitive_map.graph.nodes[node_id]["data"]
    assert node.visits == 2
    assert node.block_histogram == {"stone": 1, "ore": 1}
    assert node.risk > 0

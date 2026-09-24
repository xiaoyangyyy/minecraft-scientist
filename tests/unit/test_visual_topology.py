import json
from pathlib import Path

import pytest

from minecraft_scientist.cognitive_map import VisualTopologicalMap


def test_visual_route_persists_and_rebinds_opaque_controls(tmp_path: Path) -> None:
    learned = VisualTopologicalMap(match_tolerance=1.5)
    source = (1, 9, 13, 8, 11)
    target = (0, 0, 13, 41, 17)
    roles = ("shift_a", "shift_a", "advance", "shift_b")
    route_id = learned.record_route(source, target, roles, success=True)
    target_id = learned.node_id_for(target)

    path = tmp_path / "visual-map.json"
    learned.save(path)
    restored = VisualTopologicalMap.load(path)
    plan = restored.plan((1, 9, 14, 8, 11), target_id)

    assert plan is not None
    assert plan.route_ids == (route_id,)
    assert plan.action_roles == roles
    assert plan.bind({
        "shift_a": "obj_episode2_c",
        "advance": "obj_episode2_a",
        "shift_b": "obj_episode2_d",
    }) == (
        "obj_episode2_c",
        "obj_episode2_c",
        "obj_episode2_a",
        "obj_episode2_d",
    )
    serialized = path.read_text(encoding="utf-8")
    assert not any(token in serialized for token in ("xpos", "zpos", "coordinate"))


def test_visual_map_composes_successful_routes_by_expected_cost() -> None:
    topology = VisualTopologicalMap()
    start, middle, end = (1,), (2,), (3,)
    topology.record_route(start, end, ("slow",) * 10, success=True, cost=10)
    topology.record_route(start, middle, ("first",), success=True, cost=2)
    topology.record_route(middle, end, ("second",), success=True, cost=2)

    plan = topology.plan(start, topology.node_id_for(end))

    assert plan is not None
    assert plan.action_roles == ("first", "second")
    assert plan.node_ids == (
        topology.node_id_for(start),
        topology.node_id_for(middle),
        topology.node_id_for(end),
    )


def test_visual_map_keeps_competing_programs_and_selects_lower_risk_cost() -> None:
    topology = VisualTopologicalMap()
    source, target = (1,), (2,)
    expensive = topology.record_route(
        source, target, ("careful",) * 4, success=True, cost=4
    )
    cheap = topology.record_route(
        source, target, ("direct",), success=True, cost=1
    )

    plan = topology.plan(source, topology.node_id_for(target))

    assert topology.graph.number_of_edges() == 2
    assert plan is not None
    assert plan.route_ids == (cheap,)
    assert expensive != cheap
    assert plan.action_roles == ("direct",)


def test_ambiguous_landmark_match_fails_closed() -> None:
    topology = VisualTopologicalMap(match_tolerance=3, ambiguity_margin=0.5)
    first = topology.observe((0, 0))
    second = topology.observe((4, 0))

    assert first != second
    assert topology.match((2, 0)) is None
    assert topology.match((1, 0)) == first


def test_failed_route_is_not_replayed() -> None:
    topology = VisualTopologicalMap()
    topology.record_route((1,), (2,), ("unsafe",), success=False)

    assert topology.plan((1,), topology.node_id_for((2,))) is None


def test_visual_map_rejects_tampered_route_identifier(tmp_path: Path) -> None:
    topology = VisualTopologicalMap()
    topology.record_route((1,), (2,), ("advance",), success=True)
    path = tmp_path / "map.json"
    topology.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["routes"][0]["route_id"] = "route_forged"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identifier"):
        VisualTopologicalMap.load(path)


def test_visual_map_loads_legacy_single_route_format(tmp_path: Path) -> None:
    topology = VisualTopologicalMap(match_tolerance=0.25)
    topology.record_route((1,), (2,), ("advance",), success=True)
    path = tmp_path / "legacy-map.json"
    topology.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format_version"] = 1
    del payload["ambiguity_margin"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = VisualTopologicalMap.load(path)

    assert restored.ambiguity_margin == 0
    assert restored.plan((1,), restored.node_id_for((2,))) is not None


def test_route_binding_fails_closed_when_episode_role_is_missing() -> None:
    topology = VisualTopologicalMap()
    topology.record_route((1,), (2,), ("advance", "shift"), success=True)
    plan = topology.plan((1,), topology.node_id_for((2,)))

    assert plan is not None
    with pytest.raises(ValueError, match="shift"):
        plan.bind({"advance": "obj_a"})

import json
from pathlib import Path

import pytest

from minecraft_scientist.cognitive_map import (
    VisualRouteStatus,
    VisualTopologicalMap,
)


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
    direct = topology.record_route(
        start, end, ("slow",) * 10, success=True, cost=10
    )
    first = topology.record_route(start, middle, ("first",), success=True, cost=2)
    second = topology.record_route(middle, end, ("second",), success=True, cost=2)

    plan = topology.plan(start, topology.node_id_for(end))
    replanned = topology.plan(middle, topology.node_id_for(end))

    assert plan is not None
    assert plan.route_ids == (first, second)
    assert plan.action_roles == ("first", "second")
    assert plan.node_ids == (
        topology.node_id_for(start),
        topology.node_id_for(middle),
        topology.node_id_for(end),
    )
    assert replanned is not None
    assert replanned.route_ids == (second,)

    topology.challenge_route(second, reason="mid-route negative control")
    fallback = topology.plan(start, topology.node_id_for(end))
    assert fallback is not None
    assert fallback.route_ids == (direct,)
    assert topology.plan(middle, topology.node_id_for(end)) is None


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


def test_high_confidence_failure_challenges_route_and_selects_alternative() -> None:
    topology = VisualTopologicalMap()
    source, target = (1,), (2,)
    direct = topology.record_route(
        source, target, ("direct",), success=True, cost=1
    )
    for _ in range(2):
        topology.record_outcome(direct, success=True, cost=1)
    alternative = topology.record_route(
        source, target, ("detour", "detour"), success=True, cost=2
    )

    challenged = topology.record_outcome(
        direct,
        success=False,
        failure_reason="layout mutation blocked the direct route",
    )
    plan = topology.plan(source, topology.node_id_for(target))

    assert challenged.status is VisualRouteStatus.CHALLENGED
    assert challenged.failure_streak == 1
    assert challenged.status_reason == "layout mutation blocked the direct route"
    assert plan is not None
    assert plan.route_ids == (alternative,)


def test_challenged_route_can_be_superseded_and_lifecycle_persists(
    tmp_path: Path,
) -> None:
    topology = VisualTopologicalMap()
    old = topology.record_route((1,), (2,), ("direct",), success=True)
    replacement = topology.record_route(
        (1,), (2,), ("left", "advance", "right"), success=True
    )
    topology.challenge_route(old, reason="held-out replay failed")
    superseded, promoted = topology.supersede_route(
        old,
        replacement,
        reason="replacement passed shadow validation",
    )
    path = tmp_path / "visual-map-v3.json"
    topology.save(path)
    restored = VisualTopologicalMap.load(path)

    assert superseded.status is VisualRouteStatus.SUPERSEDED
    assert superseded.superseded_by == replacement
    assert promoted.parent_route_id == old
    assert restored.get_route(old).status is VisualRouteStatus.SUPERSEDED
    assert restored.get_route(replacement).parent_route_id == old
    assert restored.plan((1,), restored.node_id_for((2,))).route_ids == (
        replacement,
    )


def test_visual_route_lifecycle_rejects_invalid_transition() -> None:
    topology = VisualTopologicalMap()
    route_id = topology.record_route((1,), (2,), ("advance",), success=True)

    with pytest.raises(ValueError, match="challenged"):
        topology.supersede_route(route_id, route_id, reason="invalid")


def test_visual_map_loads_version_two_route_defaults(tmp_path: Path) -> None:
    topology = VisualTopologicalMap()
    topology.record_route((1,), (2,), ("advance",), success=True)
    path = tmp_path / "version-two-map.json"
    topology.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format_version"] = 2
    for route in payload["routes"]:
        for field in (
            "status",
            "failure_streak",
            "status_reason",
            "parent_route_id",
            "superseded_by",
        ):
            del route[field]
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = VisualTopologicalMap.load(path)

    assert restored.routes()[0].status is VisualRouteStatus.ACTIVE
    assert restored.routes()[0].failure_streak == 0


def test_visual_map_rejects_tampered_supersession_lineage(tmp_path: Path) -> None:
    topology = VisualTopologicalMap()
    old = topology.record_route((1,), (2,), ("direct",), success=True)
    replacement = topology.record_route((1,), (2,), ("detour",), success=True)
    topology.challenge_route(old, reason="counterexample")
    topology.supersede_route(old, replacement, reason="validated replacement")
    path = tmp_path / "tampered-lineage.json"
    topology.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    next(
        route for route in payload["routes"]
        if route["route_id"] == replacement
    )["parent_route_id"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="replacement link"):
        VisualTopologicalMap.load(path)


def test_visual_map_rejects_supersession_cycle(tmp_path: Path) -> None:
    topology = VisualTopologicalMap()
    first = topology.record_route((1,), (2,), ("left",), success=True)
    second = topology.record_route((1,), (2,), ("right",), success=True)
    path = tmp_path / "cyclic-lineage.json"
    topology.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {route["route_id"]: route for route in payload["routes"]}
    for route_id, parent_id in ((first, second), (second, first)):
        by_id[route_id].update({
            "status": "superseded",
            "status_reason": "tampered cycle",
            "parent_route_id": parent_id,
            "superseded_by": parent_id,
        })
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        VisualTopologicalMap.load(path)

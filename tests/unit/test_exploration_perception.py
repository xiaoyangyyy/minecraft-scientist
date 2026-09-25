from __future__ import annotations

from pathlib import Path

from minecraft_scientist.exploration import AnonymousSceneEncoder, ExplorationMemory
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
)


def _observation(
    *,
    origin: tuple[float, float, float],
    prefix: str,
    reverse: bool = False,
    hardness_offset: float = 0,
) -> Observation:
    relative_blocks = ((1, -1, 0, 0.5), (-2, 0, 3, 2.0), (0, 2, -4, 5.0))
    objects = [
        ObjectObservation(
            object_id=f"{prefix}_{index}",
            affordances=(ActionKind.INTERACT,),
            observable_features={
                "x": origin[0] + dx,
                "y": origin[1] + dy,
                "z": origin[2] + dz,
                "distance": 1.0,
                "diggable": True,
                "hardness": hardness + hardness_offset,
            },
        )
        for index, (dx, dy, dz, hardness) in enumerate(relative_blocks)
    ]
    if reverse:
        objects.reverse()
    return Observation(
        episode_id=1,
        step=0,
        inventory={},
        agent_state={"x": origin[0], "y": origin[1], "z": origin[2]},
        objects=tuple(objects),
    )


def test_anonymous_scene_descriptor_ignores_ids_order_and_translation() -> None:
    encoder = AnonymousSceneEncoder()
    first = _observation(origin=(0.5, 64, 0.5), prefix="obj_a")
    translated = _observation(
        origin=(100.5, 80, -39.5),
        prefix="obj_unrelated",
        reverse=True,
    )
    changed = _observation(
        origin=(0.5, 64, 0.5),
        prefix="obj_c",
        hardness_offset=4,
    )

    assert encoder.encode(first) == encoder.encode(translated)
    assert encoder.encode(first) != encoder.encode(changed)
    assert len(encoder.encode(first)) == encoder.DIMENSIONS


def test_perceptual_views_are_bounded_matchable_and_integrity_checked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    with ExplorationMemory(path, max_exemplars=2) as memory:
        first = memory.observe_place((0,), episode_id=0, step=0)
        second = memory.observe_place((10,), episode_id=0, step=0)
        memory.record_perceptual_view(first.place_id, (0, 0), episode_id=0, step=0)
        memory.record_perceptual_view(first.place_id, (1, 0), episode_id=0, step=1)
        perceptual = memory.record_perceptual_view(
            first.place_id, (4, 0), episode_id=1, step=0
        )
        memory.record_perceptual_view(second.place_id, (20, 20), episode_id=0, step=0)

        assert len(perceptual.exemplars) == 2
        assert perceptual.observations == 3
        assert perceptual.episodes == (0, 1)
        assert memory.match_perceptual_place((1, 0), tolerance=2) == first.place_id
        assert memory.match_perceptual_place((9, 9), tolerance=2) is None
        assert memory.summary()["perceptual_place_count"] == 2
        assert memory.verify_integrity()

        memory.record_perceptual_view(second.place_id, (0, 0), episode_id=1, step=1)
        assert (
            memory.match_perceptual_place((0, 0), tolerance=0, ambiguity_margin=0)
            is None
        )
        assert memory.verify_integrity()

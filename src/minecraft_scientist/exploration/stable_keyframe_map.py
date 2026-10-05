"""Acquire public optically stable, novel views before freezing a local map.

Sample slots are explicitly bound to actual image ordinals. Sparse-keyframe
correspondences do not claim continuous raw-tick identity. No world poses, private
timing, block scale, endpoint rescue or navigation acceptance enter this policy.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .causal_map_window import CausalWindowPolicy, _consensus, _relative
from .joint_multiview_map import (
    JointMapConfig,
    JointPose,
    _camera,
    fit_joint_map,
    validate_joint_future,
)
from .map_recovery import _digest
from .second_leg_shadow import retain_tracks
from .settled_map_control import PublicControlStep, _center
from .shared_map_endpoint import SharedEndpointPolicy, _visible_flow
from .temporal_tracks import TemporalTrackConfig, TemporalTracks, track_frame_chain
from .visual_geometry import _frame


class StableKeyframePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum_keyframes: Literal[6] = 6
    maximum_keyframes: Literal[12] = 12
    maximum_acquisition_trials: Literal[11] = 11
    maximum_episode_actions: Literal[288] = 288
    move_ticks: Literal[2] = 2
    maximum_same_handle_trials: Literal[4] = 4
    novelty_median_pixels: Literal[2.0] = 2.0
    minimum_translation_ratio: Literal[0.01] = 0.01
    stability: SharedEndpointPolicy = SharedEndpointPolicy()
    fitting: JointMapConfig = JointMapConfig()
    consensus: CausalWindowPolicy = CausalWindowPolicy()
    temporal: TemporalTrackConfig = TemporalTrackConfig(maximum_frames=13)

    @model_validator(mode="after")
    def _registered(self):
        if (
            self.stability != SharedEndpointPolicy()
            or self.fitting != JointMapConfig()
            or self.consensus != CausalWindowPolicy()
            or self.temporal != TemporalTrackConfig(maximum_frames=13)
        ):
            raise ValueError(
                "keyframe acquisition cannot relax registered geometry/stability gates"
            )
        return self


class KeyframeIndex(BaseModel):
    """Explicit selected-image clock, not consecutive physical ticks/poses."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    namespace: str = Field(min_length=1)
    frame_ordinals: tuple[StrictInt, ...]
    rgb_sha256: tuple[str, ...]
    tracks: TemporalTracks
    continuous_raw_identity: Literal[False] = False

    @model_validator(mode="after")
    def _bound(self):
        if (
            len(self.frame_ordinals) != self.tracks.frame_count
            or len(self.rgb_sha256) != self.tracks.frame_count
            or any(value < 0 for value in self.frame_ordinals)
            or tuple(sorted(set(self.frame_ordinals))) != self.frame_ordinals
            or any(
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in self.rgb_sha256
            )
        ):
            raise ValueError(
                "sampled keyframes need exact chronological ordinals and RGB hashes"
            )
        return self


def acquire_stable_keyframe_map(
    first, execute, camera, *, anchor_frame, seed=2000, policy=None
):
    """First eligible past proposal, then a distinct control and first stable future.

    All actual actions, stable/stalled samples and bootstrap failures are retained.
    Only training landmarks fit poses; every supported frozen map must pass future
    withheld rays and consensus. Optical stability alone never asserts body rest.
    """
    settings = StableKeyframePolicy.model_validate(
        policy.model_dump()
        if isinstance(policy, StableKeyframePolicy)
        else policy or {}
    )
    image, camera = _frame(first).copy(), _camera(camera)
    if (
        type(anchor_frame) is not int
        or not 0 <= anchor_frame <= 48
        or type(seed) is not int
        or not 0 <= seed < 2**31 - 512
    ):
        raise ValueError("keyframe acquisition needs a bounded actual anchor and seed")
    result = {
        "policy": settings.model_dump(mode="json"),
        "anchor_frame": anchor_frame,
        "seed": seed,
        "camera": camera.tolist(),
        "steps": [],
        "flow_prefixes": [],
        "settles": [],
        "samples": [],
        "attempts": [],
        "frozen": None,
        "frozen_sha256": None,
        "future": None,
        "motion": None,
        "terminal": "not_started",
        "failure": None,
        "metric_scale_known": False,
        "global_gauge_aligned": False,
        "navigation_accepted": False,
        "visual_stability_only": True,
        "continuous_raw_identity": False,
    }
    shape = image.shape
    samples, ordinals = [], []
    handle, same, uses = 0, 0, [0] * 4

    def ordinal():
        return anchor_frame + len(result["steps"])

    def finish():
        if (
            result["frozen"] is not None
            and _digest(result["frozen"]) != result["frozen_sha256"]
        ):
            raise ValueError("later controls changed the immutable keyframe proposal")
        return result

    def step(kind, requested):
        nonlocal image
        if ordinal() >= settings.maximum_episode_actions:
            result["terminal"] = "budget_exhausted"
            return False
        try:
            receipt = execute(kind, requested)
        except Exception as error:  # noqa: BLE001 -- retain partial evidence
            result.update(
                terminal="execution_failed",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return False
        if not isinstance(receipt, PublicControlStep):
            result.update(
                terminal="execution_failed",
                failure={
                    "type": "TypeError",
                    "message": "keyframes require a public control receipt",
                },
            )
            return False
        entry = {
            "absolute_frame": ordinal() + 1,
            "requested_kind": kind,
            "requested_handle": requested,
            "executed_kind": receipt.executed_kind,
            "executed_handle": receipt.executed_handle,
            "ticks": receipt.ticks,
            "compliant": receipt.compliant,
        }
        result["steps"].append(entry)
        try:
            current = _frame(receipt.frame).copy()
            entry["rgb_sha256"] = hashlib.sha256(current.tobytes()).hexdigest()
            if current.shape != shape:
                raise ValueError("keyframe controls changed the public sensor")
            if (
                type(receipt.executed_kind) is not str
                or receipt.executed_kind != kind
                or receipt.compliant is not True
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
                or (
                    kind == "move"
                    and (
                        type(receipt.executed_handle) is not int
                        or receipt.executed_handle != requested
                    )
                )
                or (kind == "wait" and receipt.executed_handle is not None)
            ):
                raise ValueError(
                    "keyframe action is not an exact compliant one-tick control"
                )
            previous, image = image, current
            if kind == "wait":
                result["flow_prefixes"].append(
                    _visible_flow(previous, image, settings.stability)
                    | {
                        "interval": [ordinal() - 1, ordinal()],
                        "namespace": f"wait-pair-{ordinal()}",
                    }
                )
            return True
        except Exception as error:  # noqa: BLE001 -- completed actions remain recorded
            result.update(
                terminal="execution_failed",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return False

    def settle(phase):
        start, streak = ordinal(), 0
        for tick in range(1, settings.stability.maximum_wait_ticks + 1):
            if not step("wait", None):
                return False
            streak = streak + 1 if result["flow_prefixes"][-1]["stable"] else 0
            if (
                tick >= settings.stability.minimum_wait_ticks
                and streak >= settings.stability.stable_prefixes
            ):
                result["settles"].append(
                    {
                        "phase": phase,
                        "interval": [start, ordinal()],
                        "stable_tail": [
                            [ordinal() - 3, ordinal() - 2],
                            [ordinal() - 2, ordinal() - 1],
                            [ordinal() - 1, ordinal()],
                        ],
                    }
                )
                return True
        result["terminal"] = f"{phase}_wait_exhausted"
        return False

    def sampled_tracks(extra=None):
        frames = samples + ([extra] if extra is not None else [])
        return track_frame_chain(frames, config=settings.temporal)

    if not settle("initial"):
        return finish()
    samples.append(image.copy())
    ordinals.append(ordinal())
    result["samples"].append(
        {
            "absolute_frame": ordinal(),
            "selected": True,
            "reason": "first public stable prefix",
        }
    )
    maps, past = None, None
    for trial in range(settings.maximum_acquisition_trials):
        if same >= settings.maximum_same_handle_trials:
            handle, same = min(range(4), key=lambda value: (uses[value], value)), 0
        for _ in range(settings.move_ticks):
            if not step("move", handle):
                return finish()
            uses[handle] += 1
        same += 1
        if not settle("acquisition"):
            return finish()
        novelty = _visible_flow(samples[-1], image, settings.stability)
        novel = bool(
            len(novelty["tracks"]["track_ids"]) >= settings.stability.minimum_tracks
            and novelty["cells"] >= settings.stability.minimum_cells
            and len(novelty["tracks"]["track_ids"])
            / max(novelty["tracks"]["seed_count"], 1)
            >= settings.stability.minimum_survival
            and novelty["median_flow_pixels"] is not None
            and novelty["median_flow_pixels"] >= settings.novelty_median_pixels
        )
        result["samples"].append(
            {
                "absolute_frame": ordinal(),
                "selected": novel,
                "handle": handle,
                "novelty": novelty,
                "trial": trial,
            }
        )
        if not novel:
            handle, same = (
                min(
                    (value for value in range(4) if value != handle),
                    key=lambda value: (uses[value], value),
                ),
                0,
            )
            continue
        samples.append(image.copy())
        ordinals.append(ordinal())
        if len(samples) < settings.minimum_keyframes:
            continue
        past = sampled_tracks()
        index = KeyframeIndex(
            namespace=f"sparse-keyframes-{ordinals[0]}",
            frame_ordinals=tuple(ordinals),
            rgb_sha256=tuple(
                hashlib.sha256(frame.tobytes()).hexdigest() for frame in samples
            ),
            tracks=past,
        )
        maps = tuple(
            fit_joint_map(
                past, camera, bootstrap_slot=slot, seed=seed, config=settings.fitting
            )
            for slot in range(2, past.frame_count)
        )
        proposal = {
            "index": index.model_dump(mode="json"),
            "past": past.model_dump(mode="json"),
            "maps": [model.model_dump(mode="json") for model in maps],
            "required_bootstraps": [
                model.bootstrap_slot for model in maps if model.supported
            ],
            "frozen_at_frame": ordinal(),
        }
        result["attempts"].append({"proposal": proposal, "sha256": _digest(proposal)})
        if sum(model.supported for model in maps) >= settings.consensus.minimum_models:
            result["frozen"], result["frozen_sha256"] = proposal, _digest(proposal)
            break
    else:
        result["terminal"] = "map_unavailable"
        return finish()
    frozen_handle = handle
    handle = min(
        (value for value in range(4) if value != frozen_handle),
        key=lambda value: (uses[value], value),
    )
    for _ in range(settings.move_ticks):
        if not step("move", handle):
            return finish()
    if not settle("future"):
        return finish()
    extended = retain_tracks(sampled_tracks(image), set(past.track_ids))
    new_ordinals = ordinals + [ordinal()]
    index = KeyframeIndex(
        namespace=result["frozen"]["index"]["namespace"],
        frame_ordinals=tuple(new_ordinals),
        rgb_sha256=tuple(
            hashlib.sha256(frame.tobytes()).hexdigest() for frame in samples + [image]
        ),
        tracks=extended,
    )
    candidates, relative, ratios = [], [], []
    for model in maps:
        evidence = validate_joint_future(
            model, past, extended, target_slots=(extended.frame_count - 1,)
        )
        candidates.append({"bootstrap_slot": model.bootstrap_slot, "future": evidence})
        if model.supported and evidence["status"] == "consistent":
            end = JointPose.model_validate(evidence["poses"][0])
            relative.append(_relative(model.poses[-1], end))
            radius = max(
                np.linalg.norm(_center(pose) - _center(model.poses[0]))
                for pose in model.poses
            )
            ratios.append(
                float(
                    np.linalg.norm(_center(end) - _center(model.poses[-1]))
                    / max(radius, 1e-9)
                )
            )
    direction, angle, rotation, translation = _consensus(relative, settings.consensus)
    result["future"] = {
        "index": index.model_dump(mode="json"),
        "candidates": candidates,
        "handle": handle,
        "absolute_frame": ordinal(),
    }
    passed = (
        len(relative) == len(result["frozen"]["required_bootstraps"])
        and rotation is not None
        and all(ratio > settings.minimum_translation_ratio for ratio in ratios)
    )
    result["terminal"] = "validated" if passed else "rejected"
    if passed:
        result["motion"] = {
            "frame_ordinals": [ordinals[-1], ordinal()],
            "relative_rotation": rotation,
            "unit_translation": translation,
            "direction_disagreement_degrees": direction,
            "rotation_disagreement_degrees": angle,
            "translation_ratios": ratios,
            "metric_scale_known": False,
            "navigation_accepted": False,
        }
    return finish()

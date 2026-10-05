"""Acquire public optically stable, novel views before freezing a local map.

Sample slots are explicitly bound to actual image ordinals. Sparse-keyframe
correspondences do not claim continuous raw-tick identity. No world poses, private
timing, block scale, endpoint rescue or navigation acceptance enter this policy.
"""

from __future__ import annotations

import copy
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
from .public_response_control import select_public_response
from .public_wait_recovery import settle_with_recovery
from .safe_track_seeds import safe_seeds
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


class StableMicroKeyframePolicy(StableKeyframePolicy):
    """Separately registered small acquisition, unchanged larger future control."""

    move_ticks: Literal[1] = 1
    validation_move_ticks: Literal[2] = 2


class StableResponsiveKeyframePolicy(StableMicroKeyframePolicy):
    """Adapt opaque controls to observed public novelty, without changed gates."""

    action_selection: Literal["public-beta-novelty-v1"] = "public-beta-novelty-v1"


class StableRecoveringKeyframePolicy(StableResponsiveKeyframePolicy):
    """Explicit extra wait/local-epoch budgets; unchanged global/pixel gates."""

    maximum_recovery_windows: Literal[2] = 2
    maximum_local_epochs: Literal[2] = 2
    maximum_total_acquisition_trials: Literal[22] = 22


class StableSafeSeedKeyframePolicy(StableRecoveringKeyframePolicy):
    """First-image support-masked detector with unchanged recovery/geometry gates."""

    seed_detection: Literal["first-image-only-safe-support-mask-v1"] = (
        "first-image-only-safe-support-mask-v1"
    )


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
    first,
    execute,
    camera,
    *,
    anchor_frame,
    seed=2000,
    policy=None,
    micro_controls=False,
    responsive_controls=False,
    recovery_controls=False,
    safe_seed_controls=False,
):
    """First eligible past proposal, then a distinct control and first stable future.

    All actual actions, stable/stalled samples and bootstrap failures are retained.
    Only training landmarks fit poses; every supported frozen map must pass future
    withheld rays and consensus. Optical stability alone never asserts body rest.
    """
    if any(
        type(flag) is not bool
        for flag in (
            micro_controls,
            responsive_controls,
            recovery_controls,
            safe_seed_controls,
        )
    ):
        raise ValueError("micro control registration must be explicit boolean")
    if safe_seed_controls and not recovery_controls:
        raise ValueError("safe seeds require explicit recovering keyframe registration")
    if recovery_controls and not (micro_controls and responsive_controls):
        raise ValueError(
            "recovery controls require explicit responsive micro registration"
        )
    if responsive_controls and not micro_controls:
        raise ValueError("responsive controls require explicit micro registration")
    policy_type = (
        StableSafeSeedKeyframePolicy
        if safe_seed_controls
        else StableRecoveringKeyframePolicy
        if recovery_controls
        else StableResponsiveKeyframePolicy
        if responsive_controls
        else StableMicroKeyframePolicy
        if micro_controls
        else StableKeyframePolicy
    )
    settings = policy_type.model_validate(
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
    if micro_controls:
        result["micro_control_registration"] = "one-tick-acquisition-two-tick-future-v1"
    if responsive_controls:
        result.update(
            response_control_registration="chronological-public-novelty-only-v1",
            control_choices=[],
            future_control_choice=None,
        )
    if recovery_controls:
        result.update(
            recovery_control_registration="bounded-public-wait-and-local-epochs-v1",
            failed_wait_windows=[],
            recovery_wait_windows=[],
            epochs=[],
            epoch_links=[],
        )
    if safe_seed_controls:
        result["seed_detection_registration"] = settings.seed_detection
    shape = image.shape
    samples, ordinals = [], []
    sample_seeds = None
    epoch_outcomes = []
    handle, same, uses = 0, 0, [0] * 4

    def ordinal():
        return anchor_frame + len(result["steps"])

    def finish():
        if (
            result["frozen"] is not None
            and _digest(result["frozen"]) != result["frozen_sha256"]
        ):
            raise ValueError("later controls changed the immutable keyframe proposal")
        if recovery_controls:
            for link in result["epoch_links"]:
                if (
                    _digest(result["epochs"][link["parent_epoch"]])
                    != link["parent_sha256"]
                ):
                    raise ValueError("later controls changed a sealed local epoch")
            for attempt in result["attempts"]:
                if _digest(attempt["proposal"]) != attempt["sha256"]:
                    raise ValueError("later controls changed a retained map proposal")
            if result["epochs"]:
                result["epochs"][-1].update(
                    status=result["terminal"], termination_frame=ordinal()
                )
        return result

    def geometry_failed(stage, error):
        result.update(
            terminal="geometry_failed",
            failure={
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        return finish()

    def visible_flow(first, second):
        if not safe_seed_controls:
            return _visible_flow(first, second, settings.stability)
        seeds = safe_seeds(first)
        flow = _visible_flow(first, second, settings.stability, seed_pixels=seeds)
        if flow["tracks"]["seed_count"] != len(seeds):
            raise ValueError("safe flow changed its first-image seed denominator")
        return flow | {
            "seed_detection_registration": settings.seed_detection,
            "seed_image_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "seed_pixels": seeds.tolist(),
        }

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
        except Exception as error:  # noqa: BLE001 -- completed actions remain recorded
            result.update(
                terminal="execution_failed",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return False
        if kind == "wait":
            try:
                flow = visible_flow(previous, image)
                if type(flow.get("stable")) is not bool:
                    raise ValueError(
                        "public optical stability must be an explicit boolean"
                    )
                result["flow_prefixes"].append(
                    flow
                    | {
                        "interval": [ordinal() - 1, ordinal()],
                        "namespace": f"wait-pair-{ordinal()}",
                    }
                )
            except Exception as error:  # noqa: BLE001 -- a solver error is not control failure
                geometry_failed("stability", error)
                return False
        return True

    def settle(phase):
        if recovery_controls:
            starts = (
                len(result["failed_wait_windows"]),
                len(result["recovery_wait_windows"]),
            )
            settled = settle_with_recovery(
                result, phase, step, ordinal, settings.stability
            )
            epoch_id = len(result["epochs"]) - 1 if result["epochs"] else None
            for name, start in zip(
                ("failed_wait_windows", "recovery_wait_windows"), starts, strict=True
            ):
                for entry in result[name][start:]:
                    entry["epoch_id"] = epoch_id
            return settled
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
        arguments = {"seed_pixels": sample_seeds} if safe_seed_controls else {}
        tracked = track_frame_chain(frames, config=settings.temporal, **arguments)
        if safe_seed_controls and tracked.seed_count != len(sample_seeds):
            raise ValueError(
                "sampled tracks changed the frozen first-image seed denominator"
            )
        return tracked

    def start_epoch():
        nonlocal sample_seeds
        result["epochs"].append(
            {
                "epoch_id": len(result["epochs"]),
                "namespace": f"{'safe-' if safe_seed_controls else ''}sparse-keyframes-{ordinal()}",
                "anchor_frame": ordinal(),
                "frame_ordinals": [ordinal()],
                "trial_ids": [],
                "attempt_indices": [],
                "status": "acquiring",
                "termination_frame": None,
                "global_gauge_aligned": False,
            }
        )
        if safe_seed_controls:
            sample_seeds = safe_seeds(samples[0]).copy()
            sample_seeds.flags.writeable = False
            result["epochs"][-1]["detector"] = {
                "registration": settings.seed_detection,
                "seed_image_frame": ordinal(),
                "seed_image_sha256": hashlib.sha256(samples[0].tobytes()).hexdigest(),
                "seed_pixels": sample_seeds.tolist(),
            }

    def scoring_history():
        return [
            {"trial": local, "handle": item["handle"], "selected": item["selected"]}
            for local, item in enumerate(epoch_outcomes)
        ]

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
    if recovery_controls:
        try:
            start_epoch()
        except Exception as error:  # noqa: BLE001 -- retain the completed stable prefix
            return geometry_failed("seed_detection", error)
        result["samples"][-1]["epoch_id"] = 0
    maps, past = None, None
    total_trials = (
        settings.maximum_total_acquisition_trials
        if recovery_controls
        else settings.maximum_acquisition_trials
    )
    for trial in range(total_trials):
        if (
            recovery_controls
            and len(epoch_outcomes) >= settings.maximum_acquisition_trials
        ):
            break
        if responsive_controls:
            decision = select_public_response(
                scoring_history() if recovery_controls else result["samples"][1:],
                required_distinct_from=handle
                if recovery_controls and same >= settings.maximum_same_handle_trials
                else None,
            )
            if decision["handle"] != handle:
                same = 0
            handle = decision["handle"]
            result["control_choices"].append(
                decision | {"trial": trial, "at_frame": ordinal()}
            )
            if recovery_controls:
                result["control_choices"][-1].update(
                    epoch_id=result["epochs"][-1]["epoch_id"],
                    scoring_trial_ids=[item["trial"] for item in epoch_outcomes],
                )
        elif same >= settings.maximum_same_handle_trials:
            handle, same = min(range(4), key=lambda value: (uses[value], value)), 0
        for _ in range(settings.move_ticks):
            if not step("move", handle):
                return finish()
            uses[handle] += 1
        same += 1
        if not settle("acquisition"):
            return finish()
        try:
            novelty = visible_flow(samples[-1], image)
            supported = bool(
                len(novelty["tracks"]["track_ids"]) >= settings.stability.minimum_tracks
                and novelty["cells"] >= settings.stability.minimum_cells
                and len(novelty["tracks"]["track_ids"])
                / max(novelty["tracks"]["seed_count"], 1)
                >= settings.stability.minimum_survival
            )
            novel = bool(
                supported
                and novelty["median_flow_pixels"] is not None
                and novelty["median_flow_pixels"] >= settings.novelty_median_pixels
            )
        except Exception as error:  # noqa: BLE001 -- retain completed settled controls
            return geometry_failed("novelty", error)
        result["samples"].append(
            {
                "absolute_frame": ordinal(),
                "selected": novel,
                "handle": handle,
                "novelty": novelty,
                "trial": trial,
            }
        )
        if recovery_controls:
            epoch = result["epochs"][-1]
            result["samples"][-1].update(
                epoch_id=epoch["epoch_id"], correspondence_supported=supported
            )
            epoch_outcomes.append(result["samples"][-1])
            epoch["trial_ids"].append(trial)
            if (
                not supported
                and result["settles"][-1]["recovery_windows"] > 0
                and len(result["epochs"]) < settings.maximum_local_epochs
            ):
                epoch.update(
                    status="connection_lost_after_recovery", termination_frame=ordinal()
                )
                result["epoch_links"].append(
                    {
                        "parent_epoch": epoch["epoch_id"],
                        "child_epoch": len(result["epochs"]),
                        "parent_sha256": _digest(epoch),
                        "at_frame": ordinal(),
                        "reason": "public_support_lost_after_recovery",
                        "supported": False,
                        "metric_scale_known": False,
                    }
                )
                samples, ordinals, epoch_outcomes = [image.copy()], [ordinal()], []
                try:
                    start_epoch()
                except Exception as error:  # noqa: BLE001 -- sealed parent stays unchanged
                    return geometry_failed("seed_detection", error)
                result["samples"].append(
                    {
                        "absolute_frame": ordinal(),
                        "selected": True,
                        "reason": "new_local_anchor_after_public_recovery",
                        "epoch_id": result["epochs"][-1]["epoch_id"],
                    }
                )
                continue
        if not novel:
            if not responsive_controls:
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
        if recovery_controls:
            result["epochs"][-1]["frame_ordinals"].append(ordinal())
        if len(samples) < settings.minimum_keyframes:
            continue
        attempt = None
        slot = None
        try:
            past = sampled_tracks()
            index = KeyframeIndex(
                namespace=f"{'safe-' if safe_seed_controls else ''}sparse-keyframes-{ordinals[0]}",
                frame_ordinals=tuple(ordinals),
                rgb_sha256=tuple(
                    hashlib.sha256(frame.tobytes()).hexdigest() for frame in samples
                ),
                tracks=past,
            )
            proposal = {
                "index": index.model_dump(mode="json"),
                "past": past.model_dump(mode="json"),
                "maps": [],
                "required_bootstraps": [],
                "frozen_at_frame": ordinal(),
            }
            if safe_seed_controls:
                proposal["detector"] = copy.deepcopy(result["epochs"][-1]["detector"])
            attempt = {"proposal": proposal, "sha256": None}
            result["attempts"].append(attempt)
            if recovery_controls:
                attempt["epoch_id"] = result["epochs"][-1]["epoch_id"]
                result["epochs"][-1]["attempt_indices"].append(
                    len(result["attempts"]) - 1
                )
            maps = []
            for slot in range(2, past.frame_count):
                model = fit_joint_map(
                    past,
                    camera,
                    bootstrap_slot=slot,
                    seed=seed,
                    config=settings.fitting,
                )
                maps.append(model)
                proposal["maps"].append(model.model_dump(mode="json"))
                if model.supported:
                    proposal["required_bootstraps"].append(model.bootstrap_slot)
            maps = tuple(maps)
            attempt["sha256"] = _digest(proposal)
        except Exception as error:  # noqa: BLE001 -- retain partial bootstrap ledger
            if attempt is not None:
                attempt["failed_bootstrap_slot"] = slot
                attempt["sha256"] = _digest(attempt["proposal"])
            return geometry_failed("past_fitting", error)
        if sum(model.supported for model in maps) >= settings.consensus.minimum_models:
            result["frozen"], result["frozen_sha256"] = proposal, _digest(proposal)
            break
    if result["frozen"] is None:
        result["terminal"] = "map_unavailable"
        return finish()
    frozen_handle = handle
    if responsive_controls:
        decision = select_public_response(
            scoring_history() if recovery_controls else result["samples"][1:],
            required_distinct_from=frozen_handle,
        )
        result["future_control_choice"] = decision | {"at_frame": ordinal()}
        if recovery_controls:
            result["future_control_choice"].update(
                epoch_id=result["epochs"][-1]["epoch_id"],
                scoring_trial_ids=[item["trial"] for item in epoch_outcomes],
            )
        handle = decision["handle"]
    else:
        handle = min(
            (value for value in range(4) if value != frozen_handle),
            key=lambda value: (uses[value], value),
        )
    future_ticks = (
        settings.validation_move_ticks if micro_controls else settings.move_ticks
    )
    for _ in range(future_ticks):
        if not step("move", handle):
            return finish()
    if not settle("future"):
        return finish()
    candidates, relative, ratios = [], [], []
    try:
        extended = retain_tracks(sampled_tracks(image), set(past.track_ids))
        new_ordinals = ordinals + [ordinal()]
        index = KeyframeIndex(
            namespace=result["frozen"]["index"]["namespace"],
            frame_ordinals=tuple(new_ordinals),
            rgb_sha256=tuple(
                hashlib.sha256(frame.tobytes()).hexdigest()
                for frame in samples + [image]
            ),
            tracks=extended,
        )
        result["future"] = {
            "index": index.model_dump(mode="json"),
            "candidates": candidates,
            "handle": handle,
            "absolute_frame": ordinal(),
        }
        for model in maps:
            evidence = validate_joint_future(
                model, past, extended, target_slots=(extended.frame_count - 1,)
            )
            candidates.append(
                {"bootstrap_slot": model.bootstrap_slot, "future": evidence}
            )
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
        direction, angle, rotation, translation = _consensus(
            relative, settings.consensus
        )
    except Exception as error:  # noqa: BLE001 -- future failure never refits the parent
        return geometry_failed("future_validation", error)
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

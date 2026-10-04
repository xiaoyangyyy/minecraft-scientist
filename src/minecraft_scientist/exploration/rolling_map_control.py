"""Past-only rolling local maps with explicit lineage and informative handles.

Every model freezes before its future actions. Earlier failures are never
reclassified; detector generations and monocular coordinate gauges stay local.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .causal_map_window import CausalMapWindow, CausalWindowPolicy
from .joint_multiview_map import _camera
from .map_gauge_bridge import GaugeBridgePolicy, register_map_gauges
from .map_recovery import _digest
from .second_leg_shadow import retain_tracks
from .settled_map_control import PublicControlStep
from .shared_anchor_map import SharedAnchorPolicy, SharedAnchorWindow
from .temporal_tracks import TemporalTrackConfig, track_frame_chain
from .visual_geometry import _frame


class RollingMapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    maximum_epochs: int = Field(default=2, ge=1, le=2, strict=True)
    history_frames: int = Field(default=12, ge=12, le=12, strict=True)
    maximum_acquisition_ticks: int = Field(default=4, ge=0, le=4, strict=True)
    maximum_total_actions: int = Field(default=200, ge=200, le=200, strict=True)
    temporal: TemporalTrackConfig = TemporalTrackConfig(maximum_frames=14)
    window: CausalWindowPolicy = CausalWindowPolicy()

    @model_validator(mode="after")
    def _bounded(self):
        if self.temporal.maximum_frames != 14 or self.window != CausalWindowPolicy():
            raise ValueError("rolling maps cannot change the registered geometry gates")
        return self


def _choose_handle(counts, excluded):
    choices = [handle for handle in range(4) if handle not in excluded]
    if not choices:
        raise ValueError("rolling controls need a distinct available handle")
    return min(choices, key=lambda handle: (counts[handle], handle))


def maintain_public_maps(
    parent,
    frames,
    execute,
    camera,
    *,
    policy=None,
    gauge_policy=None,
    shared_policy=None,
):
    """Two bounded map epochs after public recovery, not global navigation.

    Each failed/passing past fit has a distinct immutable image/track namespace.
    A least-used opaque handle supplies missing excitation; validation excludes
    the last acquisition handle. Counts express public novelty, not semantics,
    collision labels, predicted world directions or independent confidence.
    """
    settings = RollingMapPolicy.model_validate(
        policy.model_dump() if isinstance(policy, RollingMapPolicy) else policy or {}
    )
    camera = _camera(camera)
    gauge_settings = (
        GaugeBridgePolicy.model_validate(
            gauge_policy.model_dump()
            if isinstance(gauge_policy, GaugeBridgePolicy)
            else gauge_policy
        )
        if gauge_policy is not None
        else None
    )
    parent_digest = _digest(parent)
    shared_settings = (
        SharedAnchorPolicy.model_validate(
            shared_policy.model_dump()
            if isinstance(shared_policy, SharedAnchorPolicy)
            else shared_policy
        )
        if shared_policy is not None
        else None
    )
    if shared_settings is not None and gauge_settings is None:
        raise ValueError("shared reconstruction requires the retained gauge journal")
    result = {
        "policy": settings.model_dump(mode="json"),
        "parent_sha256": parent_digest,
        "epochs": [],
        "steps": [],
        "failure": None,
        "terminal": "not_started",
        "global_gauge_aligned": False,
        "metric_scale_known": False,
        "navigation_accepted": False,
    }
    if gauge_settings is not None:
        result.update(
            gauge_policy=gauge_settings.model_dump(mode="json"),
            gauge_bridges=[],
            supported_local_gauge_links=0,
        )
    if shared_settings is not None:
        result.update(
            shared_anchor_policy=shared_settings.model_dump(mode="json"),
            shared_anchor_versions=[],
            shadow_validated_shared_versions=0,
        )

    def finish():
        if _digest(parent) != parent_digest:
            raise ValueError("rolling control changed its immutable parent")
        if gauge_settings is not None:
            if any(
                _digest(item["evidence"]) != item["evidence_sha256"]
                for item in result["gauge_bridges"]
            ):
                raise ValueError("future controls changed a frozen gauge checkpoint")
            result["supported_local_gauge_links"] = sum(
                item["evidence"]["supported"] for item in result["gauge_bridges"]
            )
        if shared_settings is not None:
            for version in result["shared_anchor_versions"]:
                if (
                    version["frozen"] is not None
                    and _digest(version["frozen"]) != version["frozen_sha256"]
                ):
                    raise ValueError(
                        "future controls changed a frozen shared-anchor version"
                    )
            result["shadow_validated_shared_versions"] = sum(
                item["status"] == "shadow_validated"
                for item in result["shared_anchor_versions"]
            )
        return result

    if parent["failure"] is not None:
        raise ValueError("rolling maps cannot consume failed controls/tracking")
    child = parent["acquisition"]
    if child is None or parent["terminal"] == "completed":
        return finish()
    if parent["terminal"] not in ("map_unavailable", "continuation_aborted"):
        raise ValueError("rolling maintenance requires a declared public map failure")
    stopped = parent["settled"]
    count = stopped["evidence_frames"] if stopped else child["evidence_frames"]
    anchor = child["anchor_frame"]
    seed = parent["seed"]
    if (
        type(anchor) is not int
        or anchor < 0
        or type(count) is not int
        or count < 1
        or type(seed) is not int
        or not 0 <= seed < 2**31 - 562
    ):
        raise ValueError("rolling maps need bounded public image/seed endpoints")
    images = [_frame(image).copy() for image in frames]
    if len(images) != count or any(image.shape != images[0].shape for image in images):
        raise ValueError(
            "rolling history must contain exactly the consumed public epoch"
        )
    if stopped and (
        not stopped["prefixes"]
        or stopped["prefixes"][-1]["absolute_frame"] != anchor + count - 1
    ):
        raise ValueError("rolling parent endpoint differs from executed public history")
    counts = {handle: 0 for handle in range(4)}
    last_handle = None
    for step in child["steps"] + (stopped["steps"] if stopped else []):
        handle = step["requested_handle"]
        if handle is not None:
            if type(handle) is not int or handle not in counts:
                raise ValueError("rolling history contains an unknown action handle")
            counts[handle] += 1
            last_handle = handle
    result["initial_handle_counts"] = dict(counts)
    end = anchor + len(images) - 1
    lineage = parent_digest

    def advance(handle, phase):
        nonlocal end, last_handle
        if end >= settings.maximum_total_actions:
            result["terminal"] = "budget_exhausted"
            return False
        record = {
            "phase": phase,
            "requested_handle": handle,
            "executed_handle": None,
            "executed_kind": None,
            "ticks": None,
            "compliant": False,
            "action_response_frame": end + 1,
        }
        result["steps"].append(record)
        try:
            receipt = execute("move", handle)
            if not isinstance(receipt, PublicControlStep):
                raise TypeError("rolling maps require a public RGB/action receipt")
            record.update(
                executed_kind=receipt.executed_kind
                if type(receipt.executed_kind) is str
                else None,
                executed_handle=receipt.executed_handle
                if type(receipt.executed_handle) is int
                else None,
                ticks=receipt.ticks if type(receipt.ticks) is int else None,
                compliant=receipt.compliant
                if type(receipt.compliant) is bool
                else False,
            )
            if (
                type(receipt.executed_kind) is not str
                or receipt.executed_kind != "move"
                or type(receipt.executed_handle) is not int
                or receipt.executed_handle != handle
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
                or receipt.compliant is not True
            ):
                raise ValueError("failed rolling controls cannot enter pose evidence")
            image = _frame(receipt.frame).copy()
            if image.shape != images[0].shape:
                raise ValueError("rolling control changed RGB dimensions")
            if _digest(parent) != parent_digest:
                raise ValueError("rolling receipt mutated its parent")
        except Exception as error:  # noqa: BLE001 -- retain partial public controls
            result.update(
                terminal="control_failed",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return False
        images.append(image)
        end += 1
        counts[handle] += 1
        last_handle = handle
        return True

    for epoch in range(settings.maximum_epochs):
        entry = {
            "number": epoch,
            "parent_sha256": lineage,
            "attempts": [],
            "frozen": None,
            "validation_handle": None,
            "future_prefixes": [],
            "decision": None,
            "terminal": "collecting",
        }
        result["epochs"].append(entry)
        acquisition_handle = _choose_handle(counts, {last_handle})
        controller = None
        shared_controller, shared_images, shared_version = None, None, None
        for attempt in range(settings.maximum_acquisition_ticks + 1):
            if len(images) < 3:
                entry["terminal"] = "insufficient_history"
                break
            past_images = images[-settings.history_frames :]
            local_anchor = end - len(past_images) + 1
            try:
                past = track_frame_chain(past_images, config=settings.temporal)
                if (
                    past.frame_count != len(past_images)
                    or past.image_shape != images[0].shape[:2]
                ):
                    raise ValueError("rolling fitter changed its consumed image prefix")
                candidate = CausalMapWindow(
                    past,
                    camera,
                    anchor_frame=local_anchor,
                    seed=seed + 5 + epoch * 25 + attempt * 5,
                    policy=settings.window,
                )
                snapshot = candidate.snapshot()
                supported = sum(item["supported"] for item in snapshot["maps"])
                evidence = {
                    "attempt": attempt,
                    "anchor_frame": local_anchor,
                    "track_namespace": f"rolling-{epoch}/attempt-{attempt}/anchor-{local_anchor}",
                    "rgb_sha256": [
                        hashlib.sha256(image.tobytes()).hexdigest()
                        for image in past_images
                    ],
                    "snapshot": snapshot,
                    "snapshot_sha256": candidate.digest(),
                    "supported_maps": supported,
                }
                entry["attempts"].append(evidence)
                if supported >= settings.window.minimum_models:
                    controller = candidate
                    entry["frozen"] = evidence
                    if gauge_settings is not None and epoch > 0:
                        previous = result["epochs"][-2]["frozen"]
                        if previous is not None:
                            bridge = register_map_gauges(
                                previous["snapshot"],
                                evidence["snapshot"],
                                parent_images=previous["rgb_sha256"],
                                child_images=evidence["rgb_sha256"],
                                parent_namespace=previous["track_namespace"],
                                child_namespace=evidence["track_namespace"],
                                policy=gauge_settings,
                            )
                            result["gauge_bridges"].append(
                                {
                                    "parent_epoch": epoch - 1,
                                    "child_epoch": epoch,
                                    "evidence": bridge,
                                    "evidence_sha256": _digest(bridge),
                                }
                            )
                            if shared_settings is not None:
                                shared_version = {
                                    "parent_epoch": epoch - 1,
                                    "child_epoch": epoch,
                                    "original_snapshot_sha256": previous[
                                        "snapshot_sha256"
                                    ],
                                    "frozen": None,
                                    "frozen_sha256": None,
                                    "future_prefixes": [],
                                    "decision": None,
                                    "status": "proposed",
                                    "terminal": "not_started",
                                }
                                result["shared_anchor_versions"].append(shared_version)
                                shared_images = images[
                                    previous["anchor_frame"] - anchor :
                                ]
                                if (
                                    len(shared_images)
                                    > shared_settings.maximum_consumed_frames
                                ):
                                    shared_version.update(
                                        status="rejected",
                                        terminal="history_budget_exhausted",
                                    )
                                else:
                                    shared_temporal = settings.temporal.model_copy(
                                        update={"maximum_frames": 16}
                                    )
                                    consumed = track_frame_chain(
                                        shared_images, config=shared_temporal
                                    )
                                    # Earlier lost IDs can never be regenerated by a later tracker call.
                                    retained = set(
                                        previous["snapshot"]["past"]["track_ids"]
                                    )
                                    for prefix in result["epochs"][-2][
                                        "future_prefixes"
                                    ]:
                                        retained &= set(prefix["track_ids"])
                                    consumed = retain_tracks(consumed, retained)
                                    shared_controller = SharedAnchorWindow(
                                        previous["snapshot"],
                                        consumed,
                                        policy=shared_settings,
                                    )
                                    shared_version.update(
                                        frozen=shared_controller.snapshot(),
                                        frozen_sha256=shared_controller.digest(),
                                        rgb_sha256=[
                                            hashlib.sha256(image.tobytes()).hexdigest()
                                            for image in shared_images
                                        ],
                                        terminal="awaiting_new_controls",
                                    )
                    break
                if attempt == settings.maximum_acquisition_ticks:
                    entry["terminal"] = "map_unavailable"
                    break
                if not advance(acquisition_handle, "acquisition"):
                    entry["terminal"] = result["terminal"]
                    return finish()
            except Exception as error:  # noqa: BLE001 -- preserve failed fits and input
                entry["terminal"] = "tracking_failed"
                result.update(
                    terminal="tracking_failed",
                    failure={"type": type(error).__name__, "message": str(error)},
                )
                return finish()
        if controller is None:
            result["terminal"] = entry["terminal"]
            return finish()
        future_handle = _choose_handle(counts, {last_handle})
        entry["validation_handle"] = future_handle
        # Immutable checkpoint already exists before either future action.
        frozen_digest = controller.digest()
        extended_images = [image.copy() for image in past_images]
        for _ in range(settings.window.horizon):
            if not advance(future_handle, "validation"):
                entry["terminal"] = result["terminal"]
                return finish()
            extended_images.append(images[-1].copy())
            try:
                extended = track_frame_chain(extended_images, config=settings.temporal)
                if extended.frame_count != len(extended_images):
                    raise ValueError("rolling validation changed executed frame count")
                decision = controller.advance(extended)
                entry["future_prefixes"].append(extended.model_dump(mode="json"))
                entry["decision"] = decision.model_dump(mode="json")
                if shared_controller is not None:
                    shared_images.append(images[-1].copy())
                    shared_prefix = track_frame_chain(
                        shared_images, config=shared_temporal
                    )
                    shared_decision = shared_controller.advance(shared_prefix)
                    shared_version["future_prefixes"].append(
                        shared_prefix.model_dump(mode="json")
                    )
                    shared_version["decision"] = shared_decision.model_dump(mode="json")
            except Exception as error:  # noqa: BLE001 -- preserve future failed prefix
                entry["terminal"] = "tracking_failed"
                result.update(
                    terminal="tracking_failed",
                    failure={"type": type(error).__name__, "message": str(error)},
                )
                return finish()
        # The original checkpoint contains no future observations, even though
        # its terminal descendant retains every original model and new counterexample.
        entry["terminal_snapshot"] = controller.snapshot()
        entry["terminal_sha256"] = controller.digest()
        if (
            _digest(entry["frozen"]["snapshot"]) != frozen_digest
            or entry["frozen"]["snapshot_sha256"] != frozen_digest
        ):
            raise ValueError("rolling validation changed the frozen past checkpoint")
        entry["terminal"] = (
            "validated" if decision.instruction == "refresh" else "rejected"
        )
        if shared_controller is not None:
            shared_version.update(
                terminal="validated"
                if shared_decision.instruction == "refresh"
                else "rejected",
                status="shadow_validated"
                if shared_decision.instruction == "refresh"
                else "rejected",
                terminal_snapshot=shared_controller.snapshot(),
                terminal_sha256=shared_controller.digest(),
            )
        lineage = _digest(entry)
    result.update(terminal="completed", final_handle_counts=counts)
    return finish()

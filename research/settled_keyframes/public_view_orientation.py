"""Bounded public LOOK program selected before any new controls are dispatched.

Research prototype only. Counted LOOK phases do not establish actual heading,
arrival, or an accepted map. A native adapter/auditor registration is required
before integrating this program with keyframe acquisition.
"""

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from public_view_quality import rank_views

from minecraft_scientist.exploration.visual_geometry import _frame

ORDINALS = list(range(15, 49, 3))


@dataclass(frozen=True, slots=True)
class OrientationProfile:
    start_frame: int = 48
    maximum_look_ticks: int = 11
    waits_after_each_look: int = 2
    receipt_ticks: int = 1

    def __post_init__(self):
        if any(
            type(value) is not int or value != exact
            for value, exact in (
                (self.start_frame, 48),
                (self.maximum_look_ticks, 11),
                (self.waits_after_each_look, 2),
                (self.receipt_ticks, 1),
            )
        ):
            raise ValueError("public orientation requires exact registered clocks")


@dataclass(frozen=True)
class SurveyReceipt:
    frame: np.ndarray
    executed_kind: Literal["look", "wait"]
    executed_right_cell: bool
    compliant: bool
    ticks: int = 1


@dataclass(frozen=True)
class OrientationOutcome:
    journal: dict
    frame: np.ndarray


def orient(images, execute, *, profile=None):
    profile = OrientationProfile() if profile is None else profile
    if type(profile) is not OrientationProfile:
        raise ValueError("requires an explicit exact public orientation profile")
    if type(images) is not list or len(images) != 12:
        raise ValueError("orientation requires twelve completed startup frames")
    snapshots = [_frame(image).copy() for image in images]
    if any(image.shape != (448, 448, 3) for image in snapshots):
        raise ValueError("orientation cannot transform the public RGB sensor")
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in snapshots]
    for image in snapshots:
        image.setflags(write=False)
    ranked = rank_views(snapshots, ORDINALS)
    if hashes != [hashlib.sha256(image.tobytes()).hexdigest() for image in snapshots]:
        raise ValueError("ranking changed its frozen startup observations")
    selected = ranked["selected_frame"]
    if selected is not None and (type(selected) is not int or selected not in ORDINALS):
        raise ValueError("orientation cannot select a future or unregistered reference")
    image = snapshots[-1].copy()
    result = {
        "registration": "twelve-view-public-survey-counted-look-v1",
        "quality_registration": "first-image-separated-safe-patch-view-quality-v1",
        "start_frame": profile.start_frame,
        "end_frame": profile.start_frame,
        "reference_frame_ordinals": ORDINALS.copy(),
        "reference_rgb_sha256": hashes,
        "quality": ranked,
        "selected_reference_frame": selected,
        "requested_look_ticks": None,
        "steps": [],
        "terminal": "no_eligible_view" if selected is None else "not_started",
        "failure": None,
        "actual_heading_known": False,
        "navigation_accepted": False,
        "scope": "First-image ranking frozen before controls; fixed right-middle public LOOK target and two WAITs per counted LOOK. No assumed actual heading, matched reference identity, player position, map or arrival.",
    }
    if selected is None:
        return OrientationOutcome(result, image)
    # Replay the selected startup command PHASE, not a privileged yaw value.
    turns = ((selected - 12) // 3) % 12
    if not 0 <= turns <= profile.maximum_look_ticks:
        raise ValueError("orientation cannot exceed its declared LOOK budget")
    result["requested_look_ticks"] = turns
    for kind in [kind for _ in range(turns) for kind in ("look", "wait", "wait")]:
        try:
            receipt = execute(kind)
            if not isinstance(receipt, SurveyReceipt):
                raise TypeError("orientation needs an exact public survey receipt")
            result["end_frame"] += 1
            entry = {
                "absolute_frame": result["end_frame"],
                "requested_kind": kind,
                "requested_right_cell": kind == "look",
                "executed_kind": receipt.executed_kind,
                "executed_right_cell": receipt.executed_right_cell,
                "compliant": receipt.compliant,
                "ticks": receipt.ticks,
            }
            result["steps"].append(entry)
            observed = _frame(receipt.frame).copy()
            entry["rgb_sha256"] = hashlib.sha256(observed.tobytes()).hexdigest()
            if observed.shape != image.shape:
                raise ValueError("orientation control changed the public sensor")
            image = observed
            if (
                type(receipt.executed_kind) is not str
                or receipt.executed_kind != kind
                or type(receipt.executed_right_cell) is not bool
                or receipt.executed_right_cell is not (kind == "look")
                or type(receipt.compliant) is not bool
                or receipt.compliant is not True
                or type(receipt.ticks) is not int
                or receipt.ticks != 1
            ):
                raise ValueError(
                    "orientation received a failed or noncompliant control"
                )
        except Exception as error:  # noqa: BLE001 -- preserve actual partial prefix
            result.update(
                terminal="execution_failed",
                failure={"type": type(error).__name__, "message": str(error)},
            )
            return OrientationOutcome(result, image)
    result["terminal"] = "completed"
    return OrientationOutcome(result, image)

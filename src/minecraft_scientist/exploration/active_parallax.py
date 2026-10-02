"""Choose prospective anonymous control pairs from provisional public RGB evidence.

Direction cones are sensitivity/spread summaries, not truth confidence intervals.
Expected angular diversity selects an experiment, never certifies navigation.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .keyframe_motion import KeyframeSelection
from .motion_stability import _pose


class DirectionSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handle: int = Field(ge=0, le=3, strict=True)
    first: int = Field(ge=0, strict=True)
    second: int = Field(ge=1, strict=True)
    direction: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    sensitivity_radius_degrees: FiniteFloat = Field(ge=0, le=180)

    @model_validator(mode="after")
    def _valid(self):
        if self.second <= self.first or not np.isclose(
            np.linalg.norm(self.direction), 1, atol=1e-5
        ):
            raise ValueError(
                "direction samples require a genuine interval and unit vector"
            )
        return self


class DirectionHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handle: int = Field(ge=0, le=3, strict=True)
    samples: tuple[DirectionSample, ...]
    center: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    radius_degrees: FiniteFloat = Field(ge=0, le=180)

    @model_validator(mode="after")
    def _valid(self):
        if (
            not self.samples
            or any(item.handle != self.handle for item in self.samples)
            or len({(item.first, item.second) for item in self.samples})
            != len(self.samples)
            or not np.isclose(np.linalg.norm(self.center), 1, atol=1e-5)
        ):
            raise ValueError(
                "provisional direction hypotheses require unique homogeneous samples"
            )
        return self


class ParallaxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_budget: int = Field(default=4, ge=2, le=8, strict=True)
    exploitation_budget: int = Field(default=2, ge=0, le=4, strict=True)
    minimum_samples: int = Field(default=2, ge=2, le=20, strict=True)
    minimum_turn_degrees: FiniteFloat = Field(default=45, gt=0, lt=90)
    maximum_turn_degrees: FiniteFloat = Field(default=135, gt=90, lt=180)

    @model_validator(mode="after")
    def _budget(self):
        if self.exploitation_budget >= self.pair_budget:
            raise ValueError(
                "active acquisition must reserve unknown-control exploration"
            )
        return self


def homogeneous_direction_samples(results: list[dict], timeline: list[dict]):
    """Ignore combined-control intervals and deduplicate shared image evidence."""
    samples, seen = [], set()
    for result in results:
        selected = KeyframeSelection.model_validate(result["selection"]).selected()
        if selected is None:
            continue
        first = result["pair"]["first"]
        second = first + selected.end_offset
        spans = [
            item
            for item in timeline
            if item["start_frame"] < second and item["end_frame"] > first
        ]
        if (
            not spans
            or spans[0]["start_frame"] > first
            or spans[-1]["end_frame"] < second
        ):
            continue
        handles = {
            item["scheduled"]["motion_handle_index"]
            for item in spans
            if item["scheduled"]["kind"] == "move"
        }
        if len(handles) != 1 or any(
            item["scheduled"]["kind"] != "move" for item in spans
        ):
            continue
        handle = next(iter(handles))
        identity = (handle, first, second)
        if identity in seen:
            continue
        seen.add(identity)
        rotation, translation = _pose(selected.motion.candidate)
        direction = -rotation.T @ translation
        sensitivity = selected.motion.sensitivity
        radius = min(
            180.0,
            sensitivity.direction_deviation_95_degrees
            + sensitivity.rotation_deviation_95_degrees,
        )
        samples.append(
            DirectionSample(
                handle=handle,
                first=first,
                second=second,
                direction=tuple(direction),
                sensitivity_radius_degrees=radius,
            )
        )
    return tuple(samples)


def direction_hypotheses(samples: tuple[DirectionSample, ...]):
    hypotheses = []
    if len({(item.handle, item.first, item.second) for item in samples}) != len(
        samples
    ):
        raise ValueError("duplicate image evidence cannot increase directional support")
    for handle in sorted({item.handle for item in samples}):
        group = tuple(item for item in samples if item.handle == handle)
        vectors = np.asarray([item.direction for item in group])
        center = vectors.sum(axis=0)
        length = np.linalg.norm(center)
        if length < 1e-6:
            continue  # Contradictory observations are not a learned direction.
        center /= length
        deviations = np.degrees(np.arccos(np.clip(vectors @ center, -1, 1)))
        radius = min(
            180.0,
            max(
                float(value) + item.sensitivity_radius_degrees
                for value, item in zip(deviations, group, strict=True)
            ),
        )
        hypotheses.append(
            DirectionHypothesis(
                handle=handle,
                samples=group,
                center=tuple(center),
                radius_degrees=radius,
            )
        )
    return tuple(hypotheses)


def propose_control_pairs(
    samples: tuple[DirectionSample, ...], *, policy: ParallaxPolicy | None = None
):
    policy = policy or ParallaxPolicy()
    hypotheses = direction_hypotheses(samples)
    known = {item.handle: item for item in hypotheses}
    ranked = []
    for first in range(4):
        for second in range(4):
            if first == second:
                continue
            left, right = known.get(first), known.get(second)
            observable = left is not None and right is not None
            supported = (
                observable
                and min(len(left.samples), len(right.samples)) >= policy.minimum_samples
            )
            angle, lower, upper, score = None, None, None, 0.0
            overlap = 0.0
            if observable:
                angle = math.degrees(
                    math.acos(
                        float(np.clip(np.asarray(left.center) @ right.center, -1, 1))
                    )
                )
                radius = left.radius_degrees + right.radius_degrees
                lower, upper = max(0.0, angle - radius), min(180.0, angle + radius)
                width = upper - lower
                overlap = max(
                    0.0,
                    min(upper, policy.maximum_turn_degrees)
                    - max(lower, policy.minimum_turn_degrees),
                )
                overlap = (
                    overlap / width
                    if width
                    else float(
                        policy.minimum_turn_degrees
                        <= angle
                        <= policy.maximum_turn_degrees
                    )
                )
                support = min(
                    1.0,
                    min(len(left.samples), len(right.samples)) / policy.minimum_samples,
                )
                score = (
                    math.sin(math.radians(angle)) ** 2
                    * max(0.0, 1 - radius / 180)
                    * overlap
                    * support
                )
            ranked.append(
                {
                    "handles": [first, second],
                    "expected_turn_degrees": angle,
                    "conditional_turn_bounds": None
                    if angle is None
                    else [lower, upper],
                    "information_proxy": score,
                    "conditional_turn_overlap": overlap,
                    "robustly_noncollinear": supported
                    and lower >= policy.minimum_turn_degrees
                    and upper <= policy.maximum_turn_degrees,
                    "source_samples": [
                        0 if left is None else len(left.samples),
                        0 if right is None else len(right.samples),
                    ],
                }
            )
    ranked.sort(key=lambda item: (-item["information_proxy"], item["handles"]))
    chosen, selections = [], []
    exposures = dict.fromkeys(range(4), 0)

    def select(item, role):
        chosen.append(item["handles"])
        selections.append({"handles": item["handles"], "role": role})
        for handle in item["handles"]:
            exposures[handle] += 1

    # A tiny positive sin-squared score must not exploit a near-opposite pair.
    for item in ranked:
        if len(chosen) == policy.exploitation_budget:
            break
        if item["robustly_noncollinear"]:
            select(item, "conditional_exploitation")
    # A single sample is a useful experimental lead, never accepted motion.
    for item in ranked:
        if len(chosen) == policy.exploitation_budget:
            break
        if item["handles"] not in chosen and item["information_proxy"] > 0:
            select(item, "provisional_direction_resampling")

    def exploration_priority(item):
        first, second = item["handles"]
        unknown = [handle for handle in (first, second) if handle not in known]
        anchors = [known[handle] for handle in (first, second) if handle in known]
        supported_anchors = [
            anchor
            for anchor in anchors
            if len(anchor.samples) >= policy.minimum_samples
        ]
        anchor_strength = max(
            (
                (1 - anchor.radius_degrees / 180)
                * min(1.0, len(anchor.samples) / policy.minimum_samples)
                for anchor in anchors
            ),
            default=0.0,
        )
        # Unknown-vs-known controls discriminate a direction; unknown-vs-unknown
        # can accidentally be antipodal and has no calibrated directional anchor.
        anchor_class = (
            2 if unknown and supported_anchors else 1 if unknown and anchors else 0
        )
        return (
            -anchor_class,
            min((exposures[handle] for handle in unknown), default=policy.pair_budget),
            -anchor_strength,
            sum(item["source_samples"]),
            sum(exposures[handle] for handle in (first, second)),
            item["handles"],
        )

    while len(chosen) < policy.pair_budget:
        remaining = [item for item in ranked if item["handles"] not in chosen]
        item = min(remaining, key=exploration_priority)
        anchored = any(handle not in known for handle in item["handles"]) and any(
            handle in known and len(known[handle].samples) >= policy.minimum_samples
            for handle in item["handles"]
        )
        provisional_anchor = any(
            handle not in known for handle in item["handles"]
        ) and any(handle in known for handle in item["handles"])
        select(
            item,
            "anchored_unknown_exploration"
            if anchored
            else "provisional_anchor_resampling"
            if provisional_anchor
            else "coverage_exploration",
        )
    return {
        "selector_version": "observability-anchored-v3",
        "policy": policy.model_dump(mode="json"),
        "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
        "ranked_pairs": ranked,
        "handle_pairs": chosen,
        "selections": selections,
        "scope": "prospective experimental choice from provisional RGB directions; not true motion or navigation acceptance",
    }

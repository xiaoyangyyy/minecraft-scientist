"""Chronological RGB-prefix selection, without truth or future-based reselection.

The first supported prefix is latched. Later windows are retained as diagnostic
evidence, never used to silently replace that selection. A local keyframe pose
does not describe the terminal pose of a longer action or certify goal arrival.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .multiview_motion import (
    MultiViewMotionConfig,
    MultiViewMotionEvidence,
    estimate_multiview_motion,
)
from .temporal_tracks import TemporalTrackConfig, TemporalTracks, track_frame_chain


class KeyframeMotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_offsets: tuple[StrictInt, ...] = (2, 4, 6, 8, 12, 18, 24, 31)
    temporal: TemporalTrackConfig = TemporalTrackConfig()
    multiview: MultiViewMotionConfig = MultiViewMotionConfig()

    @model_validator(mode="after")
    def _bounded(self):
        if (
            not self.candidate_offsets
            or len(self.candidate_offsets) > 16
            or any(type(value) is not int for value in self.candidate_offsets)
            or tuple(sorted(set(self.candidate_offsets))) != self.candidate_offsets
            or self.candidate_offsets[0] < 2
            or self.candidate_offsets[-1] >= self.temporal.maximum_frames
        ):
            raise ValueError(
                "keyframe offsets must be sorted, unique and within the temporal budget"
            )
        return self

    def offsets(self, terminal_offset: int) -> tuple[int, ...]:
        if (
            type(terminal_offset) is not int
            or not 0 <= terminal_offset < self.temporal.maximum_frames
        ):
            raise ValueError("terminal keyframe offset must be within the image budget")
        return tuple(
            sorted(
                {value for value in self.candidate_offsets if value <= terminal_offset}
                | {terminal_offset}
            )
        )


class KeyframeAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    end_offset: int = Field(ge=0, strict=True)
    tracks: TemporalTracks
    motion: MultiViewMotionEvidence

    @model_validator(mode="after")
    def _interval(self):
        if self.tracks.frame_count != max(2, self.end_offset + 1):
            raise ValueError("keyframe tracks must contain every prefix frame")
        if self.motion.candidate.matches != len(self.tracks.track_ids):
            raise ValueError("keyframe motion and its track evidence must align")
        return self


class KeyframeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    terminal_offset: int = Field(ge=0, strict=True)
    attempts: tuple[KeyframeAttempt, ...]
    selected_offset: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def _first_supported(self):
        offsets = tuple(item.end_offset for item in self.attempts)
        if (
            not offsets
            or offsets != tuple(sorted(set(offsets)))
            or offsets[-1] != self.terminal_offset
        ):
            raise ValueError(
                "keyframe attempts must be chronological and include the terminal baseline"
            )
        supported = [item.end_offset for item in self.attempts if item.motion.supported]
        if self.selected_offset != (supported[0] if supported else None):
            raise ValueError("keyframe selection must latch the first supported prefix")
        return self

    def selected(self) -> KeyframeAttempt | None:
        return next(
            (item for item in self.attempts if item.end_offset == self.selected_offset),
            None,
        )


def infer_keyframe_prefixes(
    frames: Sequence[np.ndarray],
    camera: np.ndarray,
    *,
    config: KeyframeMotionConfig | None = None,
    seed: int = 0,
) -> KeyframeSelection:
    """Retain a preregistered prefix sweep; inference of each prefix is past-only.

    Every candidate starts at the same real anchor and includes all intermediate
    frames. The terminal candidate is an exact baseline, not a fabricated pose
    transported from a shorter window. No rejected identities are reseeded inside
    a track chain. Candidate seeds are identical, avoiding a seed-search ablation.
    """
    config = config or KeyframeMotionConfig()
    if (
        type(seed) is not int
        or not 0 <= seed < 2**31 - config.multiview.stability.trials
    ):
        raise ValueError("keyframe seed must leave room for signed sensitivity seeds")
    if not 1 <= len(frames) <= config.temporal.maximum_frames:
        raise ValueError("keyframe input must be a nonempty bounded image sequence")
    terminal = len(frames) - 1
    attempts, selected = [], None
    for offset in config.offsets(terminal):
        prefix = frames[: offset + 1] if offset else (frames[0], frames[0])
        tracks = track_frame_chain(prefix, config=config.temporal)
        motion = estimate_multiview_motion(
            tracks, camera, config=config.multiview, seed=seed
        )
        attempts.append(
            KeyframeAttempt(end_offset=offset, tracks=tracks, motion=motion)
        )
        if selected is None and motion.supported:
            selected = offset
    return KeyframeSelection(
        terminal_offset=terminal, attempts=tuple(attempts), selected_offset=selected
    )

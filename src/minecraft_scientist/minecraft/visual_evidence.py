"""Conservative pixel change evidence; does not infer semantic action success."""
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PixelChangeEvidence:
    control_mae: float
    intervention_mae: float
    control_stable: bool
    changed: bool | None


@dataclass(frozen=True)
class EquipmentAudit:
    """Evaluator-only check that a fixture held its assigned intervention."""

    expected: str
    observed: tuple[str | None, ...]
    verified_from_frame: int
    valid: bool


def verify_equipment_trace(
    observed: Iterable[str | None], *, expected: str, verified_from_frame: int = 2
) -> EquipmentAudit:
    """Fail closed when equipment telemetry is missing or changes mid-probe.

    Equipment labels are privileged fixture telemetry, never Agent observations.
    Initial reset frames may precede the fixture's final equipment command.
    """
    if not expected or verified_from_frame < 0:
        raise ValueError("expected equipment and nonnegative frame index are required")
    trace = tuple(observed)
    checked = trace[verified_from_frame:]
    allowed = (
        {"air", "none", "minecraft:air"}
        if expected == "air"
        else {expected, "minecraft:" + expected}
    )
    return EquipmentAudit(
        expected=expected,
        observed=trace,
        verified_from_frame=verified_from_frame,
        valid=bool(checked) and all(value in allowed for value in checked),
    )


def compare_regions(control, before, after, *, stability_limit=1.0, change_limit=10.0):
    """Return unknown when the no-action control is unstable.

    Caller supplies matched, fixed-camera regions. Thresholds must be calibrated
    independently for other cameras/fixtures. A change is not proof of removal.
    """
    arrays = [np.asarray(frame, dtype=float) for frame in (control, before, after)]
    if not arrays[0].size or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("regions must have the same nonempty shape")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("regions must contain finite values")
    if not 0 <= stability_limit < change_limit:
        raise ValueError("thresholds require 0 <= stability_limit < change_limit")
    control_mae = float(np.abs(arrays[1] - arrays[0]).mean())
    intervention_mae = float(np.abs(arrays[2] - arrays[1]).mean())
    stable = control_mae <= stability_limit
    return PixelChangeEvidence(control_mae, intervention_mae, stable,
                               intervention_mae >= change_limit if stable else None)

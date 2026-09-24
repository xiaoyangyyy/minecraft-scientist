"""Conservative pixel change evidence; does not infer semantic action success."""
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def layout_landmark_signature(features: dict) -> tuple[int, ...]:
    """Quantize a route view while ignoring peripheral color false positives.

    The caller supplies only Agent-visible image features. A colored component is
    treated as layout-defining only while it occupies the central corridor; this
    prevents unrelated edge pixels seen later in a route from changing the goal
    landmark. No metric world position or semantic object label is consumed.
    """

    marker = features.get("marker")
    if not isinstance(marker, dict):
        raise TypeError("route landmark requires visible marker geometry")
    central_pixels = int(features.get("central_obstacle_pixels", 0))
    obstacle = features.get("obstacle")
    layout_visible = isinstance(obstacle, dict) and central_pixels >= 200
    try:
        return (
            int(layout_visible),
            int(obstacle["centroid_x"] // 8) if layout_visible else -1,
            min(12, central_pixels // 200),
            int(marker["centroid_x"] // 8),
            int(marker["pixel_count"] // 500),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("route landmark features are incomplete or invalid") from error


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


@dataclass(frozen=True)
class VisualCellEvidence:
    object_id: str
    row: int
    column: int
    control_mae: float
    intervention_mae: float


@dataclass(frozen=True)
class VisualChangeRegion:
    object_id: str
    cell_ids: tuple[str, ...]
    top: int
    left: int
    bottom: int
    right: int
    total_intervention_mae: float


@dataclass(frozen=True)
class GridChangeEvidence:
    """Anonymous spatial grounding for a controlled pixel intervention."""

    evidence: PixelChangeEvidence
    selected_cell_id: str | None
    cells: tuple[VisualCellEvidence, ...]
    selected_region_id: str | None = None
    regions: tuple[VisualChangeRegion, ...] = ()


@dataclass(frozen=True)
class VisualRegionGeometry:
    """Geometry of one image-derived region, expressed without a class label."""

    pixel_count: int
    centroid_x: float
    centroid_y: float
    left: int
    top: int
    right: int
    bottom: int


def visual_region_iou(
    reference: VisualChangeRegion, candidate: VisualChangeRegion
) -> float:
    """Return grid-cell intersection-over-union for two anonymous regions."""

    left, right = set(reference.cell_ids), set(candidate.cell_ids)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_grounding_evidence(path: Path) -> GridChangeEvidence:
    """Load and validate the Agent-visible grounding artifact used by discovery."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = PixelChangeEvidence(**payload["evidence"])
    cells = tuple(VisualCellEvidence(**cell) for cell in payload["cells"])
    selected = payload.get("selected_cell_id")
    regions = tuple(
        VisualChangeRegion(
            **{**region, "cell_ids": tuple(region["cell_ids"])}
        )
        for region in payload.get("regions", ())
    )
    selected_region = payload.get("selected_region_id")
    ids = {cell.object_id for cell in cells}
    if not cells or len(ids) != len(cells) or any(
        not object_id.startswith("obj_") for object_id in ids
    ):
        raise ValueError("grounding cells require unique opaque identifiers")
    if evidence.changed is True and selected not in ids:
        raise ValueError("changed grounding requires a selected cell")
    if evidence.changed is not True and selected is not None:
        raise ValueError("non-change grounding cannot select a cell")
    region_ids = {region.object_id for region in regions}
    if len(region_ids) != len(regions) or any(
        not object_id.startswith("obj_") for object_id in region_ids
    ):
        raise ValueError("grounding regions require unique opaque identifiers")
    if any(not region.cell_ids or not set(region.cell_ids) <= ids for region in regions):
        raise ValueError("grounding region references unknown cells")
    if payload.get("schema_version", 1) >= 2:
        if evidence.changed is True and selected_region not in region_ids:
            raise ValueError("changed grounding requires a selected region")
        if evidence.changed is not True and selected_region is not None:
            raise ValueError("non-change grounding cannot select a region")
    return GridChangeEvidence(evidence, selected, cells, selected_region, regions)


def _change_regions(
    cells: tuple[VisualCellEvidence, ...], changed_ids: set[str]
) -> tuple[VisualChangeRegion, ...]:
    by_coordinate = {(cell.row, cell.column): cell for cell in cells}
    remaining = {
        coordinate for coordinate, cell in by_coordinate.items()
        if cell.object_id in changed_ids
    }
    regions: list[VisualChangeRegion] = []
    while remaining:
        pending = [min(remaining)]
        component: list[VisualCellEvidence] = []
        while pending:
            coordinate = pending.pop()
            if coordinate not in remaining:
                continue
            remaining.remove(coordinate)
            component.append(by_coordinate[coordinate])
            row, column = coordinate
            pending.extend(
                neighbor for neighbor in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                )
                if neighbor in remaining
            )
        component.sort(key=lambda cell: (cell.row, cell.column))
        cell_ids = tuple(sorted(cell.object_id for cell in component))
        raw = "visual-region:" + ":".join(cell_ids)
        regions.append(VisualChangeRegion(
            object_id="obj_" + hashlib.blake2s(
                raw.encode(), digest_size=8
            ).hexdigest(),
            cell_ids=cell_ids,
            top=min(cell.row for cell in component),
            left=min(cell.column for cell in component),
            bottom=max(cell.row for cell in component),
            right=max(cell.column for cell in component),
            total_intervention_mae=sum(
                cell.intervention_mae for cell in component
            ),
        ))
    return tuple(sorted(regions, key=lambda region: region.object_id))


def hud_safe_view(frame: np.ndarray) -> np.ndarray:
    """Remove common HUD/toast bands while retaining the central world view."""

    array = np.asarray(frame)
    if array.ndim not in {2, 3} or array.shape[0] < 96 or array.shape[1] < 96:
        raise ValueError("frame must be a 2D or 3D image of at least 96x96")
    height, width = array.shape[:2]
    top = height * 1 // 7
    bottom = height * 11 // 14
    left = width * 1 // 14
    right = width * 5 // 7
    view = array[top:bottom, left:right]
    if not view.size:
        raise ValueError("HUD-safe view is empty")
    return view


def central_view_salience(frame: np.ndarray) -> float:
    """Return a label-free brightness score around the camera crosshair.

    The vertical middle half excludes most HUD bands. The narrower, symmetric
    horizontal band remains centered in the original camera frame; callers must
    not pass the asymmetric HUD-safe localization crop.
    """

    array = np.asarray(frame, dtype=float)
    if array.ndim not in {2, 3} or not array.size or not np.isfinite(array).all():
        raise ValueError("view must be a finite, nonempty 2D or 3D image")
    height, width = array.shape[:2]
    if height < 4 or width < 4:
        raise ValueError("view must be at least 4x4")
    center = array[
        height // 4:height * 3 // 4,
        width * 3 // 8:width * 5 // 8,
    ]
    return float(center.mean())


def largest_achromatic_region(
    frame: np.ndarray,
    *,
    luminance_floor: float = 100.0,
    chroma_ceiling: float = 18.0,
    top_fraction: float = 0.08,
    bottom_fraction: float = 0.62,
    side_fraction: float = 0.12,
) -> VisualRegionGeometry | None:
    """Locate the largest bright, low-chroma component in the world view.

    This is deliberately a label-free fixture primitive: it has no knowledge of
    blocks, goals, or movement directions.  Cropping removes the HUD and extreme
    image edges, while connected-component selection prevents small bright scenery
    fragments from being merged into the returned geometry.
    """

    array = np.asarray(frame, dtype=float)
    if array.ndim != 3 or array.shape[2] < 3 or not array.size:
        raise ValueError("frame must be a nonempty RGB-like image")
    if not np.isfinite(array).all():
        raise ValueError("frame must contain finite values")
    if luminance_floor < 0 or chroma_ceiling < 0:
        raise ValueError("color thresholds must be nonnegative")
    if not 0 <= top_fraction < bottom_fraction <= 1:
        raise ValueError("vertical crop fractions are invalid")
    if not 0 <= side_fraction < 0.5:
        raise ValueError("side fraction must be in [0, 0.5)")

    height, width = array.shape[:2]
    top = int(height * top_fraction)
    bottom = int(height * bottom_fraction)
    left = int(width * side_fraction)
    right = int(width * (1 - side_fraction))
    rgb = array[:, :, :3]
    luminance = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mask = (luminance > luminance_floor) & (chroma < chroma_ceiling)
    cropped = np.zeros_like(mask)
    cropped[top:bottom, left:right] = mask[top:bottom, left:right]

    seen = np.zeros_like(cropped, dtype=bool)
    largest: list[tuple[int, int]] = []
    for seed_y, seed_x in zip(*np.where(cropped), strict=True):
        if seen[seed_y, seed_x]:
            continue
        seen[seed_y, seed_x] = True
        pending = [(int(seed_y), int(seed_x))]
        component: list[tuple[int, int]] = []
        while pending:
            y, x = pending.pop()
            component.append((y, x))
            for candidate_y, candidate_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= candidate_y < height
                    and 0 <= candidate_x < width
                    and cropped[candidate_y, candidate_x]
                    and not seen[candidate_y, candidate_x]
                ):
                    seen[candidate_y, candidate_x] = True
                    pending.append((candidate_y, candidate_x))
        if len(component) > len(largest):
            largest = component

    if not largest:
        return None
    ys = np.fromiter((item[0] for item in largest), dtype=float)
    xs = np.fromiter((item[1] for item in largest), dtype=float)
    return VisualRegionGeometry(
        pixel_count=len(largest),
        centroid_x=float(xs.mean()),
        centroid_y=float(ys.mean()),
        left=int(xs.min()),
        top=int(ys.min()),
        right=int(xs.max()),
        bottom=int(ys.max()),
    )


def localize_change_grid(
    control: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    rows: int = 4,
    columns: int = 4,
    stability_limit: float = 1.0,
    change_limit: float = 10.0,
) -> GridChangeEvidence:
    """Ground the strongest stable intervention change to an opaque grid cell.

    The function fails closed when a threshold-scale change occurs in a cell whose
    control is itself unstable. Low-amplitude drift elsewhere cannot veto a strong
    change in a separately stable cell. Coordinates are retained only as
    nonsemantic provenance; discovery consumes the opaque cell identifier.
    """

    arrays = [np.asarray(frame, dtype=float) for frame in (control, before, after)]
    if not arrays[0].size or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("regions must have the same nonempty shape")
    if any(array.ndim not in {2, 3} for array in arrays):
        raise ValueError("regions must be 2D or 3D images")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("regions must contain finite values")
    if rows < 1 or columns < 1:
        raise ValueError("grid dimensions must be positive")
    height, width = arrays[0].shape[:2]
    if rows > height or columns > width:
        raise ValueError("grid dimensions cannot exceed image dimensions")
    if not 0 <= stability_limit < change_limit:
        raise ValueError("thresholds require 0 <= stability_limit < change_limit")

    cells: list[VisualCellEvidence] = []
    for row in range(rows):
        top, bottom = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            left, right = column * width // columns, (column + 1) * width // columns
            control_mae = float(
                np.abs(
                    arrays[1][top:bottom, left:right]
                    - arrays[0][top:bottom, left:right]
                ).mean()
            )
            intervention_mae = float(
                np.abs(
                    arrays[2][top:bottom, left:right]
                    - arrays[1][top:bottom, left:right]
                ).mean()
            )
            raw = f"visual-change:{rows}:{columns}:{row}:{column}"
            cells.append(
                VisualCellEvidence(
                    object_id="obj_" + hashlib.blake2s(
                        raw.encode(), digest_size=8
                    ).hexdigest(),
                    row=row,
                    column=column,
                    control_mae=control_mae,
                    intervention_mae=intervention_mae,
                )
            )

    stable_cells = tuple(
        cell for cell in cells if cell.control_mae <= stability_limit
    )
    strongest = (
        max(stable_cells, key=lambda cell: cell.intervention_mae)
        if stable_cells
        else min(cells, key=lambda cell: cell.control_mae)
    )
    changed = (
        strongest.intervention_mae >= change_limit if stable_cells else None
    )
    confounded = any(
        cell.control_mae > stability_limit
        and cell.intervention_mae >= change_limit
        for cell in cells
    )
    if changed is False and confounded:
        changed = None
    changed_ids = {
        cell.object_id for cell in stable_cells
        if cell.intervention_mae >= change_limit
    } if changed is True else set()
    regions = _change_regions(tuple(cells), changed_ids)
    selected_region = next(
        (region for region in regions if strongest.object_id in region.cell_ids),
        None,
    )
    return GridChangeEvidence(
        evidence=PixelChangeEvidence(
            control_mae=strongest.control_mae,
            intervention_mae=strongest.intervention_mae,
            control_stable=changed is not None,
            changed=changed,
        ),
        selected_cell_id=strongest.object_id if changed is True else None,
        cells=tuple(cells),
        selected_region_id=(selected_region.object_id if selected_region else None),
        regions=regions,
    )


def successful_select_count(path: Path) -> int:
    """Count only fully compliant, successful SELECT transitions in a JSONL log."""

    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if (
            event.get("executed_action", {}).get("kind") == "select"
            and event.get("intervention_compliance") is True
            and event.get("event", {}).get("success") is True
        ):
            count += 1
    return count


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

"""Fit and validate a per-landmark visual match radius from real RGB runs."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from visual_route_crisis_benchmark import digest, load, route_program, save

from minecraft_scientist.cognitive_map import VisualTopologicalMap
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature

ROOT = Path(__file__).resolve().parents[2]


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def trajectory_signature(directory: Path, index: int) -> tuple[int, ...]:
    trajectory = load(directory / "trajectory.json")
    return layout_landmark_signature(trajectory[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crisis-run",
        type=Path,
        default=ROOT / "runs/visual-route-crisis-seed140-map1400",
    )
    parser.add_argument(
        "--composition-run",
        type=Path,
        default=ROOT / "runs/multi-stage-route-seed143-map1415",
    )
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/adaptive-landmark-calibration",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    crisis_audit_path = args.crisis_run / "integrity-audit-v3.json"
    if not crisis_audit_path.exists():
        crisis_audit_path = args.crisis_run / "integrity-audit-v2.json"
    composition_audit_path = args.composition_run / "integrity-audit.json"
    crisis_audit = load(crisis_audit_path)
    composition_audit = load(composition_audit_path)
    if crisis_audit.get("valid") is not True:
        raise RuntimeError("route-crisis source audit is not valid")
    if composition_audit.get("valid") is not True:
        raise RuntimeError("composition source audit is not valid")

    evidence_sources = {
        "train-positive-0": args.crisis_run / "source-0",
        "train-positive-1": args.crisis_run / "source-1",
        "train-negative": args.crisis_run / "repair-source",
        "heldout-positive": args.composition_run / "replay",
        "heldout-negative": args.crisis_run / "shadow-validation",
    }
    preregistration = {
        "crisis_run_name": args.crisis_run.name,
        "composition_run_name": args.composition_run.name,
        "crisis_audit_sha256": digest(crisis_audit_path),
        "composition_audit_sha256": digest(composition_audit_path),
        "positive_training_evidence": ["train-positive-0", "train-positive-1"],
        "negative_training_evidence": ["train-negative"],
        "positive_heldout_evidence": ["heldout-positive"],
        "negative_heldout_evidence": ["heldout-negative"],
        "global_tolerance": 8.25,
        "calibration_margin": args.margin,
        "policy": (
            "fit radius as maximum positive distance plus a preregistered margin; "
            "require that radius to remain at least one margin below the nearest "
            "negative; use no held-out frame while fitting"
        ),
        "required_positive_acceptance": 1.0,
        "required_negative_rejection": 1.0,
    }
    save(args.output / "preregistered.json", preregistration)

    source_files = (
        "environments/minestudio/adaptive_landmark_calibration_benchmark.py",
        "environments/minestudio/audit_adaptive_landmark_calibration.py",
        "environments/minestudio/audit_visual_route_crisis.py",
        "environments/minestudio/audit_multi_stage_route_composition.py",
        "src/minecraft_scientist/cognitive_map/visual_topology.py",
        "src/minecraft_scientist/minecraft/visual_evidence.py",
    )
    source_hashes = {name: digest(ROOT / name) for name in source_files}
    save(args.output / "runtime-manifest.json", {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("networkx", "numpy", "pydantic", "pillow")
        },
        "source_sha256": source_hashes,
    })
    snapshot = args.output / "source-snapshot"
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    upstream = args.output / "upstream"
    upstream.mkdir()
    shutil.copy2(crisis_audit_path, upstream / "crisis-integrity-audit.json")
    shutil.copy2(
        composition_audit_path,
        upstream / "composition-integrity-audit.json",
    )
    evidence_root = args.output / "evidence"
    evidence_manifest = {}
    for name, source in evidence_sources.items():
        destination = evidence_root / name
        shutil.copytree(source, destination)
        evidence_manifest[name] = tree_manifest(destination)
    save(args.output / "evidence-manifest.json", evidence_manifest)

    positives = tuple(
        trajectory_signature(evidence_root / name / "route", 0)
        for name in preregistration["positive_training_evidence"]
    )
    negatives = tuple(
        trajectory_signature(evidence_root / name / "route", 0)
        for name in preregistration["negative_training_evidence"]
    )
    heldout_positives = tuple(
        trajectory_signature(evidence_root / name / "replay", 0)
        for name in preregistration["positive_heldout_evidence"]
    )
    heldout_negatives = tuple(
        trajectory_signature(evidence_root / name / "replay", 0)
        for name in preregistration["negative_heldout_evidence"]
    )
    source_trajectory = load(evidence_root / "train-positive-0/route/trajectory.json")
    anchor = layout_landmark_signature(source_trajectory[0])
    target = layout_landmark_signature(source_trajectory[-1])
    program = route_program(evidence_root / "train-positive-0")

    baseline = VisualTopologicalMap(
        match_tolerance=preregistration["global_tolerance"],
    )
    baseline_route_id = baseline.record_route(
        anchor,
        target,
        program,
        success=True,
        cost=len(program),
    )
    target_id = baseline.node_id_for(target)
    baseline_false_accepts = [
        baseline.plan(signature, target_id) is not None
        for signature in heldout_negatives
    ]
    if not all(baseline_false_accepts):
        raise RuntimeError("broad fixed tolerance did not reproduce the alias")

    calibrated = VisualTopologicalMap(
        match_tolerance=preregistration["global_tolerance"],
    )
    route_id = calibrated.record_route(
        anchor,
        target,
        program,
        success=True,
        cost=len(program),
    )
    if route_id != baseline_route_id:
        raise RuntimeError("calibration unexpectedly changed route identity")
    source_id = calibrated.node_id_for(anchor)
    landmark = calibrated.calibrate_landmark(
        source_id,
        positives,
        negatives,
        margin=args.margin,
    )
    if landmark.calibration is None:
        raise RuntimeError("landmark calibration was not persisted")
    positive_acceptance = [
        calibrated.plan(signature, target_id) is not None
        for signature in heldout_positives
    ]
    negative_rejection = [
        calibrated.plan(signature, target_id) is None
        for signature in heldout_negatives
    ]
    between_signature = list(anchor)
    between_signature[1] = (anchor[1] + negatives[0][1]) // 2
    between_signature = tuple(between_signature)
    between_rejected = calibrated.plan(between_signature, target_id) is None
    if sum(positive_acceptance) / len(positive_acceptance) != 1.0:
        raise RuntimeError("calibrated matcher rejected a held-out positive")
    if sum(negative_rejection) / len(negative_rejection) != 1.0:
        raise RuntimeError("calibrated matcher accepted a held-out negative")
    if not between_rejected:
        raise RuntimeError("calibrated matcher accepted an intermediate control")

    map_path = args.output / "visual-topology.json"
    calibrated.save(map_path)
    report = {
        "preregistration_sha256": digest(args.output / "preregistered.json"),
        "runtime_manifest_sha256": digest(args.output / "runtime-manifest.json"),
        "evidence_manifest_sha256": digest(args.output / "evidence-manifest.json"),
        "crisis_audit_sha256": digest(
            upstream / "crisis-integrity-audit.json"
        ),
        "composition_audit_sha256": digest(
            upstream / "composition-integrity-audit.json"
        ),
        "map_sha256": digest(map_path),
        "map_format_version": calibrated.FORMAT_VERSION,
        "route_id": route_id,
        "source_id": source_id,
        "target_id": target_id,
        "program": program,
        "anchor_signature": anchor,
        "target_signature": target,
        "positive_training_signatures": positives,
        "negative_training_signatures": negatives,
        "heldout_positive_signatures": heldout_positives,
        "heldout_negative_signatures": heldout_negatives,
        "between_signature": between_signature,
        "calibration": landmark.calibration.model_dump(mode="json"),
        "baseline_false_acceptance_rate": sum(baseline_false_accepts)
        / len(baseline_false_accepts),
        "heldout_positive_acceptance_rate": sum(positive_acceptance)
        / len(positive_acceptance),
        "heldout_negative_rejection_rate": sum(negative_rejection)
        / len(negative_rejection),
        "between_signature_rejected": between_rejected,
        "global_tolerance_overridden": (
            landmark.calibration.radius
            < preregistration["global_tolerance"]
        ),
        "scope": (
            "Per-landmark radius fitted from audited real MineStudio initial RGB "
            "signatures with disjoint train/held-out episode roles. It calibrates "
            "one controlled fixture landmark, not a general visual metric learner."
        ),
    }
    save(args.output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

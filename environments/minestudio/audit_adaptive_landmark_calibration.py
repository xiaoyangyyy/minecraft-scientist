"""Independently audit per-landmark radius fitting on real visual evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_visual_route_crisis import digest, load, recompute_trajectory

from minecraft_scientist.cognitive_map import VisualTopologicalMap
from minecraft_scientist.minecraft.visual_evidence import layout_landmark_signature


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return math.sqrt(sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    ))


def evidence_signature(root: Path, name: str, subdirectory: str) -> tuple[int, ...]:
    trajectory = recompute_trajectory(root / "evidence" / name / subdirectory)
    return layout_landmark_signature(trajectory[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run
    prereg = load(root / "preregistered.json")
    report = load(root / "report.json")
    runtime = load(root / "runtime-manifest.json")
    evidence_manifest = load(root / "evidence-manifest.json")
    crisis_audit = load(root / "upstream/crisis-integrity-audit.json")
    composition_audit = load(root / "upstream/composition-integrity-audit.json")

    snapshot_checks = []
    for relative, expected_hash in runtime["source_sha256"].items():
        path = root / "source-snapshot" / relative
        actual = digest(path) if path.exists() else None
        snapshot_checks.append({
            "path": relative,
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "valid": actual == expected_hash,
        })
    evidence_checks = {
        name: tree_manifest(root / "evidence" / name) == expected
        for name, expected in evidence_manifest.items()
    }

    positives = tuple(
        evidence_signature(root, name, "route")
        for name in prereg["positive_training_evidence"]
    )
    negatives = tuple(
        evidence_signature(root, name, "route")
        for name in prereg["negative_training_evidence"]
    )
    heldout_positives = tuple(
        evidence_signature(root, name, "replay")
        for name in prereg["positive_heldout_evidence"]
    )
    heldout_negatives = tuple(
        evidence_signature(root, name, "replay")
        for name in prereg["negative_heldout_evidence"]
    )
    anchor = tuple(report["anchor_signature"])
    target = tuple(report["target_signature"])
    margin = float(prereg["calibration_margin"])
    positive_distances = tuple(distance(anchor, item) for item in positives)
    negative_distances = tuple(distance(anchor, item) for item in negatives)
    maximum_positive = max(positive_distances)
    nearest_negative = min(negative_distances)
    expected_radius = maximum_positive + margin

    topology = VisualTopologicalMap.load(root / "visual-topology.json")
    landmark = next(
        item for item in topology.landmarks()
        if item.node_id == report["source_id"]
    )
    calibration = landmark.calibration
    if calibration is None:
        raise RuntimeError("saved source landmark is not calibrated")
    target_id = report["target_id"]
    positive_results = [
        topology.plan(signature, target_id) is not None
        for signature in heldout_positives
    ]
    negative_results = [
        topology.plan(signature, target_id) is None
        for signature in heldout_negatives
    ]
    between_signature = tuple(report["between_signature"])

    baseline = VisualTopologicalMap(
        match_tolerance=prereg["global_tolerance"],
    )
    baseline_route = baseline.record_route(
        anchor,
        target,
        tuple(report["program"]),
        success=True,
        cost=len(report["program"]),
    )
    baseline_results = [
        baseline.plan(signature, baseline.node_id_for(target)) is not None
        for signature in heldout_negatives
    ]
    checks = {
        "training_signatures_recomputed": (
            [list(item) for item in positives]
            == report["positive_training_signatures"]
            and [list(item) for item in negatives]
            == report["negative_training_signatures"]
        ),
        "heldout_signatures_recomputed": (
            [list(item) for item in heldout_positives]
            == report["heldout_positive_signatures"]
            and [list(item) for item in heldout_negatives]
            == report["heldout_negative_signatures"]
        ),
        "calibration_statistics_recomputed": (
            calibration.positive_samples == len(positives)
            and calibration.negative_samples == len(negatives)
            and calibration.max_positive_distance == maximum_positive
            and calibration.nearest_negative_distance == nearest_negative
            and calibration.margin == margin
            and calibration.radius == expected_radius
            and calibration.model_dump(mode="json") == report["calibration"]
        ),
        "separation_constraint_recomputed": (
            expected_radius < nearest_negative - margin
        ),
        "broad_fixed_tolerance_false_accepts_negative": (
            all(baseline_results)
            and report["baseline_false_acceptance_rate"] == 1.0
            and baseline_route == report["route_id"]
        ),
        "heldout_positive_accepted": all(positive_results)
        and report["heldout_positive_acceptance_rate"] == 1.0,
        "heldout_negative_rejected": all(negative_results)
        and report["heldout_negative_rejection_rate"] == 1.0,
        "between_signature_rejected": (
            topology.plan(between_signature, target_id) is None
            and report["between_signature_rejected"] is True
        ),
        "global_tolerance_overridden": (
            calibration.radius < prereg["global_tolerance"]
            and report["global_tolerance_overridden"] is True
        ),
        "map_format_and_hash_match": (
            report["map_format_version"] == VisualTopologicalMap.FORMAT_VERSION
            and digest(root / "visual-topology.json") == report["map_sha256"]
        ),
    }
    audit = {
        "run": str(root),
        "auditor_sha256": digest(Path(__file__)),
        "preregistration_intact": digest(root / "preregistered.json")
        == report["preregistration_sha256"],
        "runtime_manifest_intact": digest(root / "runtime-manifest.json")
        == report["runtime_manifest_sha256"],
        "evidence_manifest_intact": digest(root / "evidence-manifest.json")
        == report["evidence_manifest_sha256"],
        "upstream_audits_intact_and_valid": (
            crisis_audit.get("valid") is True
            and composition_audit.get("valid") is True
            and digest(root / "upstream/crisis-integrity-audit.json")
            == report["crisis_audit_sha256"]
            == prereg["crisis_audit_sha256"]
            and digest(root / "upstream/composition-integrity-audit.json")
            == report["composition_audit_sha256"]
            == prereg["composition_audit_sha256"]
        ),
        "source_snapshot": snapshot_checks,
        "source_snapshot_complete": bool(snapshot_checks)
        and all(item["valid"] for item in snapshot_checks),
        "evidence_artifacts": evidence_checks,
        "evidence_artifacts_intact": all(evidence_checks.values()),
        "checks": checks,
        "checks_valid": all(value is True for value in checks.values()),
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["runtime_manifest_intact"],
        audit["evidence_manifest_intact"],
        audit["upstream_audits_intact_and_valid"],
        audit["source_snapshot_complete"],
        audit["evidence_artifacts_intact"],
        audit["checks_valid"],
    ))
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

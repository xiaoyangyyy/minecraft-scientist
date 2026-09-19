"""Post-run integrity audit for an active visual discovery artifact directory."""
import argparse
import hashlib
import json
from pathlib import Path

from minecraft_scientist.minecraft.visual_evidence import (
    load_grounding_evidence,
    successful_select_count,
    visual_region_iou,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    preregistered = root / "preregistered.json"
    report_path = root / "report.json"
    if not preregistered.is_file() or not report_path.is_file():
        parser.error("run must contain preregistered.json and report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = report.get("preregistration_sha256")
    actual = digest(preregistered)
    manifest_path = root / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hashes = manifest.get("source_sha256", {})
    snapshot_root = root / "source-snapshot"
    source_snapshot = []
    for relative, expected_hash in sorted(source_hashes.items()):
        snapshot_path = snapshot_root / relative
        actual_hash = digest(snapshot_path) if snapshot_path.is_file() else None
        source_snapshot.append({
            "path": relative,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "valid": isinstance(expected_hash, str) and expected_hash == actual_hash,
        })
    equipment = []
    discovery_regions = []
    validation_regions = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        path = directory / "equipment-audit.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        selects = successful_select_count(directory / "events.jsonl")
        probe_report_path = directory / "report.json"
        grounding_path = directory / "grounding.json"
        probe_report = (
            json.loads(probe_report_path.read_text(encoding="utf-8"))
            if probe_report_path.is_file()
            else {}
        )
        grounding = (
            load_grounding_evidence(grounding_path)
            if grounding_path.is_file()
            else None
        )
        if grounding and grounding.selected_region_id:
            selected_region = next(
                region for region in grounding.regions
                if region.object_id == grounding.selected_region_id
            )
            target = (
                discovery_regions
                if directory.name.startswith("discovery-")
                else validation_regions
            )
            target.append((directory.name, selected_region))
        equipment.append({
            "run": directory.name,
            "valid": payload.get("valid") is True,
            "expected": payload.get("expected"),
            "audit_sha256": digest(path),
            "successful_select_events": selects,
            "automatic_grounding": (
                probe_report.get("automatic_change_localization") is True
                and probe_report.get("localized_change_cell_id")
                == (grounding.selected_cell_id if grounding else None)
                and grounding is not None
            ),
        })
    selection_claimed = report.get("tool_selection_via_adapter") is True
    grounding_claimed = report.get("automatic_change_localization") is True
    persistence_claimed = report.get("persistent_change_region") is True
    threshold = float(
        json.loads(preregistered.read_text(encoding="utf-8")).get(
            "region_iou_threshold", 0.5
        )
    )
    region_matches = []
    if len(discovery_regions) == 1:
        reference_name, reference_region = discovery_regions[0]
        for candidate_name, candidate_region in validation_regions:
            overlap = visual_region_iou(reference_region, candidate_region)
            region_matches.append({
                "reference": reference_name,
                "candidate": candidate_name,
                "iou": overlap,
                "threshold": threshold,
                "matched": overlap >= threshold,
            })
    audit = {
        "run": str(root),
        "preregistration_sha256_expected": expected,
        "preregistration_sha256_actual": actual,
        "preregistration_intact": isinstance(expected, str) and expected == actual,
        "source_snapshot": source_snapshot,
        "source_snapshot_complete": bool(source_snapshot) and all(
            item["valid"] for item in source_snapshot
        ),
        "equipment_audits": equipment,
        "all_equipment_controls_valid": bool(equipment) and all(
            item["valid"] for item in equipment
        ),
        "expected_equipment_audit_count": 4,
        "equipment_audit_count_complete": len(equipment) == 4,
        "tool_selection_via_adapter_claimed": selection_claimed,
        "all_claimed_agent_selects_present": (
            not selection_claimed
            or (len(equipment) == 4 and all(
                item["successful_select_events"] >= 1 for item in equipment
            ))
        ),
        "automatic_change_localization_claimed": grounding_claimed,
        "all_claimed_grounding_artifacts_valid": (
            not grounding_claimed
            or (len(equipment) == 4 and all(
                item["automatic_grounding"] for item in equipment
            ))
        ),
        "persistent_change_region_claimed": persistence_claimed,
        "region_matches": region_matches,
        "all_claimed_persistent_regions_valid": (
            not persistence_claimed
            or (
                len(discovery_regions) == 1
                and bool(region_matches)
                and all(item["matched"] for item in region_matches)
            )
        ),
        "report_accuracy": report.get("accuracy"),
        "report_coverage": report.get("coverage"),
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["source_snapshot_complete"],
        audit["all_equipment_controls_valid"],
        audit["equipment_audit_count_complete"],
        audit["all_claimed_agent_selects_present"],
        audit["all_claimed_grounding_artifacts_valid"],
        audit["all_claimed_persistent_regions_valid"],
        audit["report_accuracy"] == 1.0,
        audit["report_coverage"] == 1.0,
    ))
    destination = args.output or root / "integrity-audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps(audit), flush=True)
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

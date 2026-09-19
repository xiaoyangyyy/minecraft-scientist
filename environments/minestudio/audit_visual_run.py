"""Post-run integrity audit for an active visual discovery artifact directory."""
import argparse
import hashlib
import json
from pathlib import Path

from minecraft_scientist.minecraft.visual_evidence import successful_select_count


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
    equipment = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        path = directory / "equipment-audit.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        selects = successful_select_count(directory / "events.jsonl")
        equipment.append({
            "run": directory.name,
            "valid": payload.get("valid") is True,
            "expected": payload.get("expected"),
            "audit_sha256": digest(path),
            "successful_select_events": selects,
        })
    selection_claimed = report.get("tool_selection_via_adapter") is True
    audit = {
        "run": str(root),
        "preregistration_sha256_expected": expected,
        "preregistration_sha256_actual": actual,
        "preregistration_intact": isinstance(expected, str) and expected == actual,
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
        "report_accuracy": report.get("accuracy"),
        "report_coverage": report.get("coverage"),
    }
    audit["valid"] = all((
        audit["preregistration_intact"],
        audit["all_equipment_controls_valid"],
        audit["equipment_audit_count_complete"],
        audit["all_claimed_agent_selects_present"],
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

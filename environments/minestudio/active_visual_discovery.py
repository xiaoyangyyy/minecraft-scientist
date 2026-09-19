"""Live active selection with fixed fixture bindings and fresh held-out seeds."""
from dataclasses import asdict
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import shutil

from minecraft_scientist.discovery.visual_tools import discover_visual_tools
from minecraft_scientist.minecraft.visual_evidence import (
    load_grounding_evidence,
    visual_region_iou,
)
from minecraft_scientist.theory.ir import TheoryInterpreter


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_BINDINGS = {
    "obj_7d318f": {"tool": "air", "slot": 2},
    "obj_09c51a": {"tool": "iron_pickaxe", "slot": 3},
}
TARGET = "obj_b3c807"
CONTEXT = {
    "mode": "survival",
    "attack_frames": 20,
    "outcome": "fixed_camera_automatic_grid_change",
}


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--validation-seed", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.train_seed == args.validation_seed:
        parser.error("training and validation seeds must differ")
    output = args.output or ROOT / f"runs/active-visual-discovery-seeds{args.train_seed}-{args.validation_seed}"
    output.mkdir(parents=True, exist_ok=False)
    source_files = (
        "environments/minestudio/active_visual_discovery.py",
        "environments/minestudio/interaction_probe.py",
        "src/minecraft_scientist/discovery/visual_tools.py",
        "src/minecraft_scientist/discovery/hypotheses.py",
        "src/minecraft_scientist/minecraft/visual_evidence.py",
        "src/minecraft_scientist/theory/ir.py",
    )
    save(output / "runtime-manifest.json", {
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("minestudio", "numpy", "torch", "pydantic")},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in source_files},
        "train_seed": args.train_seed, "validation_seed": args.validation_seed,
    })
    snapshot = output / "source-snapshot"
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    control_audits = []
    grounding_records = {}

    def run_probe(tool_id, target_id, index, seed, phase):
        if target_id != TARGET:
            raise ValueError("unknown fixture target")
        directory = output / f"{phase}-{index}-{tool_id}"
        print(f"{phase} experiment {index}: {tool_id}, seed {seed}", flush=True)
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("interaction_probe.py")),
            "--seed", str(seed), "--block", "stone", "--mode", "survival",
            "--attack-frames", "20",
            "--tool", FIXTURE_BINDINGS[tool_id]["tool"],
            "--select-slot", str(FIXTURE_BINDINGS[tool_id]["slot"]),
            "--output", str(directory),
        ], check=True, timeout=180)
        audit_path = directory / "equipment-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("valid") is not True:
            raise RuntimeError(f"equipment audit failed: {audit_path}")
        probe_report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        grounding_path = directory / "grounding.json"
        grounding = load_grounding_evidence(grounding_path)
        if probe_report.get("automatic_change_localization") is not True:
            raise RuntimeError(f"automatic grounding was not used: {directory}")
        if (
            probe_report.get("localized_change_cell_id")
            != grounding.selected_cell_id
        ):
            raise RuntimeError(f"grounding report mismatch: {directory}")
        if grounding.evidence.changed is True and grounding.selected_region_id is None:
            raise RuntimeError(f"changed evidence lacks a visual region: {directory}")
        grounding_records[(phase, tool_id)] = grounding
        control_audits.append({
            "phase": phase,
            "experiment": index,
            "tool_id": tool_id,
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "grounding_sha256": hashlib.sha256(
                grounding_path.read_bytes()
            ).hexdigest(),
            "selected_region_id": grounding.selected_region_id,
            "valid": True,
        })
        return directory.name, grounding.evidence

    result = discover_visual_tools(
        tuple(FIXTURE_BINDINGS), (TARGET,),
        lambda tool, target, index: run_probe(tool, target, index, args.train_seed, "discovery"),
        budget=4, execution_context=CONTEXT,
    )
    save(output / "discovery.json", {
        "resolved": result.resolved, "candidate_count": result.candidate_count,
        "experiments": [asdict(item) for item in result.experiments],
        "theory": result.theory.model_dump(mode="json") if result.theory else None,
    })
    if result.theory is None:
        raise RuntimeError("visual theory unresolved within intervention budget")
    predictions = {tool: TheoryInterpreter.predict(
        result.theory, tool_id=tool, target_id=TARGET, execution_context=CONTEXT
    )["success"] for tool in FIXTURE_BINDINGS}
    positive_discovery = [
        grounding for (phase, _), grounding in grounding_records.items()
        if phase == "discovery" and grounding.evidence.changed is True
    ]
    if len(positive_discovery) != 1:
        raise RuntimeError("expected exactly one grounded positive discovery probe")
    reference_region = next(
        region for region in positive_discovery[0].regions
        if region.object_id == positive_discovery[0].selected_region_id
    )
    region_iou_threshold = 0.5
    save(output / "preregistered.json", {"predictions": predictions, "seed": args.validation_seed,
         "candidate_space": "ordered_tools_with_unreachable_targets",
         "reference_change_region": asdict(reference_region),
         "region_iou_threshold": region_iou_threshold,
         "execution_context": CONTEXT, "theory": result.theory.model_dump(mode="json")})
    digest = hashlib.sha256((output / "preregistered.json").read_bytes()).hexdigest()
    results = []
    for index, tool in enumerate(FIXTURE_BINDINGS):
        evidence_id, observed = run_probe(tool, TARGET, index, args.validation_seed, "validation")
        results.append({"tool_id": tool, "evidence_id": evidence_id,
                        "predicted": predictions[tool], "observed": observed.changed,
                        "control_stable": observed.control_stable,
                        "correct": observed.changed is not None and
                                   observed.changed == predictions[tool]})
        save(output / f"validation-{index}.json", results[-1])
    region_matches = []
    for item in results:
        if item["observed"] is not True:
            continue
        grounding = grounding_records[("validation", item["tool_id"])]
        region = next(
            region for region in grounding.regions
            if region.object_id == grounding.selected_region_id
        )
        overlap = visual_region_iou(reference_region, region)
        region_matches.append({
            "tool_id": item["tool_id"],
            "reference_region_id": reference_region.object_id,
            "validation_region_id": region.object_id,
            "iou": overlap,
            "threshold": region_iou_threshold,
            "matched": overlap >= region_iou_threshold,
        })
    persistent_regions = bool(region_matches) and all(
        item["matched"] for item in region_matches
    )
    if not persistent_regions:
        raise RuntimeError("held-out visual region did not match preregistered identity")
    assert digest == hashlib.sha256((output / "preregistered.json").read_bytes()).hexdigest()
    report = {"discovery_experiments": len(result.experiments),
              "validation": results, "preregistration_sha256": digest,
              "accuracy": sum(item["correct"] for item in results) / len(results),
              "coverage": sum(item["observed"] is not None for item in results) / len(results),
              "equipment_controls": control_audits,
              "all_equipment_controls_valid": all(item["valid"] for item in control_audits),
              "active_selection": True,
              "tool_selection_via_adapter": True,
              "automatic_change_localization": True,
              "persistent_change_region": persistent_regions,
              "region_matches": region_matches,
              "identity_source": "fixture bindings",
              "scope": (
                  "Two interventions, one fixture-aimed target, fixed camera, "
                  "automatically localized grid appearance change only"
              )}
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

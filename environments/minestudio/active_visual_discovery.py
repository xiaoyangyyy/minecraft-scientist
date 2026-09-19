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
from minecraft_scientist.theory.ir import TheoryInterpreter
from validate_visual_theory import evidence


ROOT = Path(__file__).resolve().parents[2]
BINDINGS = {"obj_7d318f": "air", "obj_09c51a": "iron_pickaxe"}
TARGET = "obj_b3c807"
CONTEXT = {"mode": "survival", "attack_frames": 20,
           "outcome": "fixed_camera_roi_appearance_change"}


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

    def run_probe(tool_id, target_id, index, seed, phase):
        if target_id != TARGET:
            raise ValueError("unknown fixture target")
        directory = output / f"{phase}-{index}-{tool_id}"
        print(f"{phase} experiment {index}: {tool_id}, seed {seed}", flush=True)
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("interaction_probe.py")),
            "--seed", str(seed), "--block", "stone", "--mode", "survival",
            "--attack-frames", "20", "--tool", BINDINGS[tool_id],
            "--output", str(directory),
        ], check=True, timeout=180)
        audit_path = directory / "equipment-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("valid") is not True:
            raise RuntimeError(f"equipment audit failed: {audit_path}")
        control_audits.append({
            "phase": phase,
            "experiment": index,
            "tool_id": tool_id,
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "valid": True,
        })
        return directory.name, evidence(directory)

    result = discover_visual_tools(
        tuple(BINDINGS), (TARGET,),
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
    )["success"] for tool in BINDINGS}
    save(output / "preregistered.json", {"predictions": predictions, "seed": args.validation_seed,
         "candidate_space": "ordered_tools_with_unreachable_targets",
         "execution_context": CONTEXT, "theory": result.theory.model_dump(mode="json")})
    digest = hashlib.sha256((output / "preregistered.json").read_bytes()).hexdigest()
    results = []
    for index, tool in enumerate(BINDINGS):
        evidence_id, observed = run_probe(tool, TARGET, index, args.validation_seed, "validation")
        results.append({"tool_id": tool, "evidence_id": evidence_id,
                        "predicted": predictions[tool], "observed": observed.changed,
                        "control_stable": observed.control_stable,
                        "correct": observed.changed is not None and
                                   observed.changed == predictions[tool]})
        save(output / f"validation-{index}.json", results[-1])
    assert digest == hashlib.sha256((output / "preregistered.json").read_bytes()).hexdigest()
    report = {"discovery_experiments": len(result.experiments),
              "validation": results, "preregistration_sha256": digest,
              "accuracy": sum(item["correct"] for item in results) / len(results),
              "coverage": sum(item["observed"] is not None for item in results) / len(results),
              "equipment_controls": control_audits,
              "all_equipment_controls_valid": all(item["valid"] for item in control_audits),
              "active_selection": True, "identity_source": "fixture bindings",
              "scope": "Two interventions, one fixed-camera target, appearance change only"}
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

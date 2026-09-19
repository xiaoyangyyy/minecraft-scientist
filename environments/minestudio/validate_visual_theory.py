"""Fit opaque-ID IR from prior pixels, freeze predictions, then run holdouts.

Identity correspondence is supplied by the fixture, not learned from vision.
This evaluates fixed-duration appearance change, not general harvest capability.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from minecraft_scientist.discovery.hypotheses import ToolVersionSpace
from minecraft_scientist.minecraft.visual_evidence import compare_regions
from minecraft_scientist.theory.ir import TheoryInterpreter


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ("air", "iron_pickaxe")
IDS = ("obj_7d318f", "obj_09c51a")
TARGET = "obj_b3c807"


def evidence(path):
    frames = [np.asarray(Image.open(path / f"{name}.png"))[80:144, 80:144]
              for name in ("control", "before", "after")]
    return compare_regions(*frames)


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main():
    output = ROOT / "runs/visual-theory-holdout-seed2"
    output.mkdir(parents=True, exist_ok=False)
    space = ToolVersionSpace(IDS, (TARGET,))
    sources = []
    for tool, opaque in zip(TOOLS, IDS, strict=True):
        source = ROOT / f"runs/visual-stone20-{tool}"
        observed = evidence(source)
        if observed.changed is None:
            raise RuntimeError("training control unstable")
        source_hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                         for name in ("control.png", "before.png", "after.png")}
        sources.append({"tool_id": opaque, "hashes": source_hashes,
                        "appearance_changed": observed.changed})
        space.observe(opaque, TARGET, observed.changed, f"training-{opaque}")
    required_context = {"mode": "survival", "attack_frames": 20,
                        "outcome": "fixed_camera_roi_appearance_change"}
    ir = space.to_ir()
    ir = ir.model_copy(update={"scope": {**ir.scope, "required_context": required_context}})
    predictions = {opaque: TheoryInterpreter.predict(
        ir, tool_id=opaque, target_id=TARGET,
        execution_context=required_context)["success"] for opaque in IDS}
    plan = {
        "ir": ir.model_dump(mode="json"), "training_sources": sources,
        "predictions": predictions, "holdout_seed": 2, "attack_frames": 20,
        "roi": [80, 144, 80, 144], "stability_limit": 1.0, "change_limit": 10.0,
        "scope": "Fixed-camera appearance change; fixture-supplied identity correspondence.",
        "autonomous_experiment_selection": False,
    }
    save(output / "preregistered.json", plan)
    plan_hash = hashlib.sha256((output / "preregistered.json").read_bytes()).hexdigest()
    print("Predictions frozen before holdout launch: " + json.dumps(predictions), flush=True)
    results = []
    for tool, opaque in zip(TOOLS, IDS, strict=True):
        destination = output / opaque
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("interaction_probe.py")),
            "--seed", "2", "--block", "stone", "--mode", "survival",
            "--attack-frames", "20", "--tool", tool, "--output", str(destination),
        ], check=True, timeout=180)
        observed = evidence(destination)
        results.append({"tool_id": opaque, "predicted": predictions[opaque],
                        "observed": observed.changed,
                        "control_mae": observed.control_mae,
                        "intervention_mae": observed.intervention_mae,
                        "correct": observed.changed is not None and
                                   observed.changed == predictions[opaque]})
        save(output / f"result-{opaque}.json", results[-1])
    assert hashlib.sha256((output / "preregistered.json").read_bytes()).hexdigest() == plan_hash
    report = {"preregistration_sha256": plan_hash, "results": results,
              "accuracy": sum(r["correct"] for r in results) / len(results),
              "coverage": sum(r["observed"] is not None for r in results) / len(results),
              "scope": plan["scope"], "sample_size": len(results)}
    save(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

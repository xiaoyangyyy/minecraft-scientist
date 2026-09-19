"""Exercise the production adapter against real RGB observations, not a fake."""
import json
import os
from pathlib import Path

os.environ.setdefault("MINESTUDIO_DIR", "/opt/minecraft-scientist-engine")

from PIL import Image
from minecraft_scientist.minecraft import GridVisualController, MineStudioVisualAdapter
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


class RecordingController:
    def __init__(self, output):
        self.controller = GridVisualController()
        self.output = output
        self.frames = 0

    def perceive(self, frame, info, *, seed):
        Image.fromarray(frame).save(self.output / f"frame-{self.frames:03d}.png")
        self.frames += 1
        return self.controller.perceive(frame, info, seed=seed)

    def encode_action(self, action, scene):
        return self.controller.encode_action(action, scene)


def main():
    output = Path(__file__).resolve().parents[2] / "runs/minestudio-adapter-smoke"
    output.mkdir(parents=True, exist_ok=True)
    controller = RecordingController(output)
    transitions = []
    with MineStudioVisualAdapter(controller) as world:
        initial = world.reset(0)
        for kind in (ActionKind.SELECT, ActionKind.INTERACT, ActionKind.WAIT):
            action = next(a for a in world.available_actions() if a.kind is kind)
            if kind is ActionKind.WAIT:
                action = PrimitiveAction(kind=kind, ticks=5)
            transition = world.step(action)
            transitions.append(transition)
            if transition.observed_delta["terminated"] or transition.observed_delta["truncated"]:
                break
    (output / "events.jsonl").write_text(
        "\n".join(t.model_dump_json() for t in transitions) + "\n", encoding="utf-8"
    )
    report = {
        "backend": "real-minestudio-production-adapter",
        "seed": 0,
        "initial_observation": initial.model_dump(mode="json"),
        "actions": [t.executed_action.kind.value for t in transitions],
        "simulator_steps": sum(t.event.duration for t in transitions),
        "all_controller_calls_completed": all(t.intervention_compliance for t in transitions),
        "frames_saved": controller.frames,
        "scope": "Adapter transport smoke only; no claim of visual mechanism discovery or object grounding.",
    }
    assert report["actions"] == ["select", "interact", "wait"]
    assert report["simulator_steps"] == 7
    assert report["all_controller_calls_completed"]
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Real simulator smoke test; run with the isolated Linux interpreter."""
import json
import os
from pathlib import Path

os.environ.setdefault('MINESTUDIO_DIR', '/opt/minecraft-scientist-engine')

import numpy as np
from PIL import Image
from minestudio.simulator import MinecraftSim


def main():
    output = Path('/mnt/d/minecraft-scientist/runs/minestudio-live-smoke')
    output.mkdir(parents=True, exist_ok=True)
    sim = MinecraftSim(action_type='env', seed=0)
    try:
        obs, _ = sim.reset()
        frame = np.asarray(obs['image'])
        Image.fromarray(frame).save(output / 'initial.png')
        shapes = [list(frame.shape)]
        for _ in range(5):
            obs, reward, terminated, truncated, _ = sim.step(sim.noop_action())
            shapes.append(list(np.asarray(obs['image']).shape))
            if terminated or truncated:
                break
        Image.fromarray(np.asarray(obs['image'])).save(output / 'final.png')
        report = {'backend': 'real-minestudio', 'seed': 0, 'frame_shapes': shapes,
                  'steps': len(shapes) - 1, 'initial_pixel_std': float(frame.std())}
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        sim.close()


if __name__ == '__main__':
    main()

# Isolated MineStudio visual environment

MineStudio is deliberately excluded from the root `uv` environment. Its simulator
documentation targets Linux, Python 3.10+, JDK 8, VirtualGL/Xvfb, and a separate
Minecraft runtime; mixing those dependencies into the deterministic Windows test
environment would make the offline release gate irreproducible.

Create this environment on a Linux machine or container with a working GPU/display:

    python3.10 -m venv .venv-minestudio
    source .venv-minestudio/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r environments/minestudio/requirements.txt
    python -m pip install -e .
    minecraft-scientist check-minestudio-backend

The backend check is read-only and never downloads or launches Minecraft. A real
run must still comply with Minecraft's EULA and MineStudio's own setup steps.

`MineStudioVisualAdapter` wraps `MinecraftSim.reset()`/`step()` and satisfies the
project's `StructuredWorld` surface. It requires an explicit `PerceptionController`
implementation that converts RGB frames to opaque `ObjectObservation` values and
converts primitive scientific interventions into low-level simulator actions. The
adapter hashes frame bytes into observation metadata, keeps controller state
private, rejects non-opaque object IDs, and detects common semantic-name leakage.

The unit test uses an injected fake simulator and fake perception controller. This
proves the contract and isolation logic, not visual recognition quality or live
Minecraft performance.

`GridVisualController` is the concrete dependency-free baseline. It exposes nine
stable screen regions with mean-intensity/contrast features and opaque hotbar slots,
then maps the scientific primitives to MineStudio's documented `env` action keys.
It intentionally uses no labels or privileged voxel names. It can aim and interact
with a reset-controlled fixture, but it is not an object detector or navigation
policy; stronger vision models can implement the same `PerceptionController`.

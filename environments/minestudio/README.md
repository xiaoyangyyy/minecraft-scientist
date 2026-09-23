# Isolated MineStudio visual environment

For the verified local WSL installation and exact runtime commands, see
[INSTALL-STATUS.md](INSTALL-STATUS.md). The generic instructions below do not
include the compatibility overrides needed by the installed MineStudio 1.1.6.

`interaction_probe.py` is a controlled creative-mode probe, not a scientific
discovery benchmark. Its first run failed because absolute-origin chunks were
unloaded. The corrected loaded-chunk run has a stable target (control ROI MAE 0)
and visually confirmed removal after attack (ROI MAE 45.4381). Evidence is in
`runs/minestudio-interaction-probe-loaded/{before.png,after.png,report.json}`.
This single fixture does not establish tool tiers, general recognition or causal
discovery across worlds. Perception sees RGB only; setup commands are privileged.

The probe accepts `--seed`, `--block dirt|stone|oak_log`,
`--mode creative|survival`, and `--attack-frames 1..200`. It requires a new
`--output` directory and refuses to overwrite previous evidence. Privileged block
identity is saved in `fixture.json`, separately from opaque Agent events.

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/interaction_probe.py --seed 1 --block stone --output /mnt/d/minecraft-scientist/runs/visual-probe-seed1-stone
```

An ROI change measures appearance, not removal by itself. Inspect saved images
and reject unstable controls before treating a run as valid evidence.

`validate_visual_theory.py` fits the existing finite tool version space from the
two seed-0 stone probe image triplets, writes an opaque-ID executable IR and
predictions to `preregistered.json`, then launches seed-2 holdouts. It verifies the
prediction file hash afterward and includes abstentions in accuracy/coverage.
The runner is deliberately limited: tool/target identity correspondence is
fixture-supplied, experiment selection is fixed, and the outcome is appearance
change after 20 attack frames. It does not establish autonomous visual discovery,
drop eligibility, or a general capability hierarchy. The fixed ROI thresholds
are exploratory and require further independent calibration for other fixtures.

`active_visual_discovery.py` connects information-gain selection to real probes,
then freezes predictions before validation. The first seeds-3/4 diagnostic used
the original reachable-target prior and inferred the second tool from one failure.
Do not count that run as final discovery acceptance. The visual learner now allows
unreachable targets (including both tools failing), requires competing evidence,
and leaves budget-exhausted or unstable experiments unresolved. The default
structured-world learner retains its original world-specific prior.

The accepted controlled run is `runs/active-visual-persistent-region-seeds22-23`.
It used the corrected unreachable-target candidate space, two seed-22 discovery
probes and two seed-23 holdouts. Each probe placed a fixture-bound candidate in a
different hotbar slot and made the Agent issue an opaque `SELECT` action through
`MineStudioVisualAdapter` before interaction. All four action logs and equipment
traces passed. A HUD-safe world viewport was divided into anonymous cells and the
strongest controlled change was localized automatically; no fixed target crop was
used for outcome evidence. Holdout accuracy and coverage were both 1.0. The frozen
prediction hash is
`981759aef3c79308305f71db7477ede40719ac4c1896dd9edd6eaa0707d03d08`.
`audit_visual_run.py` independently verifies that hash, all four successful select
events, all four equipment audits, all four grounding artifacts, and the reported
metrics. The learner directly consumes strictly schema-validated grounding
artifacts; it does not recompute the legacy central crop. The run also snapshots
the exact source files used, and the independent audit verifies every snapshot
hash against the runtime manifest. Connected changed cells are now grouped into
anonymous regions. The discovery region and held-out region differed by one
boundary cell but matched at IoU 0.857 against a preregistered 0.5 threshold.
The fixture still aims the camera and supplies opaque candidate identity bindings;
this is persistent change-region matching, not general object identity or
autonomous navigation.

`camera_search_benchmark.py` separately verifies active viewpoint control. Before
launching Minecraft it preregisters negative/center/positive yaw placements and
the label-free argmax brightness rule. The accepted run is
`runs/camera-search-suite-seed30`: all 3/3 placements were selected correctly and
15/15 explicit `LOOK` transitions succeeded. `audit_camera_search.py` independently
recomputes every score from the saved PNGs, verifies every event and report hash,
and checks the five-file source snapshot. This is a controlled marker-search task,
not free-form object detection or navigation.

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/camera_search_benchmark.py --seed 30 --output /mnt/d/minecraft-scientist/runs/camera-search-suite-seed30
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_camera_search.py /mnt/d/minecraft-scientist/runs/camera-search-suite-seed30
```

`navigation_benchmark.py` tests the next deliberately narrow layer: visual approach
along a reset-controlled straight corridor. The controller exposes one opaque
`MOVE` handle; its private binding maps to MineStudio's forward action, while the
policy sees only RGB and repeats four-frame moves until the preregistered central
salience score first reaches 150. The accepted `runs/navigation-suite-seed49`
suite passed all 3/3 seeds. Each run stopped after seven moves, and independent
position telemetry measured 7.504 blocks of displacement. The wrapper records the
privileged source, strips all position fields before the adapter, and the RGB
controller fails closed if any survive. The audit verifies both sides of that
boundary and confirms that no `location_stats` or coordinate field appears in the
Agent event log. It also recomputes every score from saved PNGs and checks the first
threshold crossing, action duration and compliance, report hashes, preregistration
hash, and five source snapshots.

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/navigation_benchmark.py --seed 49 --cases 3 --output /mnt/d/minecraft-scientist/runs/navigation-suite-seed49
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_navigation.py /mnt/d/minecraft-scientist/runs/navigation-suite-seed49
```

This fixture proves closed-loop locomotion and visual stopping in one known-safe
direction. By itself it does not prove direction discovery, obstacle avoidance,
mapping, free-form exploration, or path planning.

`direction_discovery_benchmark.py` removes the known-direction assumption. Each
case uses a world seed and an independently generated mapping seed to permute four
opaque MOVE handles. The RGB-only policy tests every handle in a reset-identical
trial, chooses the unique maximum salience improvement, then repeats that handle in
a fresh approach episode. The accepted
`runs/direction-discovery-suite-seed62-map700` suite passed 2/2 cases; winners were
at different anonymous indices (3 and 1), matched evaluator-recomputed positive
target-axis progress, stopped at the first threshold crossing, and displaced 7.504
blocks. `audit_direction_discovery.py` independently recomputes every calibration
and trajectory PNG, checks all MOVE events and private control permutations, verifies
world/mapping seed separation and source hashes, and proves public artifacts contain
neither direction semantics nor coordinates.

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/direction_discovery_benchmark.py --seed 62 --mapping-seed 700 --mapping-seed-stride 2 --cases 2 --output /mnt/d/minecraft-scientist/runs/direction-discovery-suite-seed62-map700
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_direction_discovery.py /mnt/d/minecraft-scientist/runs/direction-discovery-suite-seed62-map700
```

Calibration uses controlled resets and the execution phase repeats one selected
control. It is evidence for active motor-effect discovery, not online obstacle
avoidance, route replanning, or a general spatial cognitive map.

MineStudio is deliberately excluded from the root `uv` environment. Its simulator
documentation targets Linux, Python 3.10+, JDK 8, VirtualGL/Xvfb, and a separate
Minecraft runtime; mixing those dependencies into the deterministic Windows test
environment would make the offline release gate irreproducible.

Create this environment on Linux with Java 8 and a working display. The verified
installation uses uv and explicit upstream-pin overrides, not plain pip resolution:

```bash
uv venv --python 3.11 /opt/minecraft-scientist-visual
uv pip install --python /opt/minecraft-scientist-visual/bin/python \
  --overrides environments/minestudio/overrides.txt \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r environments/minestudio/requirements.txt
uv pip install --python /opt/minecraft-scientist-visual/bin/python -e .
export MINESTUDIO_DIR=/opt/minecraft-scientist-engine
/opt/minecraft-scientist-visual/bin/minecraft-scientist check-minestudio-backend
```

The CPU wheel index is intentionally combined with PyPI to resolve MineStudio's
mixed dependency set. Use only these trusted sources. Engine installation is a
separate upstream download; the installed runtime uses the persistent directory
above, not `/tmp`. `install_engine.py` is an optional checksum-verification utility
for a separately downloaded archive; it was **not** used to install this runtime.

The backend check is read-only and never downloads or launches Minecraft. A real
run must still comply with Minecraft's EULA and MineStudio's own setup steps.

`MineStudioVisualAdapter` wraps `MinecraftSim.reset()`/`step()` and satisfies the
project's `StructuredWorld` surface. It requires an explicit `PerceptionController`
implementation that converts RGB frames to opaque `ObjectObservation` values and
converts primitive scientific interventions into low-level simulator actions. The
adapter hashes frame bytes into observation metadata, keeps controller state
private, rejects non-opaque object IDs, and detects common semantic-name leakage.

Unit tests use an injected fake simulator and fake perception controller. Live
scripts provide separate empirical evidence; neither unit tests nor startup smoke
tests establish visual recognition quality or general Minecraft performance.

`GridVisualController` is the concrete dependency-free baseline. It exposes nine
stable screen regions with mean-intensity/contrast features and opaque hotbar slots,
plus opt-in opaque movement handles, then maps the scientific primitives to
MineStudio's documented `env` action keys. It intentionally uses no labels or
privileged voxel names. It can aim, interact, and perform controlled locomotion in
reset fixtures, but it is not an object detector or general navigation policy;
stronger vision models can implement the same `PerceptionController`.

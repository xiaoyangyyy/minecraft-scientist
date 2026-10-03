# Runtime image/camera timing diagnosis

This is the preserved 66-frame timing-only checkpoint, not the current capture
schema or package release. The later full-matrix measurement, public coordinate
fix, 106-frame acquisitions and separate camera/body evaluation are documented in
`rendered-camera-motion-development.md`. Historical source snapshots remain intact.

The joint-map short-motion counterexample is still unresolved. This follow-up
identifies a real observation/evaluation mismatch in fresh runtime measurements;
it does **not** retrospectively replace player telemetry, certify old poses,
improve navigation coverage, or establish the cause of the 33.042212-degree error.

## Independent runtime measurement

At this frozen checkpoint, `environments/minestudio/capture_runtime_camera.py`
preregisters 65 one-tick
WAIT/MOVE/LOOK actions at natural seed-201 spawn. No terrain is edited and no
learned map is promoted. It compiles a JDK8-only Java instrumentation agent into
the new run directory. The agent instruments only the in-process recorder's
`resolveImageRequests()` return, after framebuffer bytes have been copied and
before the recorder increments its tick counter and wakes the environment server.
Neither the installed engine JAR nor the shared SDK template is edited.

Each sample records the raw GPU RGB SHA-256, recorder sequence/tick, last rendered
`ActiveRenderInfo` base camera, current/previous player poses, eye height, renderer
settings and FOV modifiers. All coordinate/camera telemetry remains in evaluator
files. The scientist's controller receives only opaque handles and RGB; the public
event stream is independently checked for leaked pose fields. JVM instrumentation
is supplied only to one fresh child launch, with Python environment and launch
method restored on success and failure.

`runs/runtime-camera-seed201-v1` is retained as the initial 66-frame exploratory
capture. Its source boundary predates native-image retention and the FOV fields.
`runs/runtime-camera-seed201-v2` is a second complete fresh 66-frame capture,
also retaining all 1280x720 native images alongside the 448x448 public images.
Both runs execute all 65 scheduled actions; their frames are not independent
worlds, and this is not held-out generalization.
The V2 acquisition freezes 105 source entries, including 98 package Python files,
the Java agent/manifest, capture definitions and installed SDK entry/launcher.

The independent auditor `scripts/audit_runtime_camera.py` checks source snapshots,
current project hashes, engine/agent identity, the complete fixed action schedule,
every public frame/action receipt and the complete artifact ledger. It then
recomputes all native hashes and every SDK resize, and requires **one unique
contiguous match of the entire 66-frame sequence** to the JVM log, with consecutive
recorder counters. It does not independently match frames to a nearest/best pose.
The 114 JVM samples include 48 startup/reset samples preceding the matched sequence.

## Measured result

The second acquisition passes all six capture integrity checks. Its independently
recomputed `audit-v4.json` passes all fourteen checks, including unchanged bytes
through analysis. The audit's own exact source is preserved separately as
`audit-v4.source.py`; earlier audit versions remain available.

| Quantity over all 66 matched frames | Measured value |
|---|---:|
| Nonstationary current/previous player samples | 35 |
| Maximum Python telemetry vs capture-time player component error | 0 |
| Maximum camera-base vs current player + eye-height distance | 0.461680409 blocks |
| Maximum camera-base vs previous player + eye-height distance | 0.000000114441 blocks |
| Maximum camera-base vs current yaw/pitch error | 0 degrees |
| Render partial ticks | 0 throughout |
| Integrated-server synchronization / recorder disabled | true / true throughout |
| View bobbing | enabled throughout |
| Current/previous hand FOV modifier and player FOV modifier | 1 throughout |

In this trajectory the rendered **position** is effectively the previous player
position while Python telemetry is the capture-time current player position.
The rendered yaw/pitch matches current telemetry; previous yaw/pitch happens also
to match in these samples. Therefore blindly shifting a complete pose record by
one frame is not justified. An explicit synthetic unit case additionally ensures
that translation and orientation alignment remain separately reported.

Installed-bytecode inspection agrees with the measured position mismatch:
the synchronized recorder-disabled game loop assigns render partial ticks zero;
`ActiveRenderInfo.update()` interpolates `prevPos*`/`getPos*` and eye height;
the recorder copies the preceding framebuffer before the next server/client tick;
`EnvServer.stepClient()` returns those pixels and obtains stats from the current
client player. These code paths motivated the measurement, not an assumed offset
chosen to make geometric predictions pass.

## Limits and next engineering action

`ActiveRenderInfo` is the **base camera**, not the complete view-bobbing/hurt-camera
transform. This diagnosis does not turn it into exact full rendered-pose truth.
No dynamic FOV was measured in these actions, but that does not prove FOV is
constant for sprinting, flying, swimming or other conditions. The sequence includes
natural collisions/falls and only one development seed/condition.

Next, bind new multi-frame geometric evaluation to capture-time camera evidence,
retaining simultaneous body-motion scoring and the existing old counterexample.
Measure the remaining view transform or independently verify a declared fixed-view
profile before treating camera pose as geometric truth. Only then determine what
short-baseline estimator uncertainty remains, preregister the revised method, and
validate on fresh disjoint trajectories/worlds. Do not use a lagged telemetry
model selected by private error, relax the direction gate, or claim a navigation
gain from this diagnostic.

The five-additional-world/multiple-terrain/three-condition evaluation, useful goal
navigation, natural science/map integration, 100-attempt recovery and consolidated
live/baseline/CI release requirements are unchanged and incomplete.

## Verification and reproduction

Focused tests cover sequence ambiguity/gaps/counter types, separate pose components,
invalid renderer fields, full pixel/receipt replay, source/policy/public-log/ledger
forgery before private JSON access, mutation through analysis, JVM environment/class
restoration on failure, single-launch bounds and immutable audit outputs.
At this timing-only checkpoint, no production package or core-verifier source is changed; the
V18 core source boundary and existing benchmark artifacts remain unchanged.
All 101 V18 source hashes were rechecked on both platform ledgers with zero
mismatches. Windows and WSL each pass the final 25-case runtime-camera suite
(`runs/runtime-camera-final-{windows,wsl}-v1.xml`), with no failures/errors/skips.
The preceding 70-case combined camera/joint-map regression also passes on both
platforms; the final two additional tests cover auditor-source snapshot retention
and CLI overwrite refusal. Ruff/check/format and Git whitespace checks pass.

In the dedicated WSL runtime, from `/mnt/d/minecraft-scientist`, use new outputs:

```bash
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_runtime_camera.py --seed 201 --output runs/new-runtime-camera
timeout 120 /opt/minecraft-scientist-visual/bin/python scripts/audit_runtime_camera.py runs/new-runtime-camera --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-runtime-camera/audit.json
```

JDK8 and the already installed optional MineStudio runtime are required for
acquisition, not the offline core. Native pixel audit needs Pillow/OpenCV but not
MineStudio or a running game. Existing directories, reports and auditor snapshots
are refused rather than overwritten. No provider or API key is used.

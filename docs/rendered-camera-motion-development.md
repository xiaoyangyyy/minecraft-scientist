# Rendered-camera motion evaluation: development checkpoint

This preserves the V19/99-package-source checkpoint. A later current-prefix
controller and refreshed-map replay are documented in `causal-window-development.md`;
they change the current source boundary but do not retroactively change these runs.

This follow-up measures the complete world-render pinhole projection, separates
image-camera motion from simultaneous body motion, and fixes an actual public
pixel-coordinate inconsistency. It does **not** establish reliable navigation,
five-world generalization, or the cause of the historical 33.042212-degree error.
All earlier measurements and rejected maps are preserved.

## Measurement and inference boundaries

The isolated JDK8 agent records the model-view and projection matrices at entry to
`WorldRenderer.updateCameraAndRender`, in addition to binding framebuffer hashes
to recorder snapshots. Minecraft's bobbing effects can occur in the projection
stack; adding eye height to the base camera alone does not recover these effects.
No engine JAR or shared SDK is edited. Shader/shadow frames and invalid/nonrigid
matrices fail closed, rather than being quietly interpreted as this view model.

`evaluation/rendered_camera.py` composes the GL matrices with the image viewport,
uses NumPy RQ decomposition to recover a proper rotation, camera center and upper
triangular intrinsics, and verifies projection reconstruction. Synthetic tests
compare independently projected world points, including bobbing translation and
roll. The module requires no simulator, Pillow or OpenCV. These matrices and their
derived intrinsics are **evaluator-only**; the RGB estimator does not consume them.

The public reader in `scripts/audit_runtime_camera.py` verifies native/public RGB,
every one-tick action receipt, all SDK resizes, source snapshots and bytes, without
opening either private coordinate/matrix file, even to hash it. Tests guard both
byte and text reads. Private hashes and JSON are read only in the evaluator, after
public verification. Each whole RGB sequence must have a unique contiguous match
to consecutive recorder and world-render samples.

`scripts/probe_rendered_motion.py` additionally replays the **entire** public
inference before private scoring. Both trials' past maps are frozen before any
future tracking. All nine bootstrap offsets 2 through 10 per trial are retained;
there is no private best-candidate selection. Track identities cannot resurrect,
and earlier pixel measurements cannot be rewritten. Future tests use only the
surviving original fitting/withheld partitions and do not refit the old map.

The public focal calibration uses twelve completed startup RGB views. Both its
feature points and `TemporalTracks` use OpenCV coordinates plus 0.5 (image-edge
origin). The old `CameraResizeSpec` supplied grid-index coordinates instead. Its
new explicit `pixel_edge_center` option fixes that mismatch: principal point
224, not 223.5, in the 448-square edge-origin observation. Existing grid-index
callers keep their original default. Independent ray and resize tests prove the
conversion without consulting any private Minecraft matrix.

## Preserved runs and denominators

All three `runs/runtime-camera-projection-seed201-v{1,2,3}` acquisitions retain
106 public/native RGB frames and all 105 fixed WAIT/LOOK/MOVE actions each. There
are two preregistered `(anchor, fitting end, future end)` intervals:
`(48,58,62)` and `(79,89,93)`. The first ten movement ticks fit a map; the following
four use a control distinct from both fitting controls. No terrain is edited.
These are repeated acquisitions on **one development seed**, not independent worlds.

Projection captures 1 and 2, and `runs/rendered-motion-seed201-v1`, preserve the
pre-coordinate-fix source boundary. Projection capture 3 and
`runs/rendered-motion-seed201-v4/{config,report,audit}.json` are the current
boundary. `rendered-motion-seed201-v2` is retained as the intermediate audit without
the additional rejected-pose diagnostic fields; version 3 predates the final
private-byte freeze spanning alignment and scoring. Capture 3 and capture 2 have
93/106 identical public image files, not an entirely identical sequence: changes
in the new result cannot be attributed solely to the coordinate fix.

The current capture freezes 106 source entries (99 package files plus seven
capture/agent/SDK definitions). Its six acquisition checks pass. The separate
physical audit passes 18 checks; the full inference replay/audit passes 13 checks.
Passes certify source/data/inference integrity, **not** prediction accuracy.

Across all 106 matched renders, the base camera differs from current player plus
eye height by up to 0.456362 blocks and matches previous position to 1.15e-7 blocks.
The effective rendered center additionally differs from the base by up to
0.0763421 blocks; rotation differs by up to 0.372148 degrees. Full projection
reconstruction relative error is at most 6.78e-16. This does not justify shifting
an entire pose, selecting a telemetry offset by error, or altering old evaluators.

## Current result, including failures

| Trial | Past surviving tracks | Future surviving tracks | Supported past maps / all candidates | Entire future pixel tests consistent |
|---|---:|---:|---:|---:|
| 0: 48→58→62 | 89 | 54 | 3 / 9 | 0 / 9 |
| 1: 79→89→93 | 31 | 26 | 0 / 9 | 0 / 9 |

The second trial lacks the unchanged 32-track minimum. No minimum is reduced to
make it pass. Trial 0 bootstrap 8 loses sufficient original support; bootstraps 9
and 10 pass frame 60 but fail frame 62's fixed withheld-point pixel tests.

| Trial-0 bootstrap | Past rendered direction error | Past body direction error | Incremental 58→60 rendered error | Incremental 58→62 diagnostic rendered error |
|---|---:|---:|---:|---:|
| 8 | 1.004212° | 8.546789° | unavailable | unavailable |
| 9 | 1.867998° | 7.648470° | 5.502778° | 7.469477° |
| 10 | 2.413529° | 7.160799° | 7.287796° | 8.505253° |

At frame 62, the pose diagnostics are within the existing 20-degree direction and
5-degree rotation gates for rendered motion, despite failing pixel validation.
They are **not policy acceptance**. The auditor retains both the gate-rejected
policy outcome and the unpromoted fitted-pose diagnostic. Bootstrap 10's body
direction error on that same incremental interval is 20.185214 degrees, outside
the body gate. Camera and body scores therefore must not be conflated.

This evidence changes the next action: investigate short-horizon rolling map
refresh, original-track support and withheld-point uncertainty, keeping positive
and negative pixel/physical checks separate. Simply correcting camera truth or
half a pixel does not fix future-map support. No current map is promoted; there
are zero accepted navigation attempts from this diagnostic. The historical
short-step counterexample remains unchanged.

The current offline future test uses identities surviving **all four** future
frames to score both target views. Thus frame-60 support is retrospective, not an
online decision known at frame 60. The next controller must validate each horizon
against only its own chronological surviving prefix before taking another action;
later survival must not decide an earlier online gate.

## Reproduction

In the dedicated existing WSL runtime, use **new output names**:

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_runtime_camera.py --seed 201 --output runs/new-render-capture
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/probe_rendered_motion.py runs/new-render-capture --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-render-motion
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/probe_rendered_motion.py runs/new-render-capture --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --audit runs/new-render-motion --output runs/new-render-motion/audit.json
```

No API provider/key is used. Frozen historical runs are not restamped as current
when source evolves. A separate 30-seed/two-repeat V19 scientific-core release
passes on both Windows and WSL, verifying 18 identical persisted artifacts per
seed and all six release checks. It freezes 102 source entries, including 99
package files. The core release tests discovery, not visual navigation.

The changed Python files pass scoped Ruff checks. A broader read-only check also
finds 21 lint issues in 16 untouched files and 34 files needing formatting; these
remain engineering cleanup work, not a globally clean Ruff claim. Build outputs
are produced successfully. Detailed final regression counts follow with their
precise scope.

## Verification scope

- V19 core source ledgers were rechecked: 102 entries on each platform, zero
  mismatches. No later core-package/verifier changes were made.
- Frozen full regression: Windows 927 passed / 2 skipped (929 total), WSL
  929 passed / 2 skipped (931 total), reports
  `runs/rendered-motion-full-{windows,wsl}-v2.xml`.
- The final script-only private-byte guard and two additional mutation cases were
  added after that full checkpoint. The entire affected five-module focused suite
  then passes 123/123 on both platforms, reports
  `runs/rendered-motion-targeted-{windows,wsl}-v5.xml`. This is not a claim that the
  whole suite was rerun after that final script-only addition.
- Isolated default-only WSL environment has no Torch, Pillow or OpenCV; 477 tests
  pass / 69 skip (546 total), `runs/rendered-motion-default-only-wsl-v2.xml`.
- The original MVP acceptance CLI passes: seed 0 has 1.0 accuracy/task success,
  47 experiments, detection delay 1, revision experiments 8, three symbols and
  13 reuses, `runs/rendered-motion-regression-mvp-seed0/benchmark.json`.
- Source and wheel builds, changed-file Ruff/check/format and Git whitespace
  checks pass. No hosted CI run, provider use or broad navigation release is claimed.

The first full WSL run has one retained test failure: concurrent source
formatting invalidated a fixture's hash gate before its expected replay stage.
The frozen rerun above passes without relaxing the gate. Negative evidence and
all intermediate reports remain available, including this failed run.

The original completion requirements remain: useful geometric goal navigation,
at least five additional disjoint worlds/multiple terrains/three conditions,
natural scientific/map integration, 100 attempts and recovery, current live and
baseline comparisons, and consolidated CI/release verification. None is replaced
by camera diagnostics or unit-test totals. The broad project goal remains active.

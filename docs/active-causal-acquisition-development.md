# First-eligible acquisition and live dispatch: development checkpoint

The estimator now chooses when to end acquisition and dispatch a distinct control
in the actual game. The fresh result is still negative: every new-view map test
fails. This is not useful navigation, held-out generalization, equal-budget
superiority or full-project completion.

## Historical replay that motivated the change

`scripts/probe_first_eligible_map.py` records
`runs/first-eligible-map-seed201-v1` on the preceding public capture. It attempts
each prefix from three through ten ticks, retains all bootstraps, and latches the
first prefix with two supported past maps **before** testing future pixels.

Trajectory 0 latches at frame 57 (nine ticks), with four supported maps instead of
the fixed-ten-tick window's one. Its future still has only one passing map and is
rejected. Trajectory 1 exhausts its budget without a supported map and rejects.
All 79 fits (35 + 44) remain visible. Its 13 replay/integrity and 18 camera checks
pass at the frozen **pre-live-module** source boundary. Those snapshots are now
historical; do not restamp them as current code. More past support does not prove
better future prediction or equal-compute efficiency.

## Public package and actual dispatch

`exploration/active_causal_map.py` implements `ActiveMapPolicy` and
`acquire_causal_map(initial_frame, execute, camera, anchor_frame=..., seed=...)`.
Its callback receives only an opaque handle index and returns `PublicProbeStep`
RGB/compliance receipts. No coordinates, renderer matrices or evaluator enter it.

The registered ten-tick acquisition requests handle 0 for at most six ticks,
then handle 2 if still collecting. Each compliant tick adds exactly its current
public image prefix; original track IDs intersect permanently and past pixels
cannot change. From tick three all bootstraps are retained. The first prefix
with two supported maps is frozen; the lowest handle unused by the acquisition
controls (1) is requested for the original two-tick causal validation. Handle
diversity is not necessarily physical diversity, especially with inertia.

Original support floors, withheld-ray thresholds and all-passing-map consensus
are unchanged. Lost support aborts without landmark replenishment; exhaustion
retains every failed attempt. Failed controls cannot enter evidential frames.
Control, tracking and estimator failures remain explicit. Even internally valid
motion has unknown metric scale and `navigation_accepted=false`.

`capture_runtime_camera.py --active-causal` performs the existing 48 calibration
actions, then calls this policy **between real game actions**. It saves calibration,
all attempts, public receipts, frozen maps and future decisions in
`active-acquisition.json`. Default fixed capture remains separately registered.

`scripts/audit_active_causal_map.py` independently replays all requested controls,
calibration, tracks, maps and decisions before accessing private camera/body
bytes. Dynamic registration requires this explicit replay callback; the default
fixed registration rejects it. Forged gates and extra/missing actions cannot reach
physical scoring. Public bytes freeze across the nested camera audit; private
hashes must still match the original capture ledger. Late mutation discards scores.
The independent auditor is snapshotted beside its output.

## Fresh actual-game result

`runs/active-causal-live-seed201-v1` is a fresh JDK8/MineStudio launch in the same
seed-201 development world, clear/noon, without terrain edits; **not held-out**.
It captures 60 public/native frames and 59 action receipts: 48 calibration actions,
six handle-0, three handle-2 and two handle-1. Its 108 source entries include 101
package files; all seven capture checks pass and `failure=null`.

| Past endpoint | Original tracks | Supported maps / fitted |
|---|---:|---:|
| 51 | 187 | 0 / 2 |
| 52 | 174 | 0 / 3 |
| 53 | 159 | 1 / 4 |
| 54 | 140 | 0 / 5 |
| 55 | 130 | 0 / 6 |
| 56 | 118 | 1 / 7 |
| 57 | 102 | 4 / 8; first eligible |

All 35 fits are retained. Validation ends at 59 with 71 original tracks. None of
the four supported past maps passes the future gate. Bootstrap 8/9 have 90% passing
withheld rays but 95th-percentile errors 4.453831/3.315199 pixels, above the unchanged
3-pixel limit at 448 resolution; bootstrap 6/7 fail more substantially. Do not
delete failed rays or promote rejected diagnostics.

The audit passes three outer checks and all 18 camera/public checks. Actual 57→59
translation is 0.719296 blocks for the rendered camera and 0.841293 for the player.
Both unsupported predictions score `correct=false`; direction error is null,
**not zero**. Useful motion coverage is **0/1**. Audit validity is not accuracy.

## Verification and next action

Windows targeted package/controller/runtime tests pass 89 cases; WSL package/
controller tests pass 45. Default-only WSL passes 521 cases/70 skips without Torch,
Pillow or OpenCV. Both V21 core releases pass 30 seeds/two repeats/18 identical
artifacts per seed and all six checks; their 104 entries include 101 package files.
MVP seed 0 retains accuracy/task success 1.0, 47 experiments, detection delay 1,
eight repairs and three symbols. Package builds and changed-file Ruff/format pass.

Full current regressions pass Windows 983 cases/two skips (985 total) and WSL
985 cases/two skips (987 total), `runs/active-causal-full-{windows,wsl}-v1.xml`.
All 104 source hashes were rechecked against both V21 ledgers with zero mismatches.
These tests certify their declared scopes, not visual-navigation success.

Next improve acquisition conditioning, not acceptance thresholds or private map
selection. The six-tick first-control cap is still fixed: request complementary
controls earlier using past public geometry/visibility and measure independent
future prediction. Investigate past-only landmark depth/pose conditioning without
fitting poses to future/withheld rays. Keep every failed attempt and explicit cost.

Reliable navigation, five additional disjoint worlds/multiple terrains/three
conditions, natural scientific-map integration, 100 attempts/recovery, current
live comparisons and consolidated release/CI remain open. The goal stays active.

In the installed optional WSL runtime, use new output names:

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_runtime_camera.py --active-causal --seed 201 --output runs/new-active-capture
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/audit_active_causal_map.py runs/new-active-capture --engine-jar /opt/minecraft-scientist-engine/engine/build/libs/mcprec-6.13.jar --output runs/new-active-capture/audit.json
```

No provider/key, engine edit or shared SDK edit is needed. Historical evidence and
accepted route memory remain untouched.

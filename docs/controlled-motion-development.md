# Controlled natural RGB movement diagnosis

This follow-up to `temporal-motion-development.md` separates acquisition conditions
from movement-estimator reliability. It remains single-seed development, not held-out
navigation acceptance. The existing scientific loop and accepted map are unchanged;
the full project gates remain in `quality-roadmap.md`.

## Fixed intervention and public boundary

`motion_probe_schedule.py` creates a deterministic, bounded schedule independent of
RGB scores or evaluator state. Each new MineStudio session uses natural seed 201,
creative/peaceful mode, unchanged terrain, an empty hand and 448×448 observations
resized from the explicitly configured 1280×720 render. Setup changes only the
requested time/weather and safety/display rules; it does not aim the player, place
markers or replace terrain.

The 42-action schedule contains an initial WAIT, a twelve-view public rotation
calibration scan, all four opaque movement handles at three and six ticks, twelve
WAIT ticks after each MOVE, and a final WAIT. Every simulator step is captured:
201 PNGs per session. Public pairs include all eight moving endpoints, all eight
post-WAIT endpoints, eight late-WAIT controls, two rotation controls, two baseline
WAIT controls and two identical-image controls. The moving and post-WAIT views
share eight interventions; they are not sixteen independent experiments.

The action plan is written before startup. Camera dimensions, raw image hashes,
public transition records, reference images and acquisition/runtime source
snapshots are preserved. The SDK entry source is snapshotted and checked unchanged
during capture. Offline validation verifies that snapshot, not a claim that a
Windows machine ran the Linux SDK. `PositionAuditSimulator` intercepts positions
before public perception; the controller rejects surviving coordinate fields.
Neither the image producer nor its public helpers open evaluator files.

## Replay and meaningful scoring

The producer freezes camera calibration, default temporal/multi-view thresholds,
all public inputs and inference source bytes before fitting. Horizontal focal is
estimated anew from each public rotation scan (179.934–179.988 observation pixels),
with anisotropic resize and pixel-grid centers. No SDK FOV is supplied to inference.
These small per-cell calibration differences are recorded, not hidden as identical
image preprocessing.

For each pair, inference records two alternatives without selecting a winner:

- Identity-preserving tracking through every intermediate frame, endpoint map,
  disjoint intermediate-frame PnP validation and fixed sixteen-trial sensitivity.
- Direct endpoint-only LK matches with the existing calibrated endpoint refit.
  This is an unvalidated diagnostic baseline, not an executable navigation policy.

Each independent component audit recomputes public acquisition, exact action
targets, image/reference ledgers, calibration and both complete predictions before
reading positions. All four component replays pass **30 checks**. The condition
auditor repeats these calculations rather than trusting saved audit flags, then
checks all four complete cells, shared configuration, source/input immutability
and matched physical trajectories. Its six checks pass at
`runs/controlled-motion-factorial-seed201-v1/audit.json`.

There is one independent session per light/weather cell. Every one of the 201
reported position/angle records is equal across all four cells (maximum absolute
component difference 0). This rules out a different physical path in these runs;
it does not supply positions to the estimator or establish broad causal invariance.
Live RGB randomness is not claimed byte-identical or statistically controlled by
multiple replications.

## Results

All conditions have eight actual moving trials and eight actual post-WAIT movement
pairs. The fully validated method rejects all moving-endpoint estimates. Results
for post-WAIT pairs are:

| Time | Weather | Surviving common tracks, median | Correct validated post-WAIT movement / 8 | Wrong supported movement |
|---|---|---:|---:|---:|
| Noon | Clear | 131.5 | 2 | 0 |
| Midnight | Clear | 170 | 1 | 0 |
| Noon | Thunder | 3 | 0 | 0 |
| Midnight | Thunder | 3 | 0 | 0 |

The noon/clear accepted estimates have direction errors 1.34° and 1.29°, rotation
errors 0.14° and 0.15°, and independently measured endpoint displacements 0.64
and 1.29 blocks. This is bounded positive relative-motion evidence, not a usable
trajectory, distance estimate or goal-reaching demonstration. Monocular translation
remains unit-scale; no absolute navigation scale is recovered.

Each condition has **13 actual stationary/rotation controls**, all correctly
rejected by the full method. One nominal late-WAIT control per session still moves
0.31 blocks, so it is scored as an actual movement failure, not a successful
negative control. Post-action waiting does not guarantee a settled body on uneven
terrain. The endpoint-only baseline also produces one false control translation
in each of midnight/clear and noon/thunder; those are not accepted by the protected
method. Unsupported real movement remains a failure in all reported accuracy and
coverage counts.

The two-by-two comparison localizes correspondence degradation to weather-related
rendering in this trajectory, rather than midnight alone: clear midnight retains
more common points than clear noon, while both thunder cells lose nearly all of
them. It does not distinguish rain particles, lighting changes during thunder,
occlusion or detector preference as the exact mechanism. More corners at the
initial frame do not imply more stable landmarks. One session per cell is not
sufficient for an uncertainty estimate or a general weather-robustness claim.

The remaining clear-scene coverage problem is separate. Public diagnostics show
parallax/coverage rejection in the initial minimal endpoint fit and local
sensitivity failures after refitting; even many surviving points do not ensure
an identifiable, stable pose. Do not solve that by declaring every consistent
homography to be translation or reducing reliability thresholds after scoring.

## Artifacts and reproduction

All directories below are under `runs/` and must remain separate:

| Cell | Raw acquisition | Inference |
|---|---|---|
| Noon/clear | `controlled-rgb-motion-seed201-clear-v1` | `controlled-motion-inference-seed201-clear-v1` |
| Midnight/thunder | `controlled-rgb-motion-seed201-thunder-v1` | `controlled-motion-inference-seed201-thunder-v1` |
| Noon/thunder | `controlled-rgb-motion-seed201-noon-thunder-v1` | `controlled-motion-inference-seed201-noon-thunder-v1` |
| Midnight/clear | `controlled-rgb-motion-seed201-midnight-clear-v1` | `controlled-motion-inference-seed201-midnight-clear-v1` |

From the repository root in the installed dedicated WSL visual environment:

```bash
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_controlled_rgb_motion.py --seed 201 --observation-size 448 --time noon --weather clear --output runs/controlled-acquisition-new
/opt/minecraft-scientist-visual/bin/python scripts/probe_controlled_rgb_motion.py runs/controlled-acquisition-new --output runs/controlled-inference-new
/opt/minecraft-scientist-visual/bin/python scripts/audit_controlled_rgb_motion.py runs/controlled-acquisition-new runs/controlled-inference-new --output runs/controlled-inference-new/audit.json
```

Repeat capture/inference with separate new directories for the remaining three
time/weather combinations. Then replay the complete matrix with
`scripts/audit_controlled_motion_conditions.py --case ACQUISITION INFERENCE` for
each of the four cells and a new `--output` JSON file. Existing targets are refused.
The local simulator may log optional authentication/texture/narration warnings;
these captures nevertheless complete and are verified from saved evidence.
No API key, screenshot upload or externally billed inference is used.

## Regression and next action

Windows geometry tests: **361 passed, two skipped**. WSL visual tests:
**363 passed, two skipped**. Clean default-only installation: **253 passed,
16 optional skips**. Targeted Ruff and both package builds pass. Tests cover
fixed schedules and budgets, all trial retention, public log isolation, exact
opaque target compliance, missing middle frames, calibration stops, source/input
guards, forged predictions, invalid telemetry, complete factorials and physically
incomparable cells. A Windows/WSL pytest-rewrite cache issue in deliberately hashed
mock functions was fixed by making their test-source paths native to the executing
runtime; production source-hash checks were not relaxed.

The previous core MVP regression remains byte-identical at
`runs/temporal-regression-mvp-seed0/benchmark.json`: 47 experiments, accuracy/task
success 1.0, detection delay one, eight repairs, three symbols/thirteen reuses,
library transfer three versus eleven interventions. The acquisition/analysis
scripts do not open accepted map databases or alter scientific learners.

Next actions are (1) preregister stationary-context background selection before
moving, preserving point identity and rejecting dynamic features without truth
labels; (2) investigate consensus refinement **before** minimal-pose rejection
while retaining positive-depth/parallax/control/sensitivity gates; (3) compare
against this complete negative/positive matrix and then acquire disjoint validation
trajectories before planner integration. Successful movement coverage must improve,
not just rejection reliability. Broad seed/terrain generalization, integrated
scientific/spatial exploration and long-operation recovery still remain mandatory.

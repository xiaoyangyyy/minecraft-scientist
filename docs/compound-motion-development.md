# Distinct-handle two-leg RGB acquisition (2026-10-02)

This optional branch tests bent motion rather than assuming different anonymous
control handles produce independent directions. It is development, not navigation
acceptance. The two existing development seeds are 200 and 201; neither is held out.

## Public execution and evaluation contract

`exploration/compound_motion.py` registers `paired-controls-v1`. It preserves the
canonical startup and twelve RGB reference turns, then executes fixed handle pairs
`(0,2), (2,1), (1,3), (3,0)` at four and eight ticks per leg. There is no inter-leg
wait; eight settle ticks follow each trial. Handles are opaque public targets,
not physical direction labels. The schedule does not claim perpendicular motion,
progress, absence of collision or arrival.

Defaults produce **50 actions, 225 RGB frames and 46 pairs**. Each of eight trials
retains combined moving, combined settled, first-leg, second-leg and late-WAIT
intervals; six additional pairs are rotation/WAIT/identity controls. Combined
windows fit the existing 32-image limit, within the 500-image acquisition cap.
Invalid/duplicate/self handle pairs and oversized windows reject. Failed or
stationary trials are not removed.

The thin wrapper temporarily binds the canonical capturer to the registered
configuration/schedule/pairs/timeline. Canonical sources are collected first, then
the compound module and wrapper are added: all seven raw sources, including the
SDK entry, are frozen before launch. Nested/noncanonical bindings reject; aliases
restore on exceptions. Shared SDK settings, camera, terrain and historical files
are untouched. Public validation requires the explicit sequence marker and checks
the exact schedule, opaque targets, frame/RGB ledger and current/snapshot hashes.

Inference reuses the unchanged camera calibration, chronological-prefix policy,
tracking, geometry and sensitivity thresholds. Every pair retains every configured
prefix, including failures: **164 fits per session**. First-supported local poses
are latched, not replaced using later or private scores.

The independent auditor replays all public calculations **before its first private
telemetry read**. It scores local and terminal poses separately, then characterizes
realized horizontal turns: both legs must exceed 0.15 blocks, with angle 45–135
degrees. Those thresholds are preregistered. Turn characterization is evaluator-only,
never acquisition or hypothesis-selection input. Private-file identity is checked
between scoring stages; changed inputs/sources, missing artifacts and invalid
declarations reject. Integrity passing is not a claim of estimator accuracy.

## Fresh natural sessions

Both noon/clear sessions complete all 225 frames and pass the raw capture gates.
Both full prefix reports pass **35 independent replay/integrity checks**. Protocol
and thresholds match, but physical trajectories and scenes do not. RGB inspection
shows trees/stepped ground in seed 200 and a steep ridge in seed 201; this is not a
formal biome classification or a completed terrain-generalization suite.

| Development world | Realized horizontal turns | Combined moving: correct local / terminal | Combined settled: correct local / terminal | First leg: correct local / terminal | Second leg: correct local / terminal |
|---|---:|---:|---:|---:|---:|
| Seed 201 | 3/8 | 0/8 / 0/8 | 0/8 / 0/8 | 0/8 / 0/8 | 1/8 / 0/8 |
| Seed 200 | 4/8 | 4/8 / 1/8 | 4/8 / 1/8 | 3/8 / 2/8 | 1/8 / 0/8 |

Denominators are scheduled intervals, not guaranteed physical movements. Seed 201
has two stationary terminal first-leg intervals and two stationary combined-settled
intervals; seed 200 has one and two, respectively. Abstaining on them is not a
successful movement estimate. Groups share eight interventions and cannot be summed
as independent successes. No candidate is promoted into navigation.

### Seed 201

Only `leg_two-07` selects a pose, at offset two rather than terminal eight. Local
direction error is **4.839°**, rotation error **0.114°**, scored distance 0.2495
blocks. Its terminal 1.4693-block movement is still unsupported. This does not
recover final leg displacement or combined motion.

Thirteen actual negatives in designated control groups reject. One late-WAIT
interval actually moves and remains a failed positive, not a negative success.
Across 164 prefixes, 137 reject endpoint geometry, 24 remain unstable, one lacks
common tracks and one fails intermediate geometry. Only one is supported. Endpoint
failures most often lack triangulated parallax/coverage. Distinct handles therefore
do not guarantee useful curvature or observable geometry in this session.

### Seed 200: coverage and an accuracy counterexample

Thirteen selected pair predictions contain twelve correct local motions and one
incorrect supported motion. Deduplicating identical `(first, selected-end)` image
intervals gives **eight intervals: seven correct and one wrong**, not thirteen
independent observations. Correct intervals occur in four intervention trials.
All fourteen actual designated negative controls reject.

The wrong case is **`leg_two-02`, frames 92→96**: direction error **26.176°**, rotation
error **0.457°**, physical distance 0.6629 blocks. It exceeds the fixed 20-degree
direction tolerance despite passing intermediate geometry and existing sensitivity
gates. Its selected prefix is also its terminal four-tick interval; the full-window
baseline has the same error. Do not attribute it uniquely to prefix selection,
conceal it behind integrity checks, or raise tolerance after scoring.

Several combined/settled/leg-one predictions share one first-leg interval, e.g.
frames 120→124 occur three times. Combined moving/settled terminal groups each
have one correct estimate. An early local success is not transported to a final
frame and does not prove that a later turn is recovered. Two development sessions
do not establish statistical superiority or held-out navigation generalization.

## Artifacts and reproduction

- `runs/controlled-rgb-compound-seed201-clear-v1`: first raw capture.
- `runs/rgb-compound-prefix-seed201-clear-v1`: predictions/source snapshots and final
  `audit-current.json`. Earlier `audit.json` retains the prior auditor hash and
  34-check schema; it is not relabeled as current evidence.
- `runs/controlled-rgb-compound-seed200-clear-v1`: second raw capture.
- `runs/rgb-compound-prefix-seed200-clear-v1`: predictions and final `audit.json`.

From the repository root in the installed WSL environment:

```bash
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_compound_rgb_motion.py --seed 201 --time noon --weather clear --output runs/new-compound-capture
/opt/minecraft-scientist-visual/bin/python scripts/probe_compound_rgb_motion.py runs/new-compound-capture --output runs/new-compound-prefixes
/opt/minecraft-scientist-visual/bin/python scripts/audit_compound_rgb_motion.py runs/new-compound-capture runs/new-compound-prefixes --output runs/new-compound-prefixes/audit.json
```

Outputs refuse overwrite. Geometry replay uses frozen WSL/OpenCV; public capture
validation also works with Windows visual extras. No API key/remote model is used.

## Regression and next action

Eleven schedule/binding tests and ten optional pipeline tests cover configuration,
source restoration, public validation, tampering, all-prefix-before-private
ordering, private identity changes, missing artifacts and physical turn scoring.
Synthetic stationary/90-degree fixtures test boundaries, not image accuracy.

Regressions: **517 tests/two skips on Windows**, **519/two on WSL**, **329/42
optional/platform skips** with clean default dependencies. Targeted Ruff and both
package builds pass. Current v6 scientific-core releases pass 30 seeds, two repeats
and eighteen exact artifacts on both platforms; all 91 frozen source hashes match.
The MVP seed-zero report is byte identical, with executable-symbol transfer using
three versus eleven probes.

The supported-direction counterexample changes the next action: develop independent
subsequent-frame validation of a latched local map/pose, retaining wrong/abstained
cases, and test observability on actually bent intervals. Do not simply relax gates,
drop stationary trials or promote an early local pose as final position. Metric/
topological motion integration, fresh held-out worlds/terrain/conditions, actual
goal arrival, natural scientific integration and at least 100 frontier/experiment
attempts with recovery remain open. The full objective is unchanged and incomplete.

# Public-only active parallax development

This is development on the existing natural seeds, not held-out navigation
acceptance. Public RGB directions select prospective anonymous control pairs;
neither the planner nor capture wrapper reads evaluator telemetry. The capture
profile binds the exact constructor, plan, source snapshots and raw artifacts.
Independent audit replays the past public predictions, action choice and all fresh
prefix computations before opening the new evaluator positions. All thresholds
and accepted navigation memory remain unchanged.

## V1: retained planning failure

`runs/active-parallax-plan-seed200-v1` replayed every prior public compound prefix.
The plan chose `(0,3), (3,0), (1,2), (2,1)` before acquisition. The known pair's
conditional expected turn was 162.355 degrees, with bounds [110.044,180], and
neither direction was robustly noncollinear. Handle 2 had one provisional sample;
the two-sample eligibility rule removed it from all geometric comparisons.
The positive-score exploitation rule nevertheless selected the near-opposite
known pair. Unknown-vs-unknown filling supplied no stable directional anchor.
These public limitations were recorded before execution, not inferred only after
seeing privileged positions.

The fresh source-bound noon/clear capture
`runs/controlled-rgb-active-parallax-seed200-clear-v1` contains 225 RGB frames,
50 action records and 46 pairs. Six profile checks pass. Its inference/audit in
`runs/rgb-active-parallax-prefix-seed200-clear-v1` passes 42 independently replayed
integrity checks. It does **not** establish useful active view acquisition:
all eight physical two-leg turns are 180 degrees, and **0/8** meet the fixed
45-to-135-degree criterion. Four/eight combined moving trials supply correct
selected local poses; twelve selected pair predictions are correct, but deduplication
gives only six image intervals. The fourteen designated controls have no false
translations. Seven compound-settled endpoints return close to their origin;
these are not seven failed intervention controls or seven independent successes.
No pose is promoted to navigation. No causal gain is inferred by comparing the
changed trajectory with previous schedules.

V1 snapshots are historical evidence once the selector evolves. Do not rewrite
their plan, reports, source hashes or failures to pass a current-source gate.

## Conditional future-view diagnosis

`runs/public-parallax-seed200-clear-v1` publicly replays all 46 prior shadow pairs
and measures future baseline/turn/withheld-ray diversity, conditional on the
original frozen reconstructed map. There are no ground-truth or executable pose
fields in this diagnostic. It is not independent accuracy validation, competing
map discrimination or a new navigation gate.

The retained wrong `leg_two-02` pose's future frames have median ray angles
0.196 and 0.335 degrees; conditional turns are 41.005 and 43.125 degrees.
Neither supplies the preregistered conjunction of baseline ratio >=0.25, turn
45..135 degrees and ray angle >=1 degree. Only one of the eight shadow-accepted
pair predictions meets this conditional criterion. Turning it into a pose gate
would discard most correct predictions as well; therefore it remains diagnostic.
The result motivates improved view acquisition, not an accuracy-repair claim.

## Next development requirement

Require eligible noncollinear evidence for exploitation. Keep single-observation
directions provisional but usable for experiment proposals. Explore unknown
controls against a directionally supported anchor, retain failures, and validate
the next plan on fresh trajectories before any promotion. Broad visual generalization,
goal arrival, natural scientific integration and long operation remain open.

## V2 selection contract

The default selector now labels conditional exploitation, provisional direction
resampling, anchored unknown-control exploration and unanchored coverage separately.
Exploitation requires at least two unique intervals per handle and the entire
conditional angular range inside 45..135 degrees. Resampling may use a single
observation: its score includes the overlap with that angular window and reduced
support weight. This proxy is not expected Shannon information gain or a calibrated
probability. A single observation never certifies exploitation or navigation.

Reserved exploration prefers an unknown control against a two-sample anchor,
ranking anchors by sensitivity/spread and balancing exposures of unknown handles.
Where no such anchor exists, declared coverage exploration remains uncalibrated.
This does not guarantee turns when the evidence is weak. The capture/geometry
budgets, physical turn criterion, error tolerances and first-supported latch remain
identical to V1. Current plan validation rejects V1 as historical and rechecks prior
input/source/snapshot bytes after the full public replay, before executing V2.

Seed 200's V2 prospective pairs are `(0,2), (2,0), (0,1), (1,0)`: two provisional
resampling choices and two anchored unknown experiments, zero exploitation choices.
The initial public angle for `(0,2)` is 44.466 degrees, with sensitivity/spread
bounds overlapping the desired range; it is explicitly uncertain, not promoted.
Seed 201 has no two-sample anchor. Its V2 pairs are `(1,2), (1,3), (2,3), (2,1)`;
all four are labeled coverage exploration. Both plans replay their complete past
public data; development is still restricted to seeds 200/201.

### Fresh seed-200 V2 result

`runs/controlled-rgb-active-parallax-seed200-clear-v2` passes all acquisition and
six profile checks, with the same 225-frame/50-action/46-pair budget. Its complete
164-prefix public recomputation and private scoring in
`runs/rgb-active-parallax-prefix-seed200-clear-v2/audit.json` pass all 42 checks.
Six/eight trials now meet the fixed physical turn criterion, versus zero/eight
in V1. Two first legs have zero horizontal displacement and remain failed turn
trials. The observed turns are 70.438..90 degrees. This descriptive acquisition
improvement is one development session, not held-out superiority or a pose gain.

| Seed 200 V2 group | Correct selected local movements | Wrong selected local movements | Trials |
|---|---:|---:|---:|
| Combined moving | 3 | 0 | 8 |
| Combined settled | 3 | 0 | 8 |
| First leg | 2 | 0 | 8 |
| Second leg | 2 | 3 | 8 |

Thirteen selected pair predictions contain ten correct/three wrong predictions;
they deduplicate to eight image intervals, five correct/three wrong. The fourteen
designated stationary/rotation controls have no false translations. All eight
compound endpoints are physically moving; two first-leg endpoints are stationary.
Early selected local poses do not recover final displacement. Terminal combined
moving coverage is only one/eight, and settled terminal coverage zero/eight.

The new wrong second-leg intervals are retained: frames 60→64 (23.628-degree
direction error), 128→130 (45.389 degrees), and 176→178 (37.599 degrees). Rotation
errors are respectively 0.335, 0.261 and 0.289 degrees. The fixed 20-degree
direction/5-degree rotation criteria are unchanged. These failures disprove the
claim that acquiring turns alone makes the current geometric gate reliable.
No candidate is promoted to navigation and no new default pose gate is added.

### Fresh seed-201 V2 result: no useful recovery

The unchanged selector also completes a fresh seed-201 225-frame capture and
independent full 164-prefix replay. Both its six capture-profile checks and
42 final audit checks pass. Artifacts are
`runs/controlled-rgb-active-parallax-seed201-clear-v2` and
`runs/rgb-active-parallax-prefix-seed201-clear-v2/audit.json`.
There are **zero/eight** effective turns and **zero** selected local motion poses
in every group. All eight combined moving/settled endpoints are actual movement,
so zero pose coverage is not negative-control success. Fourteen designated controls
reject; this does not repair the missing movement estimates. Three turn trials
lack a sufficiently moving leg; the other angles are 0, 42.862 or 180 degrees.
The unchanged lower bound is 45 degrees; the near-boundary case is not relabeled.

Public compound-moving endpoint failures include eleven triangulation/parallax/
coverage failures, thirteen epipolar-refinement failures, five refined positive-depth
spatial failures, three initial positive-depth failures and four pure-rotation/
stationary ambiguities. Three additional fits fail sensitivity and one lacks
withheld common tracks. These are algorithmic diagnostics, not proven unique
physical causes. Raw images show different natural scenes: seed 200 near forest
paths and seed 201 on a steep, tree-covered slope. Scene inspection does not
establish a causal explanation or a benchmark terrain-class label.

The singleton seed-201 hypothesis is handle 0, but V2's unanchored fill never
retests it: least-sampled pair preference repeatedly uses the other three handles.
Thus labeling a phase "coverage exploration" does not guarantee control coverage
or revisiting a weak anchor. This retained public planning limitation changes the
next action: improve terrain-aware, sequential experiment design and explicitly
resample lone provisional anchors, alongside competing-map/pose discrimination.
Do not extrapolate seed 200's 6/8 turning result to the second world, claim useful
navigation from zero incorrect output, or replace failed worlds with easier scenes.

## Reproduction

Run with the installed WSL visual runtime from the repository root, using a new
output path for every step:

```bash
/opt/minecraft-scientist-visual/bin/python scripts/plan_active_rgb_motion.py runs/controlled-rgb-compound-seed200-clear-v1 runs/rgb-compound-prefix-seed200-clear-v1 --output runs/new-active-plan
timeout 600 /opt/minecraft-scientist-visual/bin/python environments/minestudio/capture_active_parallax.py --plan runs/new-active-plan --output runs/new-active-capture
/opt/minecraft-scientist-visual/bin/python scripts/probe_active_rgb_motion.py runs/new-active-capture --output runs/new-active-predictions
/opt/minecraft-scientist-visual/bin/python scripts/audit_active_rgb_motion.py runs/new-active-capture runs/new-active-predictions --output runs/new-active-predictions/audit.json
```

No API key or external model is used. The raw source snapshots, `.active-profile`,
plan and inference snapshots are required together for the active audit. It
replays all past public pixels again before scoring; private files are never
candidate inputs. V1 artifacts require their historical source revision, not a
rewritten manifest. All outputs refuse overwrite.

## Engineering verification

Thirty-eight new tests cover proposal roles, negative near-opposite cases, single
sample proposals, duplicate/mixed image evidence, synthetic camera geometry,
constructor/alias restoration, public-before-private ordering, plan/profile
tampering, post-replay byte changes and public-only diagnostic failure retention.
Fixtures are correctness tests, not substitutes for natural-scene accuracy.
Full regressions pass Windows 576/two skips, WSL 578/two, and clean default-only
347/49 optional/platform skips. Targeted Ruff and source/wheel builds pass.
Both V8 thirty-seed/two-repeat scientific-core releases freeze 94 current sources
and reproduce eighteen artifacts per seed; MVP seed-zero bytes are unchanged.
These checks do not complete any outstanding navigation or full-project gate.

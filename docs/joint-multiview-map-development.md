# Joint multi-view map development

This follow-up implements joint estimation rather than another threshold change.
It preserves the original open-world roadmap. The current result is a bounded
development improvement and an important remaining counterexample, not working
goal navigation, a unique map, or held-out world generalization.

## Estimator and evidence separation

`exploration/joint_multiview_map.py` fits a bounded complete past RGB track chain.
Raw landmark IDs are partitioned before any geometric estimation. Bootstrap
essential fitting, initial PnP, and joint optimization use only training IDs.
Reserved validation pixels cannot influence camera poses or training points.

The optimizer simultaneously adjusts all past camera poses and training 3D points
using Huber-weighted reprojection residuals, analytic left-SE(3)/point Jacobians,
damped Schur elimination and strictly improving steps. The first camera is fixed;
the bootstrap translation norm fixes monocular scale. Neither simulator coordinates
nor metric blocks enter fitting. Intrinsics remain the public calibrated camera.
There is no new SciPy dependency. OpenCV's world-to-camera projection and PnP
conventions follow its [official documentation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).

Withheld points are triangulated only after training pose optimization, with those
poses fixed. All their past-view residuals, positive depths and spatial coverage
must pass the existing geometry thresholds. This is independent point validation,
not independent-time validation: the subsequent control views remain essential.

The diagnostic enumerates every bootstrap offset 2 through the completed past
terminal. Every map is frozen before any future image tracking. No private-pose
winner is selected, no validation-error winner is substituted, and no future view
can rerun joint optimization. Competing local minima remain separate records.
Only surviving original fitting/withheld IDs can enter new-view PnP/prediction;
there is no reseeding, resurrection or replacement. The original 32 total / 12
training / 8 validation floors, spatial checks and pixel thresholds are unchanged.

New-view camera PnP uses the surviving original training points only. Withheld
points remain fixed. Public inference and auditing retain all eight original trials,
including the two original acquisition aborts, every chronological prefix and
every candidate. Map fields always state `navigation_accepted=false`.

## Source-frozen development replay

The historical 203-frame RGB/receipt acquisition is
`runs/controlled-rgb-second-shadow-seed201-clear-v1`. No new Minecraft actions,
terrain edits, provider calls or keys are used in this replay. Its source snapshots,
config hash, contiguous PNG/RGB ledgers, all compliant action/frame receipts,
reference images/calibration and fitting/shadow MOVE bounds are independently
checked. This reader does not claim to replay the complete old adaptive controller
or to turn its historical source set into a current acquisition. Its earlier
controller/map audit remains preserved with the V17 source boundary.

Current component output is `runs/joint-maps-seed201-v3`, with 101 inference source
entries. It completes all eight records and **36** registered bootstraps. A
separate audit reexecutes the complete public fit and future validation before
first reading private positions. All **11** component provenance/replay checks
pass, including unchanged public/evaluator bytes through scoring. These eleven
checks have a different scope from the earlier 29-check adaptive-controller audit;
they must not be compared as interchangeable acceptance gates.

V1 imposed an unnecessarily strict all-landmarks-survive condition. Its four
internally consistent maps could not receive a future check even when the unchanged
original support floors remained satisfied. V1 and its completed audit are retained.
V2 restores the original-partition support floors without replacing or refitting
any point. V3 leaves the fitter and future tests unchanged, adding physical scoring
of both anchor-to-future and frozen-endpoint-to-future intervals. The V2 audit is
historical after that script-only scoring addition. No manifest is overwritten.

## Results and remaining counterexample

Four internally consistent maps represent **three** distinct past intervals from
eight trials. Their direction errors are within the unchanged 20-degree gate:

| Trial / fitting interval | Bootstrap offset | Past direction error | Original future support | Future pixel result |
|---|---:|---:|---|---|
| 0 / 56→65 | 7 | 3.859847° | 8 training + 14 validation; insufficient | Unavailable |
| 0 / 56→65 | 9 | 9.415539° | 15 training + 14 validation; insufficient total | Unavailable |
| 1 / 77→84 | 7 | 10.302308° | 42 training + 24 validation | Inconsistent |
| 3 / 107→113 | 5 | 18.492276° | 54 training + 32 validation | Consistent |

The earlier fixed two-view compound-terminal model on the same 77→84 images had
21.754527° direction error. The new joint candidate on that interval has
10.302308°, but it still fails subsequent views (withheld 95th percentile residuals
4.291392 and 4.466402 pixels). This single matched-interval development observation
does not establish a benchmark-wide gain: initialization, seed/partition and
estimation procedure differ, and no global performance comparison is claimed.

The trial-3 candidate passes both future pixel tests with 32 surviving withheld
points, acceptance 1.0, spatial coverage four/five cells, and 95th percentile
residuals 1.339860/2.500560 pixels (the unchanged 448-pixel gate is 3 pixels).
Its anchor-to-future direction errors at frames 115/117 are 19.327299°/15.185079°.
Those whole-path directions hide a more important incremental error:

| Future interval | Physical displacement | Incremental direction error | Result |
|---|---:|---:|---|
| Frozen endpoint 113→115 | 0.290298 blocks | **33.042212°** | Wrong supported motion |
| Frozen endpoint 113→117 | 0.648979 blocks | 15.812530° | Correct under current gate |

Thus pixel consistency and correct aggregate direction still do not prove correct
short-step navigation. No map is promoted and no navigation attempt is counted as
successful. Repeated candidates of one interval are not independent successes;
the original two aborted trials and all rejected initializations remain visible.
All data are from a development world previously examined for the preceding method,
not a disjoint held-out world.

## Verification

Forty-seven focused tests pass. They include analytic Jacobian finite differences,
synthetic bent-trajectory and future pose recovery, monotonic noisy-training cost,
fixed origin/scale, deterministic replay/storage-order invariance, camera/input
immutability, validation-only corruption leaving fitted poses unchanged, no future
bundle/refit, original support-floor rejection, bounded bootstrap/seed validation,
future-prefix pixel integrity, all-candidate-before-future ordering, all-trial abort
retention, forged input/source/config/map rejection before private reads, relative
pose composition and byte mutation during scoring.

Both V18 core releases pass all six checks over thirty seeds/two repeats/eighteen
exact artifacts. Their 101 package/verifier/configuration hashes were rechecked
with zero mismatches. The seed-zero MVP output remains byte-identical with SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
Full regressions pass **858 tests/two skips on Windows**, **860/two on WSL**, and
**443/66 optional or platform skips** in the isolated default-only offline suite.
JUnit reports are `runs/joint-map-regression-{windows,wsl}-v1.xml`,
`runs/joint-map-regression-default-offline-v1.xml` and the 47-case
`runs/joint-map-targeted-v6.xml`, all with zero failures/errors. Related geometry
tests pass separately on both backends, and Ruff/format plus both builds pass. Unit success
does not negate the real incremental-direction counterexample.

## Next engineering action and full scope

Investigate short-baseline pose uncertainty and unmodeled camera/projection changes
using public pixels and installed runtime configuration.
Check renderer-camera versus player telemetry and tick alignment as possible
observation-model mismatches, without treating them as demonstrated causes.
Evaluate training-only
perturbation ensembles or explicit projection-model alternatives before further
navigation integration; preserve fitting/selection/future-validation separation.
Do not select a model by private direction error, ignore the midpoint, substitute
aggregate direction, or relax the error gate. Fresh disjoint acquisition is needed
before any method promotion, with all failure denominators retained.

The original requirements still include five additional worlds/multiple terrains/
three conditions, reliable goal navigation, natural science/map integration,
at least 100 attempts and recovery checks, live/baseline release revalidation and
CI consolidation. The broad goal remains active.

Subsequent fresh runtime instrumentation, documented in
`docs/runtime-camera-timing-development.md`, confirms a base-camera/player-position
sampling mismatch using uniquely matched native RGB bytes. It does not supply
missing camera metadata to these historical images, change the 33.042212-degree
counterexample, or promote any joint map. Capture-time camera-aware evaluation is
now required before attributing all residual short-motion error to the estimator.

## Reproduction

Use the dedicated WSL environment from `/mnt/d/minecraft-scientist` and new outputs:

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/probe_joint_maps.py runs/controlled-rgb-second-shadow-seed201-clear-v1 --output runs/new-joint-maps
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/probe_joint_maps.py runs/controlled-rgb-second-shadow-seed201-clear-v1 --audit runs/new-joint-maps --output runs/new-joint-maps/audit.json
```

Outputs are refused if already present. Public inference never opens the evaluator.
Audit opens it only after complete public replay; public/privileged roles and source
boundaries remain explicit. This is saved-image computation, not new live capture.

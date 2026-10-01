# Dense RGB correspondence development

This is development on natural seed 201, not held-out navigation acceptance.
All candidates remain outside the default planner and accepted map. The full
project gates in `quality-roadmap.md` remain incomplete.

**Historical camera-model caveat:** the original results below used an incorrect
equal-axis focal assumption for a widescreen image stretched to a square. The
public resize correction and temporal follow-up are in
`temporal-motion-development.md`. Corrected dense runs score 3/8 MOVE at both
resolutions, with 4/5 controls at 224 and 5/5 at 448; neither accepts navigation.
Do not attribute all earlier failures solely to the correspondence model.

## Method and public input boundary

The classical branch had too few stable correspondences on the night/thunder
route. The optional adapter evaluates the outdoor
[LoFTR model](https://github.com/zju3dv/LoFTR) using the
[Kornia API](https://kornia.readthedocs.io/en/latest/models/loftr.html).
Inference is local and CPU-only: no screenshot upload, API credential, semantic
Minecraft name, coordinate, yaw or hidden FOV enters perception.

`dense_correspondence.py` crops the world ROI before grayscale, equalization,
interpolation and attention. Padding is masked and independently excluded from
coordinates. Pixel-center restoration, finite/score checks, deterministic ordering,
two-image deduplication and match budgets are explicit. Scores are not probabilities
of correct landmark identity. The backend verifies local checkpoint bytes before
safe `weights_only=True` loading, and never invokes automatic SDK downloads.

The dedicated WSL runtime uses Kornia 0.8.3, Kornia-rs 0.2.0, PyTorch 2.8.0+cpu and
OpenCV 4.8.0. Default project dependencies are unchanged. Kornia's mask-resize path
requires floating masks; the failed boolean-mask run is preserved at
`runs/dense-correspondence-seed201-224-v1`.

The explicitly downloaded checkpoint is 46,341,978 bytes with SHA256:

```text
21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec
```

`runs/pretrained-loftr-outdoor-v1/download.json` records its HTTPS URL, size,
digest and code-license reference. The fetcher has a bounded streaming download
and refuses existing output directories. Weights are not redistributed; the
code-license reference is not a separate assertion about weight-distribution rights.

## Frozen experiment and independent evaluation

The public action timeline selects the first four and last four MOVE pairs, one
WAIT pair, two LOOK pairs and two identical-frame controls. Pair selection,
thresholds, source snapshots and input hashes are frozen before prediction.
Focal length comes from the public reference rotation scan, not telemetry;
intrinsics are scaled to the raw image dimensions.

`two_view_motion.py` checks coverage, rotation-homography ambiguity, essential-model
cheirality, triangulation, reprojection and parallax. Geometric support is not
motion accuracy. The auditor independently replays the verified PNGs, checkpoint
and geometry before opening evaluator-only positions. An actual movement is at
least 0.15 blocks; correctness additionally requires direction error at most 20
degrees and rotation error at most 5 degrees, frozen before the first score.
Player pose only approximates rendered camera pose, including view bobbing; this
does not certify trajectory continuity or goal arrival.

The strengthened auditor rejects incomplete frame/reference ledgers, failed focal
calibration and misaligned evaluator data before dependent operations, and checks
source/manifest stability after replay. All non-finite evaluation thresholds are
rejected so NaN cannot convert a real movement into a correct control.

| Run under `runs/` | Correct MOVE / 8 | Correctly rejected controls / 5 | Audit checks | Result |
|---|---:|---:|---:|---|
| `dense-correspondence-seed201-224-v2/audit-v2.json` | 3 | 5 | 23 | Four supported movement estimates are wrong |
| `dense-correspondence-seed201-448-v1/audit-v2.json` | 1 | 4 | 23 | Higher resolution still fails; WAIT false translation |
| `motion-stability-seed201-224-v2/audit.json` | 0 | 5 | 16 | All motion unsupported or unstable |
| `motion-stability-seed201-448-v2/audit.json` | 0 | 5 | 16 | All motion unsupported or unstable |
| `refined-motion-seed201-224-v2/audit.json` | 0 | 5 | 17 | No supported refined geometry |
| `refined-motion-seed201-448-v2/audit.json` | 0 | 5 | 17 | No supported refined geometry |

All six audits validly reproduce their results; none accepts navigation. The two
raw routes are fresh sessions of the same seed/route, not independent terrain
generalization. Baseline inference supports seven and five MOVE estimates
respectively. One wrong 224-pixel estimate has 173.792 degrees of direction error.

## Sensitivity and precision repair

`motion_stability.py` repeats 16 deterministic 80% point-subset fits, perturbing
coordinates by at most 0.25 reference pixels and focal length by at most 1%.
It requires at least 80% supported fits and 95th-percentile direction/rotation
deviations at most 20/3 degrees. These are sensitivity statistics, not calibrated
confidence or truth: a coherent false correspondence field could still pass.
Rejected results cannot expose an executable pose.

For the almost-reversed 224-pixel estimate, the direction deviation reaches
149.121 degrees at the 95th percentile. The false WAIT translation at 448 pixels
has only 1/16 supported sensitivity fits. Refusing these estimates is appropriate,
but refusing all movement is not useful navigation or a success claim.

`correspondence_refinement.py` separately tests bounded LK tracking initialized
from learned matches, reverse-cycle agreement, normalized local-patch correlation,
texture support and target deduplication. Everything operates on the cropped ROI;
invalid forward tracks cannot seed reverse tracking. Seed indices and original
uncalibrated scores remain explicit provenance.

MOVE pairs retain 0–19 matches at 224 pixels and 0–26 at 448 pixels. They fail
count, coverage or triangulation requirements even before sensitivity checking.
WAIT controls retain 524/508 matches and correctly reject geometry. This motivates
investigating correspondence quality and temporal baselines, but does not identify
the exact cause of every failure or justify relaxed thresholds.

Sensitivity/refinement producers never open evaluator files; separate auditors
reproduce their frozen public calculations first. Initial v1 runs remain historical;
v2 captures the final strengthened audit-helper source and is the current replay
target. Historical artifacts identify their own source hashes: use frozen code
when replaying changed helpers, not newer code attributed to an older run.

## Reproduction

From the repository root, use the dedicated visual runtime. All output targets must
be new. The fetch command is an explicit download; inference is local. Existing
PyTorch/OpenCV versions above are required for matching audited results.

```bash
/usr/local/bin/uv pip install --python /opt/minecraft-scientist-visual/bin/python kornia==0.8.3 kornia-rs==0.2.0
/opt/minecraft-scientist-visual/bin/python scripts/fetch_loftr_checkpoint.py --output runs/loftr-new
/opt/minecraft-scientist-visual/bin/python scripts/probe_dense_correspondence.py runs/dense-visual-route-seed201-v1 --checkpoint runs/loftr-new/loftr_outdoor.ckpt --sha256 21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec --render-size 640 360 --output runs/dense-new
/opt/minecraft-scientist-visual/bin/python scripts/audit_dense_correspondence.py runs/dense-visual-route-seed201-v1 runs/dense-new --checkpoint runs/loftr-new/loftr_outdoor.ckpt --output runs/dense-new/audit.json
/opt/minecraft-scientist-visual/bin/python scripts/diagnose_motion_stability.py runs/dense-new --output runs/stability-new
/opt/minecraft-scientist-visual/bin/python scripts/audit_motion_stability.py runs/dense-visual-route-seed201-v1 runs/dense-new runs/stability-new --output runs/stability-new/audit.json
/opt/minecraft-scientist-visual/bin/python scripts/probe_refined_motion.py runs/dense-visual-route-seed201-v1 runs/dense-new --output runs/refined-new
/opt/minecraft-scientist-visual/bin/python scripts/audit_refined_motion.py runs/dense-visual-route-seed201-v1 runs/dense-new runs/refined-new --output runs/refined-new/audit.json
```

The SHA above locks the observed checkpoint, not future upstream URL contents.
A changed checkpoint requires separate provenance, not bypassing verification.
The current producer corrects camera resize geometry; the historical numbers
above require their source snapshots. Current reproduction does not recreate
those legacy calibration assumptions.

## Regression and next action

Windows: 293 tests pass, two expected optional/platform skips. WSL visual runtime:
295 pass, two Node/platform skips. A clean `uv run --isolated pytest` environment
with no optional visual/model packages passes 227 with 11 expected skips. Targeted
Ruff, source distribution and wheel build pass. Fifty-two added cases cover public
ROI isolation, restoration/filtering, model identity, synthetic geometry/controls,
motion conventions, finite thresholds, sensitivity, refinement and audit stop gates.
OpenCV versions can select different minimal-set geometry; tests assert declared
stability/accuracy bounds, not cross-version numerical equality. Precisely generated
noisy synthetic points use a correspondingly tight RANSAC fit threshold.

`runs/dense-regression-mvp-seed0/benchmark.json` still reports accuracy/task success
1.0, 47 experiments, detection delay 1, eight revision experiments, three symbols
and 13 reuses. Library transfer remains 3 versus 11 interventions. These offline
read-only visual experiments do not mutate the accepted topological memory.

Next: use the intermediate raw RGB frames to track stable identities over short
adjacent steps, triangulate after accumulated parallax, then jointly validate
multi-frame reprojection and trajectory consistency. Compare against these negative
baselines and preregister before fresh disjoint trajectories. Explicitly test
temporal/static-world filtering if weather/animation dominates, without evaluator
labels. Useful movement accuracy **and** coverage must improve before goal-search
integration. Broad seed/terrain generalization, combined scientific/spatial
exploration and longer-operation gates remain required for overall completion.

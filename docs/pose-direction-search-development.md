# Direction-constrained competing maps and reference replay

This is a development observability experiment, not a promoted pose estimator or
navigation gate. It closes a specific weakness of the previous finite rotation
grid: the zero-offset training refit was not a replay of the original map.
The original project still requires reliable held-out navigation, integrated
natural-world science/map learning, long operation/recovery and release comparison.

## Fixed public-only computation

`exploration/pose_direction_search.py` first triangulates with the **unchanged**
original endpoint rotation/translation. It preserves the full track ordering,
original training/validation IDs and original PnP seed, then recomputes every
intermediate-frame check. Counts, flags and numerical values must match the
original checks (reason strings are descriptive, not compared). A failed or
nonmatching reference retains its evidence, starts no alternative search, and
is explicitly uninformative rather than evidence for uniqueness.

The alternative family fixes camera-center directions on two cones, 25 and 45
degrees from the original direction, at eight deterministic tangent azimuths
each. Sixteen trials are retained regardless of success. For each direction,
a bounded 16-iteration finite-difference Levenberg-Marquardt fit adjusts rotation
by at most three degrees, minimizing Huber-weighted signed Sampson residuals.
Only original **training endpoint rays** enter this fit. Translation is
`-R @ center`, so optimization cannot change the requested center direction or
select its sign using validation points. Accepted steps must reduce the objective.
No new dependency, API key or simulator telemetry enters this computation.

Each fitted map must pass the unchanged 0.5-reference-pixel training error-95
threshold (one pixel at 448px width), retain every original training/validation
identity under positive-depth, one-degree parallax and endpoint reprojection,
and pass all original withheld intermediate pixels using training-only PnP.
A validated alternative exceeding the fixed 20-degree direction difference is a
witness. Neither its pose nor the original pose is exposed as a new executable
motion by the diagnostic. No witness still does not prove uniqueness.

This remains an incomplete search, with important limits: it has only two cones,
eight azimuths, a local rotation optimizer and a fixed three-degree bound. Original
point partitions inherit the baseline's endpoint selection; withheld intermediate
points are not an independent future acquisition. The complete-identity/parallax
demands can exclude possible explanations before their withheld views are tested.
Observed ambiguity is not itself a ground-truth error: a correct pose can be
insufficiently identifiable from these pixels.

## Replay and provenance

`pose_direction_common.py` reexecutes the entire public active acquisition/prior
plan, all 46 original pair computations and all 164 keyframe prefixes through the
existing competing-map pipeline. Only then does it inspect the verified frozen
selected candidates. The producer freezes public inputs, original snapshots,
current producer/auditor/evaluator sources and the fixed configuration; partial
search failures retain completed pair records. It refuses existing output paths.

`audit_pose_direction.py` independently repeats every public reference replay,
fit and validation before any baseline private scoring. It verifies all bytes
before and after scoring, complete pair/interval identities, and evaluator identity.
Tampered/incomplete evidence receives no evaluation. An optional-veto comparison
only removes original selected predictions; it never replaces a candidate,
rescues an unsupported interval or updates accepted memory. The older finite-grid
results are also retained per pair, not silently replaced by the new family.

## Real development result: better search, still no accuracy repair

The existing fresh V3 seed-201 noon/clear acquisition is reused; this is **not**
new acquisition or held-out validation. Source-bound artifacts are:

- `runs/public-pose-direction-seed201-clear-v1/config.json`
- `runs/public-pose-direction-seed201-clear-v1/report.json`
- `runs/public-pose-direction-seed201-clear-v1/audit.json`

The producer completes with unchanged inputs/sources. All twelve selected pair
records exactly reproduce their original map checks; zero reference failures
occur. The independent audit passes all **54** checks. The original twelve records
contain eleven correct/one wrong; nine unique intervals contain eight correct/one
wrong. The new search finds witnesses in three originally **correct** records:

| Pair | Selected interval | Valid alternatives | Original direction error |
|---|---|---:|---:|
| `leg_two-04` | 128→130 | 7 | 5.871942° |
| `compound_moving-05` | 144→148 | 7 | 3.789562° |
| `compound_settled-05` | 144→148 | 5 | 3.789562° |

The moving/settled results in trial 05 share image endpoints and are **not**
independent successes. A new-family-only veto would leave eight correct/one wrong
pair records, or six correct/one wrong unique intervals. It loses positives but
does not reject the known wrong 152→154 pose, whose error remains 32.412640 degrees.
Of that interval's sixteen alternatives, eleven fail the training error gate and
five fail complete original endpoint retention; none reaches intermediate PnP.
These counts do not isolate whether tracking, the imposed map family or the
baseline's endpoint selection causes the surviving error.

The old finite grid's correct 168→170 ambiguity witness is not found by these
different fixed cones. Absence here does not invalidate that earlier witness.
Combining both families as vetoes would remove four correct pair records (three
unique intervals) while still retaining the wrong direction. Neither veto is
promoted. All stationary/control failures remain unchanged in the original audit.

## Tests and reproducibility

Twenty-eight new tests cover exact original-map replay, bounded monotonic fitting,
fixed direction, deterministic grids, reference/validation corruption, a known-pose
planar witness, complete original IDs, schemas, public-before-private order,
tampering, post-score byte changes, partial failure and overwrite refusal. The
known-pose synthetic fixture does not claim the baseline selects that map.

Full regression: Windows **632 passed/two skips**, WSL **634/two**, clean
default-only **361/53 optional or platform skips**. Both V10 core releases freeze
96 source entries and pass thirty seeds/two repeats/eighteen exact artifacts;
MVP seed-zero report bytes remain unchanged. These tests verify engineering and
finite scientific-core behavior, not reliable natural-world motion.

Run from the repository root with a new output directory at each step:

```bash
/opt/minecraft-scientist-visual/bin/python scripts/probe_pose_direction.py runs/controlled-rgb-active-parallax-seed201-clear-v3 runs/rgb-active-parallax-prefix-seed201-clear-v3 --output runs/new-direction-diagnosis
/opt/minecraft-scientist-visual/bin/python scripts/audit_pose_direction.py runs/controlled-rgb-active-parallax-seed201-clear-v3 runs/rgb-active-parallax-prefix-seed201-clear-v3 runs/new-direction-diagnosis --output runs/new-direction-diagnosis/audit.json
```

## Next work

Do not solve the retained error by increasing the physical tolerance or calling
more abstention navigation success. Evaluate truly subsequent, independently
acquired, non-collinear views against frozen alternative maps, recording which
original identities survive and whether reference and competing predictions are
distinguishable. Investigate the public tracking/endpoint-retention boundary of
the retained wrong interval without choosing a replacement by private accuracy.
Sequential terrain-aware experiments and broad held-out validation remain needed.

## Subsequent source-registration correction

The artifacts above freeze revision `2e8b5f0` and its recorded WSL runtime. A
later cross-platform check found that a decorated compound binding's inference
source entry pointed to stdlib `contextlib.py`, not the project module. The raw
capture ledger separately includes the actual project module, but the mislabeled
entry did not fulfill its intended source contract. `inspect.unwrap` now repairs
that locator; old manifests are preserved and intentionally fail current-source
checks. See `docs/source-ledger-development.md` for tests, corrected prior replay
and new acquisition boundary. Do not present the historical 54-check computation
as a current-source visual release after this correction.

The corrected-source subsequent action-switch experiment is documented in
`docs/action-switch-motion-development.md`. All 54 checks pass, but the wrong
direction loses all tracked points before its holdout views; several correct
maps also fail. This is not a navigation repair. Next acquire diverse views while
the original identities are still visible rather than extending fixed waits.

# Stable-keyframe geometry research

Latest complete public-image bridge diagnosis is recorded in
`docs/long-wait-public-bridge-diagnostic.md`. The safe-seed bridge probe explicitly
binds four registered capture schemas and all five used module snapshots, retains
partial terminal controls, and compares only complete bounded intervals. Windows
and Linux each pass 31 research tests. Four actual known-development reports retain
46 original trials and compare 38 raw bridges; none reaches continuous support.
This does not promote a tracker or establish navigation.

The next sustained-motor candidate is preserved, not applied, in
`sustained_motor_candidate.patch`. Its registered eight one-tick receipts in
acquisition/future retain all prior geometry/global gates. Both fresh independently
audited development captures still acquire only their initial view and no map.
See `docs/sustained-motor-keyframe-development.md` for source separation, retained
partial controls and exact verification scope; main V35 full counts do not cover it.

This directory is an unreleased development probe outside the package/default
test discovery. It does not change the frozen V28 source, policies, captured
failures, default dependencies or navigation acceptance.

## Training optimizer comparison

`ceres_fit_probe.py` substitutes a process-local PyCOLMAP/Ceres training bundle
adjuster, restoring its alias on success or exceptions. Only original surviving
training rays enter reconstruction. Partitions, withheld-point triangulation,
fitting budget, Huber scale, public intrinsics, origin and bootstrap unit remain
unchanged. PyCOLMAP 4.0.1 runs one solver thread and at most thirty seconds per solve.
The probe only accepts the registered centered 448-pixel sensor.

No private camera/player files are opened. Public-parent/image hashes, frozen
lineage, source snapshot, backend extension identity and unchanged package sources
are checked/recorded. The two already observed endpoints are tested after fitting
freezes. They are known development counterexamples, not unused holdouts or new
live interventions, and cannot rescue the original failed version.

The [official COLMAP API](https://colmap.github.io/pycolmap/index.html) and
[4.0.1 documentation](https://pypi.org/project/pycolmap/4.0.1/) inform the adapter.
The Linux wheel is isolated in ignored `runs/pycolmap-prototype-deps-v1`, not
installed in Windows or either existing runtime. On this Windows/WSL host:

```text
uv pip install --target runs/pycolmap-prototype-deps-v1 --python-version 3.11
  --python-platform x86_64-manylinux_2_28 --only-binary :all: --no-deps pycolmap==4.0.1
```

Use the existing Linux visual Python with process-local
`PYTHONPATH=/mnt/d/minecraft-scientist/runs/pycolmap-prototype-deps-v1`:

```text
python research/settled_keyframes/ceres_fit_probe.py
  runs/shared-endpoint-live-seed202-v1 --output runs/<unused-probe-directory>
python -m pytest research/settled_keyframes/test_ceres_fit_probe.py
```

Five separate research tests pass in `runs/ceres-probe-tests-wsl-v1.xml`: input
immutability, origin/unit gauge/reprojection, sensor/origin refusal, invalid depth,
and exception restoration. They are not included in the package test counts and
do not establish physical control.

## Negative result

`runs/ceres-same-evidence-seed202-v2` and `-v3` produce identical full reports,
SHA-256 `B45A813512809E1B87C73E8DB9DF20CD8EAF1C2187F42E6DEE5ADE3662AAF23B`.
All four input/source/restoration checks pass. Initial `-v1` remains historical
with its source snapshot after the provenance checks were added.

| Model / measurement | Existing optimizer | Ceres probe |
|---|---:|---:|
| Bootstrap 10 final training objective | 148.258970612 | 148.258970583 |
| Bootstrap 11 final training objective | 143.441494374 | 143.441494337 |
| Bootstrap 10 initial/final withheld error, pixels | 5.976908 / 9.001402 | 6.068445 / 9.001400 |
| Bootstrap 11 initial/final withheld error, pixels | 6.050534 / 9.301208 | 6.050518 / 9.301212 |

Both maps remain past-consistent and both endpoint predictions still reject the
unchanged three-pixel gate. This does not support adopting the optimizer alone;
it does not prove all other initializations/methods would fail. No numerical speed
or accepted new-control prediction is claimed from this comparison.

## Next geometric intervention

Acquire informative publicly stable keyframes before model freezing. Retain all
actual receipts and failures; explicitly map keyframe slots to global image
ordinals, not raw one-tick intervals. Keep identities and partitions fixed within
a proposed version, fit all declared past bootstrap candidates, then test a new
control at its first public stable endpoint. Private poses may score afterwards.
Repetitive dirt/leaf surfaces suggest correspondence/conditioning hypotheses,
not a proven causal diagnosis. Do not tune acceptance gates to old private scores.
All full-project requirements in `docs/quality-roadmap.md` remain open.

## Runtime diagnostics

NumPy [documents BLAS thread controls](https://numpy.org/doc/stable/reference/global_state.html).
Twenty solves of the same seeded 96x96 positive-definite system take 0.922850
seconds with default WSL threads versus 0.001427 with one OpenBLAS thread. This is
one small local timing sample, not a general speedup claim.

`runtime_io_probe.py` creates 100 new small temporary samples per filesystem.
`runs/shared-endpoint-io-profile-v1/report.json` records 1.151941 seconds on the
mounted drive versus 0.021776 on native `/tmp`. Both sample directories remain;
no prior data is overwritten/deleted.

All three initial mounted-source WSL profiles terminate nonzero without JUnit
and are not passes. Their observed statuses remain in
`runs/shared-endpoint-runtime-profiles-status-v1.json`; no test counts are invented.

`freeze_test_tree.py` copies the complete tracked/unignored test tree plus required
checkpoint fixtures to a new native directory, verifies all 416 file hashes and
all 111 V28 ledger entries, and refuses changed sources/symlinks/credential files.
`runs/v28-frozen-test-tree-v1/report.json` records this exact 4,592,836-byte snapshot.
The isolated package import is verified to resolve inside the frozen tree.
`persist_frozen_tree.py` retains `source-test-fixtures.tar.gz` beside that manifest
and verifies every archived member; archive SHA-256 is
`F632DEE989F6BCD180CB9D71E36ADE8EC8D91DFF9E699A480D644C43EBCFA4EC`.
This is source provenance, not successful testing.

The exact-copy, single-thread, native-source/native-temporary full suite completes
with a separate 3600-second cap and configuration at
`runs/shared-endpoint-full-frozen-tree-wsl-v1.config.json`. It keeps the entire
unchanged suite, budgets and gates. The terminal process exits zero; its JUnit
records 1199 passes/two skips, zero failures/errors and 752.149 seconds. All 416
frozen file hashes are rechecked unchanged. This verifies V28, not subsequent
keyframe acquisition revisions, and remains independent of the failed profiles.

## Past-only fixed-pose point diagnostic

`point_fit_probe.py` changes only the withheld points' estimator: DLT initialization
followed by thirty bounded point-only robust LM iterations on the original twelve
past views. Training poses/points, intrinsics, partitions, budgets and gates remain
exactly unchanged. Every consumed-frame check is recomputed. No extension/future
pixel enters point fitting, and all original candidates including failures remain.
This is another known-data diagnostic, not new-control validation or promotion.

Seven synthetic tests pass on Windows, including exact recovery, a genuinely unseen
synthetic camera, noisy-ray determinism/cost decrease, malformed input and zero-
baseline refusal. The real diagnostic's two repetitions at
`runs/point-fit-same-evidence-seed202-v1` and `-v2` have identical report SHA-256
`4DBE7FCE794E5A248622197BDF878E43DBB40581453F06E24BC6B679D2DB95C0`.
All five preservation checks pass. Past cost decreases from 46.122245 to 44.828475
and 46.962647 to 45.641959 for bootstraps 10/11, but initial withheld endpoint
errors worsen to 6.165828/6.237189 pixels and final errors to 9.433341/9.530474.
Both still reject the unchanged three-pixel gate. The point estimator alone is
not adopted; improved historical loss is not evidence of better future prediction.

## Sampled-keyframe tracking diagnostic

`tracking_gap_probe.py` compares direct endpoint matching with complete public raw
intervals, retaining identical anchor detector/gates and recording surviving IDs,
image ordinals, cells and endpoint disagreement. Pair-local identities are not
stitched across intervals. Over-128-frame raw intervals remain explicitly
unavailable rather than subsampled. No map is fitted or private pose opened.
Four synthetic tests cover unchanged inputs, shared detector, differing observation
counts, failed middle-frame refusal and exact bounds.

Reports at `runs/tracking-gap-same-evidence-seed202-v1` and `-seed201-v1` preserve
all known development comparisons. Seed-202 initial survival increases from
0.333333 to 0.454365, still below the live 0.5 gate. Its pair-local 78→90 interval
increases from 0.194245 to 0.501199, but retaining the original frame-56 anchor over
56→90 gives only 0.315476. Local improvement cannot rescue the global namespace.
Seed-201 first interval increases from 0.5 to 0.595238; this is not a supported
multi-view map or new-control prediction. Every source/public input hash remains
unchanged. Full-project scope remains in the quality roadmap.

## Persistent exact verification inputs

`freeze_test_tree.py --ledger <config.json>` now accepts an explicit source ledger,
without changing earlier archived V28 bytes. V29 freezes 423 files/4,665,376 bytes;
archive SHA-256 is `533D8C163635CEBF7AB74FD5A1A6AF780499A4991F7CE2A08D993D988FECC957`.
During the first V29 full suite both source/temp paths disappear from `/tmp`.
The process subsequently terminates nonzero, including failure to restore its
missing working directory; no test counts or green result are inferred without
JUnit. Startup tmpfiles service activity correlates with disappearances. The
prior focused/default-only failures remain separately recorded.

`restore_test_tree.py` verifies archive/manifest identity, every member path/type
and byte hash before allocating a fresh tree in a persistent task-specific parent.
It never uses tar extraction, overwrites a checkout, deletes evidence or changes
host cleanup policy. Six research tests cover exact restoration, refusal of
overwrites/escaping paths/links/credentials and changed archive identity. The
423-file byte-identical restoration report at
`runs/v29-persistent-frozen-test-tree-v1/report.json` points to the `/opt` snapshot.
Its default-only suite passes 640 cases/74 skips. The full persistent-snapshot suite
completes 1227 passes/two skips with zero failures/errors in 918.562 seconds.
All 423 frozen source hashes remain unchanged after completion. This verifies V29,
not subsequent micro-control edits.

## Micro-control loss and raw-projection diagnostics

The fresh micro-control protocol is documented in
`docs/micro-keyframe-acquisition-development.md`. Its two development captures
produce one/six selected frames, but no supported map. The gap probe now explicitly
accepts either known capture schema and uses `diagnose_frame_chain` to preserve
every seed's first failing gate; five tests verify loss accounting and unchanged
raw bounds. `runs/micro-keyframe-tracking-loss-seed202-v1` records initial direct
survival 0.331349 versus continuous 0.509921, with adjacent-cycle losses 110 versus
24. Later original-anchor survival still decays. Local feature recovery is not a
map, unique motion, live success or navigation proof.

`raw_keyframe_fit_probe.py` compares the original six-view seed-201 sparse tracks
with all 66 raw images in the exact 56→121 past interval, sampled only after
complete identity-preserving tracking. Both methods retain all four bootstrap
candidates at the same seed/fitting gates. Eight synthetic tests verify exact
identity/clock/gate preservation, refusal of invalid endpoints and failure of a
middle image even when that image is not a selected keyframe.

Both known-data reports `runs/micro-keyframe-raw-projection-seed201-v1` and `-v2`
exactly replay the sparse models and pass four source/input preservation checks.
Each freezes all executing probe/package bytes. Raw projection retains 123 IDs
and still rejects every model. V2 additionally shows that the sparse slot-5 model
has only 28/31 positive-depth withheld points throughout; small positive-depth ray
errors cannot rescue its physically invalid points. V1 remains untouched. Neither
probe opens private poses, executes controls, chooses new views or fits a future.
The tracker/fitter change is not adopted as a successful production map estimator.

## Compound-control settling-time prototype

`long_wait_profile.py` validates the complete original compound-motor geometry
policy before choosing an exact wait-time profile: initial windows remain 16,
post-motor acquisition/future windows become 24, minimum WAITs remain eight and
three consecutive public stable pairs are still required. It exercises the
unchanged production recovery helper; no pixels/private positions or old native
capture tails are extrapolated. Thirteen prototype cases plus the eight helper
tests pass on Windows and Linux (`runs/long-window-prototype-{windows,wsl}-v1.xml`).
Synthetic stable booleans verify finite/shared budgets and first-eligible stopping,
not actual image convergence, geometry, map acquisition or goal arrival.

An exact V34 archive was separately restored into an ignored development source
copy, then intentionally changed for a distinct capture/audit registration.
`runs/long-window-development-tree-v1/report.json` proves the original restoration,
not byte identity after subsequent development. The candidate has to pass its own
tests, source-bound native captures and complete independent public replay. Main
V34 regressions cannot be credited to it.

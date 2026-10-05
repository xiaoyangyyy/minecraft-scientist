# Stable-keyframe geometry research

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
Its default-only suite passes 640 cases/74 skips. Full suite remains pending.

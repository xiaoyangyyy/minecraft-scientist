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

The exact-copy, single-thread, native-source/native-temporary full suite is running
with a separate 3600-second cap and configuration at
`runs/shared-endpoint-full-frozen-tree-wsl-v1.config.json`. It keeps the entire
unchanged suite, budgets and gates. Wait for its actual terminal result and JUnit
before asserting Linux completion. It remains independent of the failed profiles.

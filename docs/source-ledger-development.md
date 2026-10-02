# Decorated runtime-source identity correction

A final Windows-versus-WSL source check exposed a real registration bug in
`scripts/compound_motion_common.py`: `compound_bindings` is decorated with
`contextlib.contextmanager`. Reading its outer function's `__code__.co_filename`
returns **stdlib `contextlib.py`**, not the executing project module. Earlier
inference/plan/diagnostic manifests therefore labeled stdlib bytes as
`compound_motion.py`; their hashes also differ between installed Python runtimes.
Passing a hash comparison against that same mistaken locator did not verify the
intended runtime-source identity.

The raw acquisition ledger uses `Path(__file__)` inside the actual hook and already
freezes the real project module. This separate boundary remains meaningful, as do
complete pixel replays. The mislabeled inference entry does not itself establish
that the old geometry was false. However the intended source-registration and
portability contract was wrong and needs correction, not an exclusion from checks.

The locator now uses `inspect.unwrap` before inspecting the code filename. It
resolves all decorator layers to the actual project function on both platforms.
Tests assert exact source-module equality, retain support for another wrapper,
and reject a forged manifest/snapshot that consistently substitutes stdlib bytes
before any private evaluation. No source gate is loosened.

All three new tests and the complete regressions pass: Windows 635/two skips,
WSL 637/two, and clean default-only 361/54 optional or platform skips. The latter
skips the optional visual-pipeline module, not the default scientific core.

## Historical versus current evidence

All previous images, predictions, source snapshots and audits are retained without
rewriting manifests. In particular the direction-search development result at
`runs/public-pose-direction-seed201-clear-v1` freezes source revision `2e8b5f0`
before this correction. Its 54-check audit and negative accuracy result are
historical within-runtime computation evidence, not a current-source release or
proof that the mislabeled entry fulfilled its intended contract. Historical replay
requires the recorded WSL runtime and frozen sources. Current-source validation
must reject the old manifests.

The original seed-201 compound raw acquisition can still be verified under current
capture sources. It has been recomputed at
`runs/rgb-compound-prefix-seed201-source-v2`; all 46 pair results/164 prefixes are
exactly equal to their previous public results. Its new independent audit passes
35 checks, with one correct selected leg interval and no combined coverage. This
is provenance repair, not new image acquisition or an accuracy improvement.

`runs/active-parallax-plan-seed201-source-v4` then replays that corrected prior and
chooses the unchanged V3 control pairs `(0,1), (0,2), (0,3), (1,0)`. The suffix
identifies a new source-bound run, **not** a new policy or relaxed geometry.
Fresh active acquisition is required because the old active profile binds its old
plan/source bytes; those declarations are never updated to masquerade as current.

That fresh capture has completed at
`runs/controlled-rgb-active-parallax-seed201-source-v4`: 225 RGB frames, all raw
acquisition gates and all six active-profile checks pass. The corrected prior's
runtime-source hashes now match from Windows as well as its recorded WSL runtime.
This does not imply cross-version OpenCV prediction equality or byte-identical
real rendering. Fresh prefix inference and independent visual scoring are separate
steps; capture integrity alone supplies no direction accuracy or navigation result.

Those steps subsequently finish at
`runs/rgb-active-parallax-prefix-seed201-source-v4`: all 46 pairs/164 prefixes are
computed, and the independent `audit.json` passes 42 checks with correct project
source registration. Four/eight effective turns and four/eight correct combined
local poses are recovered. Twelve selected pair records contain eleven correct/
one wrong; nine unique intervals contain eight correct/one wrong. The 152→154
direction error remains exactly 32.412640 degrees. Twelve actual stationary
controls reject translation, while two nominal settle controls actually move and
remain unsupported. This repeat does not improve geometry or establish navigation.

## Scope

The package-only V10 scientific-core source ledger remains unchanged by this
script locator fix. Its thirty-seed/two-repeat/eighteen-artifact results do not
prove visual navigation, integrated natural-world science or recovery. The
original full completion requirements remain in scope. After reestablishing
current visual acquisition, test frozen competing explanations against truly
subsequent non-collinear views and retain the surviving direction counterexample.

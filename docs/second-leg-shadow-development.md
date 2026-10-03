# Frozen second-leg different-control validation

This is a single-seed development experiment, not held-out navigation acceptance
or completion of the full project. It adds a missing causal test: a map fitted on
the completed second leg must predict later images acquired with another opaque
movement handle. Public process integrity and physical accuracy remain separate.

## Acquisition and inference contract

The optional `--shadow-second-leg` capture path preserves the original two-leg
controller and all eight trials. After a completed second leg, it recomputes every
past prefix, latches its first supported map, and freezes both that selection and
the training-only bounded competing-direction search before executing any new
control. The fitting endpoint is not the freeze time: fitting is performed at the
completed leg boundary, not asserted to have happened live at the earlier prefix.

`SecondLegShadowPolicy` registers four one-tick controls and chooses the lowest
public handle distinct from both original handles. Distinct handles do not prove
physical non-collinearity. Trial fitting seeds are 1002, 1007, ..., 1037. The
existing geometry, temporal tracking and pixel-consistency thresholds are not
relaxed. There is no evaluator pose input, future refitting or replacement map.

Original fitting/withheld landmark identities are protected through every new
prefix. Lost IDs cannot return; surviving IDs cannot revise their past pixels.
Control or tracking failures remain failures without retries, and insufficient
map support aborts before the next action. Generic surviving tracks cannot replace
the fitted/withheld partitions. A trial with no candidate may collect bounded
diagnostic images but has no available map-validation plan. Originally aborted
trials retain explicit skipped outcomes, not fabricated third-leg controls.

Public inference replays all receipts, prefixes, first-supported fits, frozen
searches and support decisions. It uses the captured fitting seed and selection,
not an inference-time seed or a newly fitted terminal map. Candidate maps and
alternative maps are compared on fixed midpoint/end new-control views. All eight
third-shadow records remain in the denominator, bringing the record count from
46 to 54. Third-shadow records describe acquisition; they are not eight additional
independent map successes.

## Current development artifacts

- Acquisition: `runs/controlled-rgb-second-shadow-seed201-clear-v1`.
- Public inference: `runs/rgb-second-shadow-maps-seed201-v1`.
- Windows/WSL scientific core: `runs/core-release-seeds0-29-{windows,wsl}-v17`.

Acquisition completed with 203 RGB frames, all seven integrity checks and all eight
original trials retained. Six original trials finished and two aborted. The six
eligible third shadows completed four one-tick actions each (24 extra actions);
the two original aborts remain skipped. Only one second-leg candidate exists:
trial 1, anchor 80, fitting endpoint 83, actual freeze 84, withheld new-control
views 86 and 88. This is candidate coverage **1/8**, not navigation success.

Public inference completed all 54 records, verified unchanged input/source bytes,
and passed all fifteen public acquisition replay checks. The lone candidate's
new-control pixel test is `reference_inconsistent`: midpoint/end withheld 95th
percentile residuals are respectively 4.302939 and 11.974825 pixels. The surviving
withheld set has sixteen landmarks spanning four cells at both views, so this
cannot be described merely as missing generic support.

Independent audit completed successfully with **29/29 checks**. It replays the
entire public acquisition and inference before first reading private positions,
then verifies both public and evaluator bytes remained unchanged. The lone
second-leg candidate's direction error is **38.119170 degrees**, beyond the
unchanged 20-degree gate. Its new-control displacement is 0.173684 blocks and
direction difference 132.967212 degrees, meeting the registered 45–135 degree
physical-diversity gate. Those are evaluator-only results, not policy inputs.
This candidate is therefore both physically wrong and contradicted by new pixels,
despite adequate surviving original validation support. All six prefit-plausible
competitors are also inconsistent; ten other bounded alternatives were not
prefit-plausible and have no future test. No candidate survives, and rejection
does not prove the alternative family contains every possible correct map.

| Group | Trials | Selected records | Correct / wrong selected | Retained maps |
|---|---:|---:|---:|---:|
| Compound moving | 8 | 3 | 3 / 0 | 0 |
| Compound settled, including new controls | 8 | 4 | 4 / 0 | 0 |
| First leg | 8 | 1 | 1 / 0 | 0 |
| Second leg | 8 | 1 | 0 / 1 | 0 |

These nine selected records overlap: they represent **five unique intervals,
four correct and one wrong**. Every supported prefix is retained in the audit:
besides the wrong second-leg interval 80→83, the compound terminal interval 77→84
has a 21.754527-degree direction error. Thus first-supported selection does not
erase a later supported counterexample. The eight settle controls include seven
physically stationary controls; all seven abstain on translation. The two each
WAIT/rotation/identity controls also produce no supported false translation.
Abstention is not evidence of useful navigation coverage.

The extra controls change later positions and settled comparisons. In particular,
`compound_settled` now includes third-control displacement before WAIT. These
paths are not matched to the historical two-leg experiment. Neither changes in
local-fit counts nor this single development seed establish an accuracy gain.

## Verification and source boundary

Forty-seven focused tests pass: policy/handle/receipt validation; no future
refitting; permanent identity intersection; protected support loss; frozen-prefix
pixel preservation; capture ordering; all-trial public replay; forged policy,
budget, frame and freeze rejection; captured-seed/selection reuse; and synthetic
geometric fitting/search. Mocked protocol tests are not physical accuracy tests.

Both V17 core releases pass thirty seeds/two repeats/eighteen exact artifacts and
all six checks. All 100 package/verifier/configuration source hashes were rechecked
with zero mismatches. The seed-zero MVP benchmark remains byte-identical with
SHA-256 `B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
The new raw capture freezes 125 project/SDK sources; public inference freezes 128
sources. Prior V16 core and prior capture/map/diagnostic manifests remain historical
and are not rewritten to claim current-source execution.

Full current regressions pass **811 tests/two skips on Windows**, **813/two on
WSL**, and **443/64 optional/platform skips** in an isolated default-only offline
installation. JUnit artifacts are `runs/second-leg-shadow-regression-{windows,wsl}-v1.xml`,
`runs/second-leg-shadow-regression-default-offline-v1.xml` and the 47-case
`runs/second-leg-shadow-targeted-v2.xml`; all have zero failures and errors.

## Remaining work

Do not integrate this candidate as a trusted navigation map. Retain its independent
public replay, physical error and failed future-pixel evidence. Next
investigate multi-view estimation and the provenance of these residuals, using
separate fitting, model-selection and future-validation observations rather than
refitting against a failed holdout and calling that same holdout validation.
Fresh disjoint trajectories are required before promotion. The full roadmap still
requires multi-seed/terrain/condition evaluation, natural scientific/map integration,
long operation/recovery, live release/baseline revalidation and CI consolidation.

## Reproduction

In the dedicated WSL runtime, from `/mnt/d/minecraft-scientist`, use fresh output
directories; no output overwrite is permitted:

```bash
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/probe_visibility_maps.py runs/controlled-rgb-second-shadow-seed201-clear-v1 --output runs/new-second-shadow-maps
timeout 900 /opt/minecraft-scientist-visual/bin/python scripts/audit_visibility_maps.py runs/controlled-rgb-second-shadow-seed201-clear-v1 runs/new-second-shadow-maps --output runs/new-second-shadow-maps/audit.json
```

Saved-image inference/auditing requires neither Minecraft execution nor an API
key. New real acquisition requires the dedicated MineStudio runtime. Preserve
frozen snapshots when subsequent source changes invalidate current-source replay.

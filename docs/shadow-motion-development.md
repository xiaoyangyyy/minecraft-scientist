# Subsequent-frame pose veto: tested, not promoted (2026-10-02)

This optional development method follows the retained seed-200 26.176-degree
direction error. It tests whether two subsequent RGB images can independently
veto a frozen local pose. Results show they cannot reliably resolve this case.
The original navigation acceptance standard and full project scope are unchanged.

## Executable contract

`exploration/shadow_motion.py` defines immutable configuration/evidence and a
public-data-only validator. The existing first-supported prefix remains latched.
The validator can retain or veto it, never choose another prefix, refit its pose
using future observations, transport it to the terminal frame or fabricate scale.

Exactly two consecutive images after the selected endpoint are required. The
extended chain retains the original anchor and stays inside the 32-image budget.
Missing future images or insufficient budget are explicit abstentions. Track IDs
must survive from the original prefix; new/recovered IDs are ignored. Surviving
past pixel observations must agree with the frozen prefix within 1e-6 pixels.

The 3D map is triangulated only from original endpoint rays and the original
camera/pose. The original disjoint PnP training/validation IDs are preserved, not
repartitioned using future data. Each future PnP fits/refines only surviving
training points and predicts all surviving validation points. Original minimum
track budgets, positive depth, coverage and reprojection gates are unchanged.
The original two-view map used endpoint rays from both point partitions: this is
a temporal holdout and disjoint **future-PnP** check, not fully independent spatial
holdout of the original essential-model fit or proof of true direction.

Accepted evidence must contain both future checks and the exact original pose.
Rejected evidence cannot contain an executable pose. Schema validation rejects
reselection, invented/reassigned identities, incomplete checks and false success.
Pinhole intrinsics and signed frame-seed budgets are validated before geometry.
No simulator position, yaw, control direction label or evaluator accuracy is input.

`shadow_motion_common.py` reconstructs compound acquisition/configuration, freezes
sources/inputs and records every prefix, extended chain and veto. Thin producer
and auditor entry points reuse the unchanged prefix implementation. Nested or
noncanonical runtime bindings reject and restore aliases after exceptions.
The independent auditor reconstructs every public prefix and shadow result before
its first telemetry read, then scores the original and vetoed local intervals
separately. Candidate identity, evaluator identity and public/source bytes remain
checked through the final scoring stage. Invalid/partial artifacts are not scored.

## Complete development replay

Both existing compound acquisitions (development seeds 200/201, noon/clear) are
reused. This is **not new acquisition or held-out validation**. All 46 pairs and
164 prefixes per session are preserved, with no removal of failed/stationary
trials. Comparison finds every original prefix selection/track/pose record
identical to the prior compound producer. Both final audits pass **36** integrity/
replay checks, which establish computation/provenance, not navigation success.

| Seed 200 group | Correct original selected local movements | Correct after veto | Wrong supported after veto |
|---|---:|---:|---:|
| Combined moving | 4/8 | 3/8 | 0 |
| Combined settled | 4/8 | 2/8 | 0 |
| First leg | 3/8 | 1/8 | 0 |
| Second leg | 1/8, plus one wrong | 1/8 | 1 |

Seed 200 starts with thirteen selected pair predictions: twelve correct and one
wrong. The veto accepts eight: seven correct and the **same wrong prediction**.
Four correct predictions lose required surviving partitions; one fails future
geometry. Deduplicating image intervals gives six accepted intervals, five correct
and one wrong, versus the original eight intervals, seven correct and one wrong.
These groups overlap and must not be summed as independent trials.

The seed-201 single correct `leg_two-07` local pose loses its required original
partition budget and is rejected. There is then zero useful supported local motion
in that session. No previously unsupported candidate or negative control is rescued.
The original physically moving late-WAIT failure remains in the complete report.

### Retained counterexample

`leg_two-02`, seed 200, frames **92→96**, still passes the new future checks at
frames 97 and 98. The independently scored local direction error remains
**26.176°**, over the unchanged 20-degree tolerance; rotation error remains 0.457°.
Future validation retains 74 original training identities and 21 validation
identities. Both original and new poses are identical, as required by the veto-only
contract. There is no accuracy repair, and the candidate is **not promoted**.

The public schedule places frames 97/98 inside the following WAIT. This method
does not require or measure a new non-collinear baseline. A freely fitted future
PnP may therefore reproduce new pixels using a biased frozen map. That is a
geometric explanation consistent with the retained failure, **not a proven unique
cause**; stronger observability/competing-map evidence is needed. Simply adding two
time-separated frames did not distinguish the wrong direction and reduced coverage.

## Artifacts and reproduction

- `runs/rgb-shadow-prefix-seed200-clear-v1`: complete prefix/shadow predictions,
  configuration, source snapshots and `audit.json`.
- `runs/rgb-shadow-prefix-seed201-clear-v1`: the same complete artifact contract.
- Raw images/actions remain in `runs/controlled-rgb-compound-seed{200,201}-clear-v1`.

From the project root in the installed WSL/OpenCV environment:

```bash
/opt/minecraft-scientist-visual/bin/python scripts/probe_shadow_rgb_motion.py runs/controlled-rgb-compound-seed200-clear-v1 --output runs/new-shadow-prefixes
/opt/minecraft-scientist-visual/bin/python scripts/audit_shadow_rgb_motion.py runs/controlled-rgb-compound-seed200-clear-v1 runs/new-shadow-prefixes --output runs/new-shadow-prefixes/audit.json
```

Outputs refuse overwrite. No API key/remote model is used. Accepted map/navigation
memory is not a target of these diagnostic entry points.

## Regression and next action

Eleven optional geometry tests verify positive synthetic motion, withheld future
corruption, frozen past pixels/identities, point-budget loss, missing future images,
determinism, intrinsics, seed bounds and evidence schema. Seven pipeline tests verify
complete public-before-private replay, tampering, identity checks and post-replay
source/input stability. Three default-only schema tests verify lazy optional
geometry, frozen configuration and no-candidate abstention. Synthetic tests are
boundary/correctness tests, not real-world accuracy evidence.

The current v7 scientific-core release freezes the optional shadow source (92
package/configuration source entries) and passes 30 seeds/two repeats/eighteen exact
artifacts on Windows and WSL. The MVP seed-zero report remains byte identical and
source/wheel builds and targeted Ruff pass. Final regressions pass **538/two skips
on Windows**, **540/two on WSL**, and **332/44 optional/platform skips** in a clean
default-only installation; see `docs/core-release-verification.md`.

The next development must measure whether new views supply discriminating parallax
or a genuinely changed viewing direction, and retain competing map/pose explanations
when reprojection cannot separate them. Test that criterion against this wrong case
**and** the correct cases lost here, then acquire fresh disjoint active-view sessions
before any planner promotion. Do not count more abstention as useful navigation,
change tolerances after evaluation or switch to a private-selected alternative.
Held-out worlds/terrain/weather, actual goal arrival, natural scientific integration,
long-operation recovery and the complete release comparisons remain required.

# Phase-relative visibility acquisition

This development correction changes experiment acquisition, not map acceptance
or navigation promotion. The full requirements in `quality-roadmap.md` remain.

## Diagnosed public controller defect

The independently replayed original adaptive capture has eight second legs, all
ending at their two-tick minimum. At each first-leg switch, original-anchor track
retention is already below 0.70. Applying that same loss threshold again to the
second leg immediately retriggers it, even when the remaining points are stable.
This diagnosis uses public controller state, not private direction-error labels.

The corrected policy declares `retention_reference=phase_start_v2`. Its first-leg
denominator remains the detector population at the original anchor, including
points lost in the first executed action. At the switch, only the denominator
changes to the number of still-retained original IDs. Second-leg decisions record
both original-anchor and phase-relative population/fraction. A point lost at any
earlier prefix never returns, including when the raw tracker returns that ID.

The absolute minimum of 32 original tracks/two spatial cells, projected-edge risk,
minimum two ticks per leg, six/four maximum ticks, tracking bounds and downstream
geometry/physical error thresholds are unchanged. New loss can still legitimately
end the second leg after two ticks. A controller completion is not valid motion.

Seven additional tests cover stable second-phase completion at the full budget,
new second-phase loss at the minimum, absolute-support abort before the relative
threshold, invalid phase denominators/fractions, first-phase anchor preservation
and explicit policy versioning. Existing immutable-prefix, failed-control,
no-replenishment and complete public-first map replay tests remain required.

## Evidence boundaries

Raw `controlled-rgb-visibility-seed201-clear-v1` and inference
`rgb-visibility-maps-seed201-v2` retain their unchanged snapshots and negative
results. They precede this package-source/policy boundary, so their historical
current-source checks must not be relabeled as validation of the new controller.
Likewise V13 scientific-core ledgers are historical after this source change.

The new capture uses the same development world/conditions/public parent plan at
`active-parallax-plan-seed201-source-v4`, with a fresh output directory and new
source snapshots. This is not matched-path estimation, disjoint-seed validation
or independent terrain generalization. Changed trajectories can change costs,
effective turns, map support and ambiguity; report all eight trials and controls.

## Runtime transport incident

The first three WSL launch requests did not execute their Python commands: process
inspection in the system and distribution namespaces found only waiting init
relays, no capture/test/core-verifier processes, and no new output directories.
Those specific Windows launcher processes were canceled and observed terminal
before another attempt. The distribution was not restarted or reset.

The working recovery enters the same running distribution's mount/PID/network/IPC
namespaces through `wsl --system -u root` and `nsenter`. A read-only Python probe
verified the installed 3.11.16 runtime and repository directory before submission.
`env --chdir` changes directory after root/namespace entry; the earlier
`nsenter --wd` diagnostic did not provide a usable working directory. Application
jobs retain their 600-second timeout. This workaround is specific to the observed
local WSL transport issue, not a new simulator or scientific algorithm.

## Complete fresh result (2026-10-02)

`runs/controlled-rgb-visibility-seed201-clear-v2` completes all eight probes,
188 RGB frames/93 public actions and all seven raw integrity checks. The probes
execute 59 one-tick movement actions; the declared 209-frame ceiling is unchanged.
First legs take two-to-five
ticks; all second legs take four ticks. Completion retains 99-to-245 original IDs,
with phase retention 0.66-to-0.942308 and original-anchor retention
0.386740-to-0.610973. The policy permits earlier completion; it does not force
every second leg to reach its maximum.

`runs/rgb-visibility-maps-seed201-v3` completes all 46 records/136 prefixes and
its independent `audit.json` passes all 26 checks. The auditor recomputes public
capture decisions, frozen models and future checks before reading private poses.
All 126 inference sources and all 123 raw sources/snapshots remain bound.

| Group | Trials | Selected correct / wrong | Correct terminal moving |
|---|---:|---:|---:|
| Combined moving | 8 | 3 / 0 | 2 |
| Combined settled | 8 | 5 / 0 | 2 |
| First leg | 8 | 2 / 0 | 1 |
| Second leg | 8 | 0 / 2 | 0 |

Twelve selected pair records contain ten correct/two wrong results; seven unique
intervals contain five correct/two wrong. Overlapping groups are not independent
successes. Five/eight control pairs are physically non-collinear. All fourteen
actual stationary controls reject translation (eight settle/two rotation/two WAIT/
two identity), with no false translation. No trial or failed prediction is dropped.

Second-leg intervals 136-to-140 and 167-to-171 remain wrong by **22.762669** and
**20.831977 degrees** under the unchanged 20-degree direction gate. Their rotation
errors are 0.452286/0.324960 degrees, and their movement lengths are both 0.657440
blocks. These maps have no later different-control validation in this protocol.
Different paths and additional observations prevent a matched improvement claim
against the historical 36.654909-degree error. Navigation remains unaccepted.

Of the two supported first-leg maps, 89-to-91 loses every original fitting/
validation landmark before the end of the switched action, while 148-to-152 has
83 fitting/27 validation landmarks in its future checks. The latter passes both
withheld-pixel checks and the optional diagnostic retains one correct interval.
Its comparison is `no_prefit_witness`, not uniqueness: the fitted alternative
family is incomplete, and its actual shadow motion is antiparallel (180 degrees),
not independently non-collinear. More observations alone have not solved pose.

## Next public diagnosis: protect actual fitted landmarks

For the correct 89-to-91 candidate, public fitting uses 36 training/12 validation
IDs at tick two. By the first-leg switch at tick three these populations are only
7/1, despite 222 generic points still surviving. They fall to 1/0 at tick four and
0/0 at tick seven. Generic retention therefore does not protect the actual map.

Using the same existing one-step extrapolation, central ROI and 12-reference-pixel
margin at tick two, **48/48 map landmarks** are projected near/outside the safe
region; the generic population's projected-edge fraction is only **0.117845**.
The map exists before the next action and supplies a causal public warning that
the generic controller misses. This derives from frozen public tracks/IDs, not
private direction labels or retrospectively selected replacement landmarks.

Next integrate online freezing/protection of fitting and withheld map identities
once a public candidate exists, and obtain genuinely different-control validation
for second-leg maps. Preserve all old failures, original ID partitions, absolute
support and error gates. Acquiring distinct handles still does not prove physical
non-collinearity; joint map refinement/active excitation remain open if new views
do not resolve competing explanations. Do not turn unavailable maps or no-witness
searches into navigation acceptance.

## Engineering checkpoint

Seven new tests and complete regressions pass **725/two skips on Windows**,
**727/two on WSL**, and **408/60 in the isolated default-only offline environment**.
Targeted Ruff/format checks and package builds pass. Both V14 scientific-core
releases pass six checks over thirty seeds/two repetitions/eighteen artifacts,
and all 98 source hashes still match. The fresh seed-zero MVP benchmark is
byte-identical to the historical artifact, including its scope/cost limitations.

This correction and fresh evidence are progress, not overall project completion.
Five additional held-out worlds/multiple terrains/three conditions, integrated
natural scientific experiments/maps, at least 100 exploration/recovery attempts,
live release/baseline comparisons and consolidated CI/release verification remain
required. No full-project gate is removed or reduced by this development result.

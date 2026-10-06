# Fixed four-MOVE child-map stimulus

This is an isolated, explicitly registered development candidate, not promoted
main-source navigation. It inherits the complete frozen new-map candidate in
`docs/map-epoch-reacquisition-development.md`; it does not overwrite its sources,
captures, audits or negative results. The restored candidate tree is
`runs/development-trees/verified-source-b9zf56o3`. The complete twenty-file delta
against main V35 is `research/settled_keyframes/excited_map_candidate.patch`.

## Problem and intervention

The parent seed-201 native experiment genuinely creates a new local map and passes
its public future checks, but both independently evaluated physical intervals are
`false_translation`: player-body/rendered displacement is only about 0.103 blocks
under the unchanged 0.15-block threshold. Seed 202 still cannot create a startup
map. Neither result is navigation or goal arrival.

The new intervention changes only the **child map's future stimulus** from two to
four actual one-tick MOVE receipts. It does not change startup acquisition, the
parent future stimulus, acquisition MOVE length, opaque handle selection, image
detectors, optical settling, fitting, withheld validation, model consensus or the
final pixel/geometry gates. The first public optically stable WAIT endpoint is
still used, not an endpoint selected by private displacement or prediction error.
There is no guarantee that four MOVEs produce sufficient motion or correct
geometry: collision, map challenge and insufficient support remain valid failures.

`StableExcitedReacquiringWalkPolicy` has exact integer `validation_move_ticks=4`
and `future_stimulus_registration="fixed-four-move-child-future-v1"`.
`ExcitedMapEpochPolicy` uses registration
`bounded-four-move-map-reacquisition-v1`. Both reject modified gates/budgets;
the old and new profiles cannot be disguised as one another. The startup policy
remains the original `StableMonitoredWalkKeyframePolicy` with two future MOVEs.

Every actual future MOVE and WAIT is checked against the frozen child map before
dispatch continues. If the third MOVE challenges it, the fourth MOVE is not
executed. Recovery consumes the remaining original passive budget rather than
resetting it. The same shared limits remain: 288 total actions including the
calibration/orientation prefix, 22 acquisition trials, two local epochs, two
additional acquisition windows, two passive windows and at most one child map.
The parent history is immutable, and child tracks stay in a new unconnected gauge.
No metric scale, old-map reinstatement, global gauge alignment or navigation is
claimed.

## Native boundary

`--excited-keyframes` requires `--reacquire-keyframes` and all eleven explicit
parent flags. Its distinct schema is
`excited-reacquiring-monitored-walking-keyframe-camera-projection-v1`.
Capture registers the unchanged startup policy separately from the exact new
epoch policy. Old schemas reject the new profile, and the new schema rejects a
two-MOVE child or relaxed gates.

The independent auditor replays the full actual public calibration, orientation,
startup, recovery, new-map acquisition and every future action before private
pose files are opened. Rendered and player-body scores remain separate, with
the original 0.15-block translation threshold and no endpoint rescue. Successful
experiment provenance alone is not successful map acquisition or physical motion.

## Current evidence

`runs/excited-epoch-source-registration-v1.json` freezes twenty source paths before
Linux tests, broader regressions or native launches. The inherited eighteen-file
parent identity was verified before the new edits; its old evidence is not
restamped as new-source coverage.

The new focused Windows and Linux batches each terminate zero with thirty passes,
zero failures/errors/skips: seventeen controller/schema cases and thirteen
public-first native-auditor/CLI cases. Controller fixtures use real RGB/receipts
with explicitly documented flow/model/producer doubles; audit fixtures are mock
captures and their positive cases do not create a native child map. They test
chronology, registration and rejection boundaries, not native geometry or useful
body motion. The Windows/XML and Linux/XML hashes are preserved in the
source-bound receipts; native effects remain unverified at this checkpoint.

A broader current-source Windows batch has launched for 273 cases, including the
old controller/shared-WAIT cases, old new-map audit guards and these thirty new
cases. Launch is not terminal verification. Compatibility, full auditor and full
package regressions remain separate; none can borrow the parent's counts.

## Next gates and unchanged full objective

Close the current-source regression batches and freeze the complete native source,
engine, policies and absent output paths before fresh familiar-development seeds
201 and 202. Preserve every failed capture and independently audit both. Do not
call these two clear/noon worlds disjoint held-outs, paired superiority or arrival.
Feature-support-sensitive acquisition for the difficult world remains separate.

Even useful local body motion will leave reliable goal arrival, at least five
disjoint held-outs across terrains/three conditions, sufficient samples and
uncertainty, natural science/map integration, one hundred exploration/experiment
attempts with restart/injected-failure recovery, matched baselines/ablations and
consolidated release/hosted CI open. `docs/quality-roadmap.md` retains the full
objective; this candidate does not replace it with a smaller test suite.

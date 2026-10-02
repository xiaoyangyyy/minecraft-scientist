# Frozen-map validation after a public action switch

This is a retained negative development experiment, not a navigation gate.
`action_switch_motion.py` fits the original map and sixteen fixed direction
alternatives using only the selected chronological prefix. The next different
public movement handle determines two subsequent midpoint/end observations.
Neither physical direction nor evaluator coordinates select those observations.
A different handle does not guarantee a non-collinear physical trajectory.

Original fitting retains the 32-frame bound; subsequent tracking explicitly uses
a 64-frame bound. This is additional observation/computation cost, not a matched
budget improvement. Holdout is per frozen map in an existing session, not a fresh
held-out acquisition, seed or terrain.

## Dataflow and failure contract

All original prefixes are replayed before fitting alternatives. Alternative
rotation refinement consumes original training endpoints only. Frozen evidence
is checked before/after tracking; future pixels cannot refit or select a pose.
Original identities and disjoint training/validation partitions are preserved.
Every map uses the same surviving IDs. Lost points, altered prefix pixels or
insufficient support are `unavailable`, never a geometric contradiction.

Future PnP fits only training points and checks original withheld points using
unchanged pixel/depth/coverage gates. All sixteen alternative records are retained;
failed prefit alternatives cannot consume future pixels. A passing alternative
cannot rescue a failed reference. No-witness and a finite family's rejection are
not uniqueness. The public evidence exposes no replacement pose or navigation
acceptance.

The independent auditor replays the complete public computation before opening
private telemetry. Source/input/snapshot/evaluator bytes remain bound through
scoring. It separately measures actual non-collinearity without changing policy.

## Real result (2026-10-02)

Inputs are the corrected-source seed-201 noon/clear development session:
`runs/controlled-rgb-active-parallax-seed201-source-v4` and
`runs/rgb-active-parallax-prefix-seed201-source-v4`. The new artifact is
`runs/public-action-switch-seed201-clear-v1`; all **54 independent checks pass**.
All 46 pair records remain, including 34 with no original candidate.

| Comparison on the 12 selected records | Count |
|---|---:|
| Unavailable | 5 |
| Reference inconsistent | 3 |
| Ambiguity survives | 1 |
| No prefit witness (not uniqueness) | 3 |
| All tested alternatives rejected | 0 |

The original selection has eleven correct/one wrong pair records, or eight
correct/one wrong unique intervals. The optional veto retains only **three correct
pair records (two unique intervals)**. It loses eight correct records. The known
152-to-154 error of **32.412640 degrees** loses all original points by the later
172/176 observations and is unavailable, not correctly identified as wrong.
Consequently zero retained errors is not a successful accuracy result. Several
correct original maps fail future consistency; one such alternative passes, but
is deliberately not substituted. These results reject navigation promotion.

## Verification and reproduction

Thirty-seven new tests cover schema/control planning, frozen dataflow, missing
points, identical partitions, withheld corruption, complete grids, public-first
replay, tampering, partial failure and overwrite refusal. Known-pose synthetic
fixtures verify these contracts, not real-world estimator selection.

At this checkpoint full regressions pass **672/two skips on Windows**,
**674/two on WSL**, and **378/56 in default-only installation**. The last result
also passes with `--offline --isolated` after a locked dependency-cache setup;
the first offline setup failed because a NumPy wheel was absent. No model/API
network is used by the default algorithm. Both V11 thirty-seed core releases
freeze 97 entries and reproduce eighteen artifacts per seed twice. These are
engineering/scientific-core checks, not visual generalization.

From the dedicated WSL environment, with new output paths:

```bash
/opt/minecraft-scientist-visual/bin/python scripts/probe_action_switch.py runs/controlled-rgb-active-parallax-seed201-source-v4 runs/rgb-active-parallax-prefix-seed201-source-v4 --output runs/new-action-switch
/opt/minecraft-scientist-visual/bin/python scripts/audit_action_switch.py runs/controlled-rgb-active-parallax-seed201-source-v4 runs/rgb-active-parallax-prefix-seed201-source-v4 runs/new-action-switch --output runs/new-action-switch/audit.json
```

Next implement bounded visibility-aware online acquisition before original tracks
disappear, then independently validate fresh trajectories and stable landmark
lifecycle/refinement. Do not loosen error tolerances, replace development seeds,
count lost-track abstention as error detection, or remove the full held-out,
natural science/map integration, long-recovery and release requirements.

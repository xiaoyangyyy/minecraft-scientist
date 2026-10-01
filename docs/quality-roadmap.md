# Project quality and completion plan

The user has requested autonomous planning and implementation until this project
reaches an excellent standard. This plan covers the original theory-acquisition
system, not just the latest navigation experiment. Completion remains unproven
until each required gate has current, inspectable evidence.

## Required outcomes

| Gate | Required behavior and evidence | Current assessment |
|---|---|---|
| Executable scientific discovery | Recover equation, rule, state-machine and program theories through controlled interventions; predict unseen interventions; resolve theory crises through independently validated versions | Strong structured and live-fixture evidence exists; revalidate against current release |
| Evolving symbols | Promote learned executable abstractions only after independent reuse; demonstrate compression and intervention savings against no-library and fixed-library controls | Existing paired ablations; inspect scope and reproduce release matrix |
| Honest comparison | Matched budgets, shared evidence, random/context/skill-memory/fixed-symbol/full-system baselines; report per-seed samples, uncertainty and costs | Existing evidence; verify statistical claims and separation of offline/LLM/live results |
| Visual generalization | Anonymous RGB observations, calibrated positive/negative boundaries, ambiguity rejection, held-out sessions/seeds/terrain/conditions; no evaluator labels in policy | Bounded natural scenes and condition shifts verified; broad terrain/seed generalization incomplete |
| Persistent cognitive map | Recover journal and projections after restart; bounded exemplars; executable directed routes; reject false loops; retain unknowns, failures and theory references | Persistent route reuse verified; longer exploration and recovery need expansion |
| Open-world scientific exploration | Select reachable experiments by information gain, execute them in natural worlds, expand/reuse the map, discover mechanisms and revise theories with a fixed budget | Natural navigation and live scientific fixtures remain separate; integration incomplete |
| Reliable goal navigation | Use agent-visible spatial evidence; independently verify arrival; no privileged runner-up selection; compare against appearance-only navigation on held-out trajectories | Appearance-only goal search has a verified counterexample |
| Reproducible engineering | Clean installation, automated portable core checks, explicit optional backends, bounded processes/timeouts, frozen source/data provenance, meaningful tests and documentation | Package/tests/build verified locally; portable CI and release evidence need consolidation |

## Execution order

1. Repair the visual navigation evidence. Evaluate correspondence/geometry signals
   from existing RGB scans without supplying evaluator coordinates to the policy.
   Reject low-support or inconsistent geometric estimates. Preregister a candidate
   method before fresh-world validation; retain the appearance-only failure as a
   negative result. Do not tune a direction to its privileged arrival distance.
2. Expand natural-world evaluation. Use disjoint acquisition, calibration and
   evaluation sessions; develop on at most two natural seeds and evaluate on at
   least five additional seeds, multiple terrain classes and three lighting/weather
   conditions. Report retrieval acceptance/rejection, false loop closures, route
   completion, interventions, memory growth and recovery failures. Target at least
   0.90 positive acceptance and 0.95 negative rejection, with confidence intervals
   and sufficient samples; do not silently drop failed seeds or reduce scope.
3. Integrate the scientific and spatial loops. Represent mechanism evidence,
   competing hypotheses and available experiments at map nodes. Travel to a
   reachable experiment, perform controlled interventions, update executable
   theories and learned symbols, then reuse that knowledge on another instance.
   Include unchanged-rule transfer and changed-rule revision in natural-world
   evidence, rather than relying solely on independent resettable fixtures.
4. Validate longer operation and recovery. Run bounded long trajectories of at
   least 100 frontier/experiment attempts, controller restarts and injected
   execution failures. Verify that failed controls cannot become theory evidence,
   known negatives cannot be consumed by memory adaptation, unresolved routes
   remain unresolved, and accepted knowledge survives restart. Record all failure
   categories and investigate them autonomously.
5. Consolidate the engineering. Put reusable perception, planning and persistence
   logic in the package, keep experiment scripts thin, add portable CI, lock
   optional runtime requirements, and provide one release-verification entry point.
   Preserve frozen historical evidence when production code evolves. Fix genuine
   correctness/maintenance problems rather than hiding them behind a narrow gate.
6. Run the release comparison and audit. Reproduce structured mechanisms over at
   least 30 randomized seeds, live mechanism/revision gates, paired symbol/planner
   ablations and the held-out visual/open-world suites. Audit every requirement,
   named artifact and claimed number against the current source and run evidence.
   Document remaining scientific limits explicitly. A green unit suite or one
   natural route does not establish overall completion.

## Evidence discipline

- Agent policy inputs contain RGB/opaque public observations and learned action
  programs. Coordinate telemetry and hidden rules remain evaluator-only.
- Development results are not held-out evaluation. State which failed trajectories
  informed a new method and use fresh disjoint validation before promotion.
- Preserve positive and negative runs, source hashes, raw temporal image samples,
  public actions, privileged evaluation and journal/projection verification.
- A rejected expansion must leave accepted memory unchanged. A tentative node is
  not a resolved transition until independent recognition and arrival checks pass.
- Deterministic structured runs must replay exactly. Real image-rendering variation
  is evaluated statistically; do not claim byte-identical live RGB frames.
- Keep optional external-model comparisons and their cost ledger separate from
  offline defaults. Use existing authorized local resources for implementation.

## Current checkpoint

`ce48e34` provides executable A→B→E→F→G route reuse in seed 201 with fresh-session
recognition, shadow validation and independent audit. The accepted G expansion has
22 passing audit checks. Recursive continuation selected heading 3 using panorama
appearance but failed its physical progress veto; 17 rejection-audit checks prove
that accepted memory remained byte-identical. That counterexample changes the next
engineering action: improve spatially meaningful visual evidence before scaling
claims about goal navigation. The overall project goal remains active.

The subsequent RGB-geometry development is documented in
`docs/visual-geometry-development.md`. Dense acquisition and independent replay
are implemented, but a supported-pose increase failed its disjoint error check.
The stricter bootstrap now abstains on the night/thunder development route.
This improves evidence integrity, not navigation success. A fresh higher-resolution
route failed node-B recognition; reference-scale normalization then passed a new
five-node replay with 24 independently recomputed checks and zero physical waypoint
error. The acquisition audit runs offline on Windows without the simulator stack.
Geometry still abstains on that higher-resolution sequence. Gate 1 remains incomplete.

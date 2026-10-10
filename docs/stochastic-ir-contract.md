# Shared stochastic IR: staged semantics and evidence contract

Date: 2026-10-10. This is a foundation for H2 with uncertain delivery/measurement,
not a claim of complete H2, novel inference or unknown-controller discovery.
It is isolated while the existing full development batch's 211 sources are
immutable. It uses the original TheoryIR state-machine representation and typed
parameters; no parallel theory/learner framework. Main-interpreter integration
and its full regression remain explicit next gates, not silently assumed.

## Probability semantics

Language `rational-stochastic-mealy-v1` has a finite initial distribution and
conditional branches (source, issued action, target, public output, rational mass).
Each state/action has exactly unit mass; positive rational weights are reduced,
bounded and exact. Duplicate outcomes must be combined. Labels are arbitrary.
No eval, file/network access or private world reads occur in execution/filtering.

Joint compilation first samples delivery/controller transition, applies a
deterministic mechanism transition only when delivered, and samples the measured
output from a stationary memoryless sensor conditional on the latent mechanism
signal. Explicit idle output defines nondelivery. Conditional factor independence,
reset of mechanism and distributional reset of controller, known action alphabet
and supplied model structure are assumptions. The controller can retain latent
modes across actions; it is not an IID fault flag. Sensor memory or unknown sensor
calibration must be modeled separately later, not claimed solved by this slice.

Filtering predicts joint next-state/output mass, conditions on the actual public
output, and propagates the normalized next-state belief without reset. A history
likelihood multiplies the conditional likelihoods. Zero likelihood means model
mismatch; no arbitrary uniform belief or reset makes contradictory evidence fit.
Sequence predictions marginalize all latent paths and sum to one or fail.
These are standard exact finite-state probability operations, not new algorithms.

Bounds: <=16 states, <=8 actions/outputs, <=2048 branches, rational denominator
<=1000000, <=64 filtering steps, <=16 prediction actions, <=4096 output prefixes
and <=100000 branch expansions per sequence prediction. Raw sampling uses a
conditional common denominator of <=256 bits. Product states/rational weights
must fit their resulting bounds; compilation fails rather than approximates or
truncates. Scope/preconditions not implemented by this runtime are rejected.

## Actual execution and audit

Raw component execution does not call the compiler or predictive interpreter.
Random sampling uses exact integer intervals; seed replay is deterministic under
the recorded Python random contract. begin_trial resets states/clocks but does
not rewind the random stream. Public records expose issued action and measured
signal only; compliance certifies command dispatch, with treatment verification
unknown. No actual controller/mechanism state, delivery flag or random seed enters
the AgentWorldView. This trusted capability boundary is not a hostile Python
reflection sandbox. Evaluator receives private spec/trace separately, after
public ledger is frozen. The owner never strongly caches its capability view.

The standalone auditor uses separate raw dictionaries and rational intervals,
not world/runtime/compiler methods. It requires an externally frozen complete
reset/action sequence and compares all before/after/event/context/provenance
fields with type-preserving canonical JSON. Dropped/extra receipts, reordered
dispatches, altered clocks/signals or coercion of true to 1 fail. Trustworthy
source/private-spec provenance must still be established by the campaign owner;
an Agent-authored truth file cannot certify itself.

The execution campaign uses four supplied-model fixtures: two seeds, with/without
measurement noise, each 3 resets/48 real atomic commands. Public predictions are
frozen before each dispatch. Every partial failure retains public receipts and
raw evaluator evidence without a completion manifest. Completion retains config,
TheoryIR, all predictions/beliefs/receipts, raw spec/trace, full replay audits and
source pins. All resets/actions pay explicit unit costs, not Minecraft time.
These are known-model semantics checks, not four independent scientific worlds.

## Unclosed research gates

Next use this representation for competing mechanism/controller/sensor hypotheses,
noise-aware experiment choice, calibrated uncertainty/abstention and independently
validated revision. Unknown parameters are not unknown structures; grammar-based
controller generation and family/grammar misspecification need separate tests.
No posterior threshold can be called a frequentist error guarantee without a
valid registered design. Adaptive stopping/repeated noisy experiments require
their own rules and costs. Current deterministic scores cannot be reused for
these probabilistic claims.

Formal structural holdouts/statistical planning, faithful random/LLM/adaptive
baselines, symbol/negative transfer, controlled Minecraft, natural RGB/navigation,
long recovery and integrated release remain required and incomplete.

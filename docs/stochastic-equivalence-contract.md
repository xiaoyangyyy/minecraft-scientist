# Exact stochastic equivalence: public-model contract

Date: 2026-10-10. Staged analysis helper on the single shared TheoryIR. Existing
211/236 frozen runtime/planner sources stay immutable during paid experiments.
Not a second learner, main-interpreter integration or original-method claim.

## Meaning and existing foundation

For each public action/output pair, construct its rational state transition
operator. A prefix probability is initial belief times these operators times
the all-ones terminal vector. The direct sum of two models with the second
belief negated represents their probability difference. A forward reachable
basis closed under every declared operator and orthogonal to the terminal
vector proves equality for all finite words. Otherwise a nonzero reachable
prefix supplies a concrete output event with differing probabilities.

This is established weighted-automaton equivalence, not new theory. Primary
reference: [Kiefer 2020, Algorithm 2 and Theorem 2.3](https://arxiv.org/html/2009.01217v1).
The dimension bounds witness length; this does not establish minimum action
cost, optimal noisy sample complexity or finite-budget empirical identification.
The action word is an experiment proposal, not an actually executed intervention.

## Inputs and full scope

Two valid total rational stochastic TheoryIR models; common public action/output
alphabets; explicit normalized exact state beliefs (defaults are model initial
beliefs); a unique allowed action tuple (defaults are all actions, empty allowed).
Beliefs are propagated without implicit reset. State naming/order may differ.
All finite words refers to this MODEL contract and action subset, not physical
world feasibility or arbitrary environment actions. Fixed-belief equality also
gives identical adaptive public-history laws under the same allowed policies,
but neither hidden mechanisms nor control factors are thereby uniquely recovered.

Do not promote a theory from an equality proof. Different physical mechanisms
may be behaviorally equivalent, especially with an uninformative sensor. A tiny
exact probability difference remains a difference, not proof that a noisy Agent
can detect it with its given budget. No prior is reweighted or made uniform over
behavioral classes. That would change inference rather than merely deduplicate.

## Certificate and independent checking

Bind both IR hashes, caller-pinned beliefs and allowed action scope. Equivalent
proofs store actual reachable vectors/prefixes, not nonphysical Gaussian residual
words. Independent checker rebuilds them using raw branch dictionaries, then
batch Gauss-Jordan verifies basis independence, initial inclusion, zero terminal
values and closure under ALL action/output pairs. Distinguishing proofs have raw
prefix probabilities independently recalculated. Checker never calls the search,
incremental basis code or compiled predict/filter/sequence functions.

Schema validation is shared with the stochastic runtime, not a fully independent
parser theorem. Search operation count is issuer-reported, shape/bounds checked
but NOT independently replayed; checker arithmetic is separately measured.
Generator operations include its rational transition/elimination arithmetic,
not full Python/IR-loading overhead. Search wall/CPU timing stops before storage;
checker/storage/total orchestration are excluded and not silently free costs.

Models have at most 16 states each; direct-sum dimension at most 32. Exact rational
coefficients at most 8192 numerator/denominator bits; search and independent audit
each at most 10 million counted rational arithmetic operations. Search can request
smaller explicit limits, never larger ones. Exceeding any bound fails without an
equivalence decision, approximate fallback, action/horizon truncation or complete
manifest. Certificates accept canonical reduced signed rationals only, not bool
integers or extra fields. Partial written proof artifacts remain on failure.

## Full public matrix and remaining gates

Complete development proof matrix: two public naming seeds, every pair among all
18 declared factor-combination programs (306 proofs). Every proof independently
checked; public definitions, certificates, checks, rows and denominators retained.
This reads no private world selection or raw experiment evidence and executes zero
physical trials. Naming repetitions are not independent scientific worlds.

The Agent planner still retains its declared length<=4 finite query space. This helper
does NOT change that algorithm or replace the real budget-96 noisy matrix with
easier symbolic tests. Unknown controller grammar synthesis, model adequacy and
probability calibration, valid noisy shadow promotion, online revision, formal
holdouts/statistics, faithful strong baselines, H4 symbolic/negative transfer,
Minecraft, RGB/navigation/long recovery, integration and release remain required.

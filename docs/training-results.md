# Experiment Planner Training Results

Date: 2026-09-16

## Promoted four-form policy

The current promoted checkpoint is
`checkpoints/four-form-experiment-policy.pt` (SHA-256
`CEBB08E5D939AAEF813B2670D1A615CBE2686151136CF79C05CCD594A9B51CB4`).
It uses one permutation-equivariant policy and one six-feature experiment schema
for Rule, State Machine, Program, and Equation theory spaces.

- Training worlds: 15,000
- Curriculum: 30% rules, 20% state machines, 25% programs, 25% equations
- Training examples: 88,412
- Epochs: 50
- Batch size: 256
- Hidden size: 128
- Exploration-state fraction: 25%
- Initial/final loss: 0.965969 / 0.513853

Training-time held-out evaluation (2,000 worlds per form):

| Form | Success | Optimal action | Learned experiments | Oracle | Random |
|---|---:|---:|---:|---:|---:|
| Rule | 1.000 | 1.000 | 7.1775 | 7.2065 | 8.4130 |
| State Machine | 1.000 | 1.000 | 6.0030 | 6.0190 | 8.4325 |
| Program | 1.000 | 1.000 | 5.2900 | 5.3355 | 8.5215 |
| Equation | 1.000 | 1.000 | 3.0000 | 3.0000 | 6.8230 |

The promotion gate then used seed 9,000,000 and 5,000 new worlds per form
(20,000 total):

| Form | Success | Optimal action | Learned experiments | Oracle | Random |
|---|---:|---:|---:|---:|---:|
| Rule | 1.000 | 1.000 | 7.2112 | 7.2026 | 8.3736 |
| State Machine | 1.000 | 1.000 | 5.9646 | 5.9056 | 8.4914 |
| Program | 1.000 | 1.000 | 5.3424 | 5.3622 | 8.3884 |
| Equation | 1.000 | 1.000 | 3.0000 | 3.0000 | 6.8440 |

An additional risk-bearing promotion gate used 5,000 new curriculum worlds:

| Metric | Result |
|---|---:|
| Partial evidence states | 29,679 |
| Nonzero-risk actions | 122,606 |
| Raw neural maximum-utility action rate | 0.999798 |
| Entropy-only maximum-utility action rate | 0.657839 |
| Runtime constrained-decoding action rate | 1.000000 |

The runtime planner first restricts actions to the exact maximum of
`information gain - 0.05 * normalized cost - 0.5 * control-failure risk`, then
uses the neural policy only to break remaining ties. The raw neural result is kept
separate from this formal guarantee.

The State Machine space models wait tokens as minimum-duration thresholds: observing
that `wait:2` succeeds does not erase the competing `wait:1` theory. The policy uses
the shorter intervention to distinguish them and substantially outperforms random
enumeration. The Program training space represents 288 initial hypotheses: every
opaque processor-labelled Hamiltonian chain both with known terminals and with an
initially unseen terminal product. Each observation eliminates globally inconsistent
programs, and the learner expands the resource universe only after a new product is
observed.

On the identical seed-9,000,000 promotion set, the superseded 64-wide policy had a
raw risk-aware action rate of 0.998821. The promoted 128-wide policy reached
0.999798 while preserving 1.0 constrained action rate, success, and optimal-action
rate for every symbolic form.

The promoted policy was then loaded into both real Minecraft curricula. The combined
Fabric 1.21.11 run recovered every hidden mechanism and both equation versions at
1.0 accuracy in 47 experiments. Relative to the exact-utility live run, it reduced
the discrete phase from 42 to 40 experiments and tool-theory revision from 8 to 7,
while retaining three learned symbols and 13 verified reuse records. The artifacts
are in `runs/minecraft-live-benchmark-learned`.

## Earlier two-form policy

## Environment

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB
- Driver: 591.74
- PyTorch: 2.11.0+cu128
- CUDA runtime: 12.8
- Training worlds: 5,000
- Curriculum: 65% ordered capability rules, 35% piecewise equations
- Training examples: 30,015
- Epochs: 30
- Batch size: 256

## Training run

The information-gain oracle was distilled across both symbolic forms into the
permutation-equivariant ExperimentPolicy and saved to
`checkpoints/multiform-experiment-policy.pt`.

| Metric | Result |
|---|---:|
| Initial loss | 1.748260 |
| Final loss | 1.080067 |
| Held-out worlds | 500 |
| Success rate | 1.000 |
| Optimal-action rate | 1.000 |
| Learned mean experiments | 7.220 |
| Oracle mean experiments | 7.204 |
| Random mean experiments | 8.416 |

Continuous-law validation:

| Metric | Result |
|---|---:|
| Success rate | 1.000 |
| Optimal-action rate | 1.000 |
| Learned mean experiments | 3.000 |
| Oracle mean experiments | 3.000 |
| Random mean experiments | 7.052 |

## Independent evaluation

An independent run used seed 4,000,000 and 5,000 newly sampled rule worlds.

| Metric | Result |
|---|---:|
| Success rate | 1.000 |
| Optimal-action rate | 1.000 |
| Learned mean experiments | 7.1932 |
| Oracle mean experiments | 7.2192 |
| Random mean experiments | 8.3822 |

Independent continuous-law evaluation used 5,000 additional worlds:

| Metric | Result |
|---|---:|
| Success rate | 1.000 |
| Optimal-action rate | 1.000 |
| Learned mean experiments | 3.000 |
| Oracle mean experiments | 3.000 |
| Random mean experiments | 6.9468 |

The learned checkpoint was then inserted into the complete ScientistAgent loop.
Across 100 additional opaque-ID worlds, hidden-intervention accuracy was 1.0 in
every world. With cross-world symbol-validation experiments included in the event
budget, the learned planner averaged 47.53 total experiments across all mechanisms,
theory revision, and library validation, compared with 47.49 for the exact shared
utility planner. Both selected a maximum-utility action at every policy decision;
the small trajectory difference comes from equally valued tie-breaking. The
maximum learned-policy run used 60 experiments, below the default budget of 80.

The learned checkpoint also drove the complete `full` benchmark at seed 42: all
symbolic forms remained executable, hidden mechanism accuracy was 1.0, fall-law
accuracy was 1.0, transfer saved one experiment, and the changed equation was revised
to 1.0 accuracy. These remain structured-protocol results, not evidence of visual
Minecraft performance; live evaluation begins after the server EULA is accepted.

## Offline baseline matrix

The deterministic ten-seed baseline gate (`seed=0..9`) produced:

| Baseline | Result |
|---|---:|
| Active tool discovery, mean experiments | 6.5 |
| Random tool discovery, mean experiments | 8.6 |
| Stateless context-only intervention accuracy | 0.6667 |
| Voyager-style skill memory, same-world accuracy | 1.0000 |
| Voyager-style skill memory, randomized-ID transfer | 0.6667 |
| Fixed DSL hidden-mechanism accuracy | 1.0000 |
| Full evolving-library hidden-mechanism accuracy | 1.0000 |
| Fixed DSL symbol count / reuse | 0 / 0 |
| Full symbol count / verified cross-world reuse | 3 / 13 |

This matrix isolates the mechanisms that can be evaluated offline. Context-only and
Voyager-style runs backed by a paid external LLM remain explicitly unmeasured; the
report does not substitute the deterministic prior or skill-memory rows for them.

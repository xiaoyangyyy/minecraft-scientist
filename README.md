# Minecraft Scientist

Minecraft Scientist is an offline, deterministic research MVP for active symbolic
world-model discovery. It studies whether an agent can select interventions,
recover executable theories, revise them after a hidden rule change, and compress
repeated mechanisms into reusable symbols.

The deterministic research core uses a structured opaque world. A live Minecraft
path is also implemented: Mineflayer performs whitelisted actions, a server-only
Fabric mod supplies resettable hidden rules, and the Python boundary replaces item,
block, and entity names with seeded opaque IDs before an agent can observe them.
Live execution still requires the user's explicit Minecraft EULA acceptance.

## Quick start

    uv sync
    uv run pytest
    uv run minecraft-scientist run --seed 0
    uv run minecraft-scientist benchmark --suite mvp --seed 0
    uv run minecraft-scientist benchmark --suite full --seed 0
    uv run minecraft-scientist benchmark-baselines --seed 0 --seeds 10

With the training extra installed, run the paired component ablations:

    uv run minecraft-scientist benchmark-ablations --seed 0 --seeds 10

This compares informative random selection, disabled theory repair, disabled library
learning, withheld cross-world symbols, and the trained planner against matched
references. Results include per-seed samples and approximate 95% intervals in
`runs/strict-ablations/ablation-matrix.json`. The transfer probe executes the source
world's learned transform primitive. Each condition has the same intervention cap.

Use --overwrite when intentionally re-running into the same output directory.
Run artifacts contain config.json, events.jsonl, theories.jsonl, symbols.json,
and report.json.

Run every offline build/test gate, optionally including all benchmark families:

    powershell -ExecutionPolicy Bypass -File scripts/verify.ps1
    powershell -ExecutionPolicy Bypass -File scripts/verify.ps1 -Benchmarks

Inspect a theory history with:

    uv run minecraft-scientist inspect-theory runs/run-seed-0 tool-capability

## Learned experiment planner

Install the CUDA training extra and reproduce the planner training or evaluation:

    uv sync --extra train
    uv run minecraft-scientist train-planner --device cuda
    uv run minecraft-scientist evaluate-planner checkpoints/four-form-experiment-policy.pt

The promoted local training run used a 15,000-world mixed curriculum containing
rules, hidden state machines, transformation programs (including initially unseen
terminal products), and continuous piecewise equations. It was followed by an
independent 5,000-world evaluation for each form plus 5,000 risk-bearing worlds
(25,000 worlds total). Use it in the complete benchmark with:

    uv run minecraft-scientist benchmark --suite full --seed 42 --planner learned --planner-checkpoint checkpoints/four-form-experiment-policy.pt

See `docs/training-results.md` for exact metrics and hardware. The learned policy
chooses experiments from information gain, normalized cost, and control-failure
risk; evidence, symbolic induction, validation, crisis handling,
and evaluation remain explicit code. Symbol candidates are promoted only after
additional interventions in a new randomly named held-out world; those validation
events count against the run budget and are persisted with namespaced provenance.

## Optional live Minecraft backend

The bridge is pinned to Minecraft Java 1.21.11 because that is the newest protocol
currently supported by the installed Mineflayer 4.39.0. The server artifact is
pinned to Mojang's official URL, byte size, and SHA-1. Java 21 and Node 22 are
required.

    cd bridge/mineflayer
    npm ci
    cd ../..
    powershell -ExecutionPolicy Bypass -File mod/scientist-world-rules/build.ps1
    uv run minecraft-scientist check-minecraft-backend
    uv run minecraft-scientist minecraft-server-status

Minecraft's server download requires accepting the Minecraft EULA. The project
does not accept it implicitly. After reviewing https://www.minecraft.net/eula,
the user can explicitly prepare and start the local research server:

    uv run minecraft-scientist minecraft-server-prepare --accept-eula --seed 0
    uv run minecraft-scientist minecraft-live-smoke --seed 0
    uv run minecraft-scientist minecraft-live-run --seed 0
    uv run minecraft-scientist minecraft-live-fall --seed 0
    uv run minecraft-scientist minecraft-live-benchmark --seed 0

To run the trained planner through both real-server curricula, add:

    --planner learned --planner-checkpoint checkpoints/four-form-experiment-policy.pt

The promoted policy's seed-0 live acceptance run completed both curricula in 47
experiments with 1.0 mechanism accuracy, law recovery, transfer, and task success.

Preparation installs the pinned Fabric loader launcher, the pinned Fabric API, and
the compiled research mod using checksum lock files. `minecraft-live-smoke` checks
the whole process boundary. `minecraft-live-run` then executes the same active
hypothesis, intervention, validation, symbol-learning, and Theory Crisis loop used
by the structured benchmark against the live resettable arena. The arena contains
an ordered mining rule, a timed hidden-state machine, and a three-stage conversion
program; the shifted episode changes all three mechanisms. `minecraft-live-fall`
runs a separate three-token, ten-height active equation curriculum, detects the
server-side law change, and promotes an independently induced v2. For a manually
managed long-running server use:

    uv run minecraft-scientist minecraft-server-start

In a second terminal, a raw controller diagnostic is available with:

    uv run minecraft-scientist minecraft-observe --version 1.21.11

Programmatic agent integrations should use `MinecraftWorldAdapter`, not the raw
diagnostic. The adapter satisfies the `observe`, `available_actions`, and `step`
contract and maintains an opaque spatial cognitive map.

## What the benchmarks discover

1. A latent ordered capability rule connecting three opaque tools and targets.
2. A hidden machine protocol including input choice and action order.
3. A three-stage resource transformation chain.
4. A reusable transform primitive when library learning is enabled.
5. A second tool theory version after a silent environment rule change.
6. A piecewise continuous fall-damage equation with latent armor coefficients.
7. Local `height -> damage` and `armor -> damage` causal edges from controlled
   interventions only.
8. A transferred equation structure on a changed map and a revised equation after
   an independently generated rule mutation.

The `full` suite writes an append-only SQLite experience database, an equation
theory history, a causal graph, the original structured artifacts, and a combined
report. Twenty randomized seeds currently pass law recovery, transfer, and revision.

## Optional OpenAI proposal adapter

The default budget is 80 controlled experiments. The default heuristic generator
is completely offline. To enable the optional
proposal-only adapter:

    uv sync --extra openai
    set OPENAI_API_KEY=your-key
    uv run minecraft-scientist run --hypothesis-generator openai

The adapter uses the Responses API with Structured Outputs and store disabled.
LLM output is only a proposal; local executable validation remains authoritative.

The MVP benchmark also runs a random-exploration baseline over three randomized
ID worlds and a held-out transfer probe. Its ablation report shows description
length reduction and downstream experiment savings from the learned primitive.
`benchmark-baselines` adds a ten-world matrix for random exploration, a stateless
context-only prior, Voyager-style opaque skill memory, fixed symbolic DSL, and the
full evolving-library system. Paid context-only and Voyager-style LLM rows remain
explicitly `not_run` inside that offline matrix; the separate cost-gated harness
produces measured external-provider artifacts.

The real comparison harness is available behind fail-closed request/token limits:

    uv sync --extra openai
    set OPENAI_API_KEY=your-key
    uv run minecraft-scientist benchmark-external-llm --seeds 3 --max-api-calls 9 --max-total-tokens 50000 --confirm-api-cost

DeepSeek's stateless Responses API is supported through the same gates:

    set DEEPSEEK_API_KEY=your-key
    uv run minecraft-scientist benchmark-external-llm --provider deepseek --model deepseek-flash --reasoning-effort none --seeds 3 --max-api-calls 9 --max-total-tokens 50000 --confirm-api-cost

It makes three non-stored Responses API calls per seed pair: a context-only target
prediction, a source-world skill-memory extraction, and a Voyager-style transfer
prediction. Prompts contain only public opaque observations and experimental
evidence. Reports retain response IDs, prompt hashes, and provider usage, but never
the API key. A usage ledger is flushed after every response, so a later failure
cannot hide already consumed calls or tokens.

The multi-mechanism shared-evidence benchmark adds tool capability, hidden machine,
and transformation-chain predictions, a same-evidence symbolic posterior, per-pair
checkpoints, bounded transient retries, and 95% confidence intervals:

    uv run minecraft-scientist benchmark-external-llm-expanded --pairs 20 --evidence-budget 6 --max-transient-retries 6 --confirm-api-cost

On Windows, DeepSeek credentials can be resolved from the generic Credential Manager
target `minecraft-scientist/deepseek` when `DEEPSEEK_API_KEY` is absent. The key is
never written to run artifacts. Environment variables and explicit test injection
remain supported.

The recorded DeepSeek acceptance run at seeds 0–5 used nine successful calls and
9,897 total tokens. Context-only accuracy was 0.7407; Voyager-style cross-world
memory accuracy was 0.8889, with 1.0 coverage for both. The artifact is written to
`runs/deepseek-external-baselines` and remains ignored by Git like all run data.

The recorded 20-pair shared-evidence run used 63 calls including three audited
retries and 305,554 tokens. Mean exact intervention accuracy was 0.8000 for
context-only DeepSeek, 0.7280 for Voyager-style transfer, and 0.9646 for the
symbolic posterior. The conservative peak/cache-miss estimate was ¥0.9877. Artifacts
are in `runs/deepseek-expanded-baselines`.

## Optional MineStudio visual boundary

`MineStudioVisualAdapter` provides a tested RGB-frame/perception-controller boundary
without adding MineStudio's heavy runtime to the default environment. Check the
current machine without launching anything:

    uv run minecraft-scientist check-minestudio-backend

The isolated WSL runtime on this host is now verified with Ubuntu 22.04, JDK 8,
MineStudio 1.1.6 and a persistent simulator engine. Real RGB reset/step and a
strictly controlled active stone-interaction experiment have completed. The
accepted seeds-22/23 run used two opaque interventions, selected both through the
production adapter, froze predictions before the holdout, passed 4/4 equipment
and action-log controls, automatically localized pixel changes on an anonymous
grid, matched the changed region across seeds at IoU 0.857, and predicted both
holdout outcomes. Its
scope is intentionally narrow: fixed camera, one fixture target, 20 attack frames,
and fixture-supplied identity bindings. It is not general object recognition or a
complete visual Minecraft scientist. Setup, exact evidence, and caveats are in
`environments/minestudio/README.md` and `docs/completion-audit.md`.

A separate preregistered seed-30 camera-search suite now exercises explicit
`LOOK` actions over negative, center, and positive yaw views. The pixel-only
selector found the fixture marker in all 3/3 independently seeded placements;
15/15 camera actions passed, and the independent audit recomputed every score
from the saved PNGs. This proves controlled active viewpoint search, while leaving
free-form exploration, natural-scene detection, and path planning open.

The preregistered seed-49 navigation suite adds an opaque `MOVE` handle and a
closed-loop RGB stopping policy. In all 3/3 real MineStudio runs, the Agent stopped
at the first preregistered central-view threshold crossing after seven four-frame
moves. An evaluator-only telemetry channel independently measured 7.504 blocks of
displacement in every case; those fields were stripped before the adapter. The
audit recomputed every PNG score, verified all movement events, and confirmed that
no coordinates entered the Agent event log.
This is evidence for controlled straight-corridor visual approach, not arbitrary
multi-direction navigation, obstacle avoidance, or general path planning.

The stronger preregistered seed-62/63 direction-discovery suite removes the known
safe-handle assumption. For each case it independently permutes four opaque MOVE
handles, probes every handle in reset-identical RGB trials, selects the unique
largest visual improvement, and then approaches with that handle. World seeds and
mapping seeds are independent; the two winners occupied different anonymous list
indices (3 and 1), both matched evaluator-measured progress toward the target, and
both completed the 7.504-block approach. The independent audit recomputed all
calibration and approach PNG scores and confirmed no direction labels or coordinates
entered public Agent artifacts. This is active visual motor calibration in a
controlled corridor.

The seed-88/89 obstacle-replanning suite adds a visible, physical fixture obstacle
after the same four-handle RGB calibration. In each continuous execution episode,
the Agent detects central obstacle-color occupancy, selects a learned lateral
handle until the direct corridor is clear, follows the learned target handle past
a preregistered safety gate, and recenters from the marker centroid. Both 2/2
independently permuted cases passed: 7.594 blocks of target-axis progress, 2.584
blocks of maximum lateral excursion, and 0.066 blocks of final lateral offset.
The independent audit recomputed every calibration and route PNG, verified every
MOVE event and source hash, confirmed the two paths never entered the obstacle
volume, and found no direction or coordinate leakage in public artifacts. This is
online visual detour selection around one color-coded fixture obstacle; it is not
natural-scene obstacle recognition, a learned metric map, or general path planning.

The seed-120 source episode and seed-121/122 held-out episodes add persistent,
coordinate-free route memory. `VisualTopologicalMap` hashes quantized RGB landmark
signatures into opaque nodes and stores the successful 13-step route as functional
motor roles, never evaluator coordinates or episode-specific control handles. Both
held-out episodes loaded the same map from disk, matched its source landmark, rebound
the program to independently permuted opaque controls, and completed collision-free
routes without online route exploration. The independent audit recomputed every PNG
feature, role classification, action binding, route event, private path metric, hash,
and non-leakage check; its result is valid with 2/2 replay success. This demonstrates
fixture-bound topological route persistence and reuse, not arbitrary-layout route
search, localization in natural scenes, or a general cognitive map.

The seed-140/141 source and seed-142/143 held-out branching suite goes beyond a
single stored route. Two offset-obstacle layouts produce distinct opaque start
landmarks and mirror-image 11-step programs in one version-2 multi-edge topology.
In held-out worlds, RGB signatures selected the correct branch under new control
permutations; both routes succeeded without online exploration and avoided their
layout-specific obstacle volumes. A deliberately unseen between-layout signature
was rejected rather than mapped to the nearest route. Independent audit verified
the 3-node/2-edge map, every PNG and action binding, both private paths, source
snapshots, and public non-leakage. This is real visual layout-conditioned route
selection with fail-closed novelty handling, still limited to two controlled,
color-coded layouts under a fixed camera.

Format version 3 added an explicit route lifecycle. A replay outcome
updates evidence and cost without changing route identity; a failure of a previously
high-confidence program moves it to `challenged`, immediately excluding it from
planning. Independently validated alternatives can supersede it with bidirectional
parent/replacement links, while rejected programs remain preserved for audit rather
than being overwritten. The current version-4 writer additionally supports an
independently fitted match radius on each landmark; versions 1–3 remain loadable.
Inspect any saved map with:

```powershell
uv run minecraft-scientist inspect-visual-map path\to\visual-topology.json
uv run minecraft-scientist inspect-visual-map path\to\visual-topology.json --json
```

The seed-140 visual-route-crisis run exercises that lifecycle in the real isolated
MineStudio simulator. Two independently permuted source episodes establish the old
route at 0.75 confidence. A mirror-layout replay then executes all 11 bound actions
successfully but advances only 2.20 blocks and fails to pass the obstacle, so the
route becomes challenged. Online RGB-only replanning learns the mirror program;
an independent shadow episode with another control permutation advances 7.594
blocks, avoids the evaluator-only obstacle volume, and validates promotion. The
final map preserves the old program as `superseded` and the replacement as `active`
with reciprocal lineage. The independent audit recomputes every image feature,
action binding, path metric, map transition, source hash, and non-leakage check.
Evidence is in `runs/visual-route-crisis-seed140-map1400`.

The seed-143 multi-stage suite adds actual graph composition. Two independently
permuted source episodes contribute separate start-to-midpoint and midpoint-to-goal
edges to a three-node map; there is no direct start-to-goal edge. In the held-out
MineStudio episode, the Agent executes the first five-role edge, recomputes the RGB
signature `(0,-1,0,16,13)`, matches the expected midpoint, invokes the planner again,
and executes the remaining six-role edge. It reaches 7.594 blocks of progress with
1.720 blocks of lateral excursion and 0.066 final offset, without online exploration.
Challenging the second edge makes the complete plan unavailable. The independent
audit recomputes both source episodes, every replay image/action/path, both boundary
matches, the second planning decision, hashes and non-leakage checks. Evidence is in
`runs/multi-stage-route-seed143-map1415`.

The adaptive-landmark-calibration gate replaces the coarse global threshold for one
source landmark with a supervised, persisted boundary. It fits only from two audited
positive RGB episodes and one audited negative episode, using
`radius = maximum positive distance + margin`; the nearest negative must remain a
second margin beyond that radius. In the accepted evidence, the fixed tolerance 8.25
false-accepts the mirror layout at distance 8.0, while the fitted radius 1.0 accepts
the disjoint held-out positive and rejects both the held-out negative and a synthetic
between-layout control. The independent auditor regenerates features from copied PNGs,
recomputes all distances, checks source/evidence hashes, and validates the saved v4
map. Evidence is in `runs/adaptive-landmark-calibration`. This establishes one
auditable controlled-fixture calibration, not a generally learned visual metric.

The next-stage exploration foundation adds an append-only `ExplorationMemory` backed
by SQLite. Anonymous places retain a bounded multi-view exemplar coreset, calibrated
match radius, visits and episode history; candidate action frontiers retain attempts,
failure streaks, risk, uncertainty and resolved destinations across process restarts.
The planner routes over previously successful directed transitions and then chooses
the reachable active frontier with the best novelty/information/cost/risk score.
`minecraft-scientist inspect-exploration-memory` verifies the journal hashes and
shows the durable projections. The deterministic six-place gate can be run with:

```powershell
uv run minecraft-scientist benchmark-exploration --seed 17 --output runs/persistent-exploration-seed17
```

That gate reaches 1.0 coverage and place recall through four forced restarts, with no
duplicate places or unresolved frontiers. It is deliberately an offline hidden-graph
test. On the real Mineflayer boundary, setting `exploration_stride` now exposes four
opaque coordinate targets with `MOVE` affordances and translates the selected target
to the bridge's existing `move_to` controller. A bounded real-server gate can be run
against an explicitly prepared vanilla server:

```powershell
uv run minecraft-scientist minecraft-exploration-smoke --server-directory minecraft-server --seed 31 --output runs/minecraft-open-world-movement
```

The clean-world seed-31 evidence in
`runs/minecraft-open-world-movement-fresh-seed31` completed its first anonymous
frontier with 2.826 blocks of displacement, 1.174 blocks of target error, a resolved
frontier, one persisted directed transition and a valid memory integrity check. The
bridge gives path search its own bounded computation budget and the gate can try up
to four anonymous directions, persisting blocked attempts instead of treating an
obstacle as controller success. This removes the previous inability to move
autonomously and grounds one short traversal in the persistent topology. It does not
yet prove long-horizon survival, chunk-scale coverage or general visual place
recognition in an uncontrolled Minecraft world.

`minecraft-explore` extends that gate into a bounded multi-session episode. It closes
and recreates the Mineflayer controller at a fixed interval, reopens the SQLite
memory each time, prefers novel active frontiers, and can bind a known route back to
a remote unresolved frontier. The same database now stores a bounded coreset of
34-dimensional anonymous structural scene views for each place. The encoder ignores
opaque IDs, object order and semantic block names, and does not serialize absolute
coordinates; it uses public coordinates only to define a fixed local grid anchor,
then summarizes geometry, hardness and diggability. It is invariant to whole-cell
translations. Ambiguous scene matches fail closed.

```powershell
uv run minecraft-scientist minecraft-explore --server-directory minecraft-server --seed 59 --move-budget 12 --restart-interval 3 --output runs/minecraft-open-world-perceptual-seed59
```

The clean-world seed-59 evidence completed 8 valid place-changing moves in 12
attempts, travelled 38.140 blocks, reached 18.385 blocks from the start, and retained
9 places, 8 directed transitions and 36 frontiers across four controller sessions.
Coordinate restart recognition and anonymous-scene restart recognition were both
3/3; known-place scene retrieval was 4/4 with no false match. One new place had a
scene descriptor identical to an earlier place, so the topology treated it as an
uncommitted alias rather than a loop closure. This is meaningful long-term structural
perception, but it is still not pixel-level visual generalization or unbounded
survival exploration.

The visual controller now publishes a 768-dimensional `rgb-lab-grid-v1` signature
derived directly from every RGB frame. It uses a central CIELAB spatial grid,
per-channel normalization and no semantic labels or learned weights. SQLite keeps
separate bounded coresets and calibrated radii per modality, so RGB and structural
views cannot silently overwrite one another. Run the disjoint-episode RGB gate with:

```powershell
uv run --extra visual minecraft-scientist benchmark-rgb-generalization --evidence-root runs/adaptive-landmark-calibration/evidence --output runs/rgb-place-generalization-v1
```

The accepted real MineStudio evidence fits a radius of 48.116 from 15 training
positives and one negative. Across 15 held-out positive transformations and 15
held-out negative transformations, acceptance and rejection are both 1.0. The
held-out positive maximum distance is 23.324 while the nearest held-out negative is
173.721. An independent audit reloads copied PNGs, re-encodes every view, validates
role separation and evidence hashes, replays SQLite, and recomputes all decisions.
This improves controlled RGB generalization substantially, but remains a hand-built
fixture-calibrated metric rather than a learned natural-biome representation.

The next gate removes the constructed arena. `RGBPanoramaEncoder` aggregates a full
active camera scan into a 1,536-dimensional order-independent signature, so the scan
can start at any yaw without introducing a compass label. In the accepted seed-200
MineStudio run, the RGB-only policy inferred a 12-intervention turn period, trained
its place radius across independent restarts plus one physical out-and-back return,
then ran a disjoint held-out loop on untouched generated forest/village terrain:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/natural_visual_loop_benchmark.py --seed 200 --moves 4 --move-frames 4 --output /mnt/d/minecraft-scientist/runs/natural-visual-loop-seed200-v2
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_natural_visual_loop.py /mnt/d/minecraft-scientist/runs/natural-visual-loop-seed200-v2 --write
```

The calibrated radius is 47.706 versus an 88.425 nearest training negative. The
held-out Agent moved 4.161 blocks from the start and physically returned within
0.200 blocks; its start and return panoramas were recognized, its outbound panorama
was rejected, and a later process restart was recognized from the reopened SQLite
memory. The independent audit re-encodes every saved PNG, verifies the physical
loop from evaluator-only coordinates, scans public artifacts for telemetry leakage,
and reports `valid: true`. This establishes one bounded natural-terrain pixel loop
under fixed noon/clear/peaceful conditions—not learned cross-biome generalization,
long-horizon autonomous exploration, survival, or metric SLAM.

A stricter condition-split gate now trains only at noon/clear and evaluates at
night/rain. `RGBEdgePanoramaEncoder` replaces color appearance with anonymous edge
occupancy/orientation, every formal panorama heading is the pixelwise median of five
raw RGB frames, and the maximum-gap radius is derived only from training positives
and negatives. Once the old memory accepts the held-out night/rain start view, a
separate fail-closed consolidation operation may add that accepted appearance to its
bounded coreset; rejected views cannot teach the memory about themselves. Reproduce
the accepted seed-200 run with:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/natural_visual_loop_benchmark.py --seed 200 --moves 4 --move-frames 4 --auto-margin --temporal-samples 5 --panorama-encoder edge --training-time noon --training-weather clear --heldout-time night --heldout-weather rain --require-condition-shift --output /mnt/d/minecraft-scientist/runs/natural-visual-loop-seed200-edge-night-rain-v12
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_natural_visual_loop.py /mnt/d/minecraft-scientist/runs/natural-visual-loop-seed200-edge-night-rain-v12 --write
```

The training-only radius is 58.036 (positive maximum 53.423; nearest negative
62.650). The held-out start was accepted at distance 41.725 and safely consolidated
online before movement; the later return was accepted at 43.966, while the outbound
view remained rejected.
The Agent moved 4.161 blocks and returned with evaluator error 0.000; a fresh
night/rain process restart was also recognized. The audit independently reconstructs
all 480 temporal source frames, replays both consolidation events, and reports
`valid: true`.

The seed-201 ridge follow-up expands the anonymous action space with a compound
forward-plus-jump program. Six RGB-selected headings are crossed with both movement
programs in independent reset sessions; selection still uses pixels only, while an
isolated evaluator checks the chosen probe after selection. Version-2 scene encoders
exclude the top HUD/toast band as well as lower/peripheral HUD. Calibrated positive
samples estimate coverage around existing exemplars without silently becoming new
centers. Later accepted weather views receive per-exemplar radii capped below the
nearest persisted negative by the fitted calibration margin, preventing continual
adaptation from consuming a known counterexample.
Reproduce the accepted run with:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/natural_visual_loop_benchmark.py --seed 201 --moves 4 --move-frames 4 --auto-margin --temporal-samples 5 --panorama-encoder edge --select-reversible-heading --heading-probe-candidates 6 --enable-jump-program --training-time noon --training-weather clear --heldout-time night --heldout-weather rain --require-condition-shift --output /mnt/d/minecraft-scientist/runs/natural-visual-loop-seed201-jump-edge-night-rain-v3
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_natural_visual_loop.py /mnt/d/minecraft-scientist/runs/natural-visual-loop-seed201-jump-edge-night-rain-v3 --write
```

The chosen jump program reached 5.429 blocks and returned within 0.651 during its
probe. In the held-out night/rain loop it reached 5.644 blocks and returned within
0.006. The training-only radius is 68.588 (positive maximum 62.578; nearest negative
74.599); start and return are recognized, outbound is rejected, restart recognition
survives reopening SQLite, and the independent audit reports `valid: true` over 624
saved PNGs and 1,095 public transitions. This is stronger evidence for terrain,
weather and persistent-memory robustness across two natural seeds. It is still a
bounded out-and-back experiment, not cross-biome, long-horizon open-world navigation,
survival, or metric SLAM.

The natural visual expedition gate extends that loop into a persistent directed
place graph. It revalidates a saved anonymous action program, learns A→B→C under
noon/clear plus night/rain calibration, and evaluates under unseen
midnight/thunder. At C it probes four visually open headings in isolated A→B→C
sessions; RGB out-and-back recovery selects heading 3, while coordinates remain
evaluator-only. After a fresh restart, the frontier planner emits both executable
travel programs and the unresolved remote intervention. The Agent recognizes A, B
and C, discovers novel D, resolves C→D, then restarts again and plans/replays
A→B→C→D from SQLite.

Reproduce the accepted seed-201 evidence and its independent audit with:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/natural_visual_expedition_benchmark.py --seed 201 --moves 4 --move-frames 4 --temporal-samples 3 --heading-probe-candidates 6 --frontier-probe-candidates 4 --reuse-heading-probes /mnt/d/minecraft-scientist/runs/natural-visual-expedition-seed201-v5 --panorama-encoder edge --training-time noon --training-weather clear --calibration-time night --calibration-weather rain --heldout-time midnight --heldout-weather thunder --output /mnt/d/minecraft-scientist/runs/natural-visual-expedition-seed201-v6
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_natural_visual_expedition.py /mnt/d/minecraft-scientist/runs/natural-visual-expedition-seed201-v6 --write
```

Omit `--reuse-heading-probes ...` to regenerate the complete 6×2 anonymous action
probe matrix instead of reusing and independently revalidating the saved selection.

The selected remote probe moved 5.093 blocks and returned within 0.603. Both the
held-out frontier episode and the independent restart revisit reached 9.466 blocks;
all four place decisions were correct on revisit. The hash-checked memory contains
four places, three validated forward transitions and 27 journal events. The run
saves 1,155 PNGs and 1,372 public transitions; the independent audit recomputes the
images, cyclic heading alignments, frontier selector, route plan, physical metrics,
memory projection and frozen-source hashes and reports `valid: true`.

This is the first real natural-terrain, pixel-only, multi-frontier expedition in the
project. The original v6 artifact remains a bounded outward expedition and contains
no inferred reverse edges.

The adaptive visual homing gate builds on that immutable artifact without changing
its place boundaries. Exact 16×16 edge panoramas still perform topological
localization and cyclic heading alignment. A second edge-spectrum modality uses
log Fourier magnitudes to tolerate sub-block screen translation while retaining
more spatial structure than an edge histogram. Its per-place radii are fitted only
from copied v6 training/calibration PNGs and frozen negative places. `VisualHomingProgram`
persists an anonymous motion slot, learned half-turn, minimum/maximum intervention
count, frame budget and post-motion settling budget. It scans after every action,
stops at the first target-spectrum match after the minimum intervention count, and
fails closed if the maximum is exhausted.

The next gate keeps arrival recognition versioned per directed edge instead of
forcing every terrain segment through one visual bandwidth. The wider spectrum is
retained for D→C, while a separately versioned radius-3 compact spectrum tolerates
the viewpoint change at B. The compact representation was selected on a failed
development trajectory, but its place radii still use only copied v6
training/calibration PNGs and frozen negative places; none of the development
frames become memory exemplars.

Reproduce the accepted D→C→B discovery plus fresh-session shadow validation with:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/adaptive_visual_homing_benchmark.py --source-run /mnt/d/minecraft-scientist/runs/natural-visual-expedition-seed201-v6 --homing-edge-count 2 --homing-edge-spec 6,1,6,8,4,10 --homing-edge-spec 6,1,1,8,1,10 --homing-edge-encoder spectrum-r4 --homing-edge-encoder spectrum-r3 --temporal-samples 3 --output /mnt/d/minecraft-scientist/runs/adaptive-visual-homing-seed201-v23-edge2-multimodal
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_adaptive_visual_homing.py /mnt/d/minecraft-scientist/runs/adaptive-visual-homing-seed201-v23-edge2-multimodal --write
```

Both new sessions independently recognize A→B→C→D under midnight/thunder, stop
D→C on intervention 6 and C→B on intervention 5. Evaluator-only telemetry now
recomputes three-dimensional arrival error against the target's outward checkpoint:
0.399 blocks for C and 0.386 blocks for B in both sessions. Only after independent
validation are both reverse edges persisted, producing five directed transitions,
57 hash-chain events and the executable D→C→B return plan. The run stores 1,548
PNGs, frozen source/evidence manifests and an independent audit with all 19 checks
true.

This is a verified two-edge reverse prefix, not complete homing. B→A remains absent,
and cross-seed/biome generalization, resource-driven exploration, survival,
long-horizon recovery and metric SLAM remain open.

The natural visual detour gate addresses the first non-reversible edge instead of
assuming that every saved action can be played backwards. From B it ranks visually
open headings, runs each B→candidate→B program in an isolated simulator session,
and selects only from candidates whose RGB evidence returns to B. A dual-band edge
spectrum supplies condition-shift retrieval, while three-frame temporal medians
suppress thunder/rain transients. Coordinates remain evaluator-only and cannot
influence candidate selection. A selected edge is written to SQLite only after a
fresh-session shadow validation also returns within 0.75 blocks.

Reproduce the accepted B↔E discovery and independent audit with:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/natural_visual_detour_benchmark.py --source-run /mnt/d/minecraft-scientist/runs/natural-visual-expedition-seed201-v6 --candidate-count 2 --moves 3 --move-frames 4 --temporal-samples 3 --output /mnt/d/minecraft-scientist/runs/natural-visual-detour-seed201-v6-audited
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_natural_visual_detour.py /mnt/d/minecraft-scientist/runs/natural-visual-detour-seed201-v6-audited --write
```

The most open heading (4) falls 10 blocks and is rejected. Heading 0 traverses
5.227 blocks to a novel E and returns to B within 0.134 blocks with no vertical
drop; the independent validation repeats the same physical result. Only then are
B→E and E→B persisted as resolved transitions and exposed by `plan_route`. The
independent auditor recomputes every temporal median and panorama from PNGs,
recognition ranks, action alignment, candidate selection, evaluator distances,
SQLite projections, source/evidence hashes and telemetry isolation; all 20 checks
are true.

This establishes a validated local detour, not a completed B→E→A route. Connecting
E to A, scaling the search beyond two preregistered open headings, and cross-seed or
cross-biome validation remain open.

The goal-frontier gate takes the next step from persistent E. Each candidate starts
in a fresh MineStudio session, records the current A panorama as goal memory,
replays the stored A→B→E route, and follows one visually open heading. Candidate
ranking uses only RGB distance to remembered A, visual novelty and deterministic
tie-breaking. Coordinates are still evaluator-only: after selection they may reject
a candidate that did not physically make progress, but cannot choose its direction.

Reproduce the accepted directed E→F expansion and independent audit with:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/natural_visual_goal_frontier_benchmark.py --source-run /mnt/d/minecraft-scientist/runs/natural-visual-detour-seed201-v6-audited --candidate-count 2 --moves 3 --move-frames 4 --allow-directed-frontier --temporal-samples 3 --output /mnt/d/minecraft-scientist/runs/natural-visual-goal-frontier-seed201-v10-audited
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_natural_visual_goal_frontier.py /mnt/d/minecraft-scientist/runs/natural-visual-goal-frontier-seed201-v10-audited --write
```

Heading 7 moves 4.426 blocks from E to novel F and reduces the evaluator-only
distance to A from 9.668 to 6.376 blocks. A separate session repeats both values.
Earlier return experiments showed that F→E is not a safe straight inverse, so the
accepted memory contains exactly one new resolved transition, E→F, and deliberately
contains no F→E edge. All runtime gates pass; the independent audit recomputes the
RGB selector, PNG temporal medians, action alignments, physical progress, directed
SQLite topology, source manifests and telemetry isolation, with all 21 checks true.

This is verified goal-directed frontier expansion and a more honest directed
cognitive map. It does not yet connect F to A or prove unrestricted exploration,
cross-seed/biome generalization, survival, or long-horizon recovery.

The general goal-search runner can now consume either that accepted frontier or its
own audited output. It retrieves an arbitrary-length route from SQLite and checks
every node against frozen visual memory during fresh-session replay. A saved goal
panorama remains immutable across rounds. The ranking module accepts only pixel
signatures and opaque place IDs; candidate physics are checked after the direction
has been selected. A novel node is fitted from the discovery and shadow-validation
views, then must be recognized during a third independent restart before its
directed transition is committed.

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/natural_visual_goal_search_benchmark.py --source-run /mnt/d/minecraft-scientist/runs/natural-visual-goal-frontier-seed201-v10-audited --candidate-count 4 --moves 3 --move-frames 4 --temporal-samples 3 --output /mnt/d/minecraft-scientist/runs/natural-visual-goal-search-seed201-v1
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_natural_visual_goal_search.py /mnt/d/minecraft-scientist/runs/natural-visual-goal-search-seed201-v1 --write
```

In the accepted first round, four isolated probes from F select heading 11 using
the persisted A goal panorama. F→G moves 6.942 blocks with no vertical drop and
reduces distance to A from 6.376 to 5.349 blocks in both discovery and shadow
validation. The final independent restart recognizes A/B/E/F/G and reaches G with
0.000 blocks of checkpoint error. SQLite retains seven places, seven transitions
and 60 replay-valid journal events. All 22 independent audit checks pass over 1,512
PNG files and 1,416 public transitions. Set `--source-run` to this audited v1 output
to continue from G with the same runner. G still has no validated route to A;
cross-seed/biome generalization and long-horizon survival remain open.

The second round (`runs/natural-visual-goal-search-seed201-v2`) tests that recursive
continuation from G. All four fresh sessions recognize the complete A/B/E/F/G
route, but the RGB ranking chooses heading 3 whose physical distance to A increases
from 5.349 to 5.366 blocks. The runner rejects it and exits nonzero; accepted memory
remains byte-identical to the source. It does not substitute a physically favorable
runner-up using evaluator coordinates. The rejection audit independently recomputes
the RGB ranking, all saved scans and physical metrics, checks the unchanged database
and reports all 17 checks true:

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/audit_visual_goal_search_rejection.py /mnt/d/minecraft-scientist/runs/natural-visual-goal-search-seed201-v2 --write
```

This exposes a remaining representation problem: panorama similarity is not a
reliable measure of geometric goal progress. Improving visual navigation evidence
and evaluating it on held-out trajectories takes priority over further claims of
goal-reaching capability. The accepted first round remains a bounded success;
the second round is explicitly rejected evidence.

A concrete label-free `GridVisualController` maps opaque screen regions and hotbar
slots to MineStudio env actions. Tests verify action translation and semantic
isolation; the accepted live experiment additionally audits successful opaque
`SELECT` events and held equipment. It removes HUD bands and grounds outcome
evidence to anonymous cells over the remaining world viewport instead of using a
fixture-supplied central target crop.

## Architecture and extension boundary

RGB geometry is now an explicitly experimental optional branch, not a replacement
for the accepted topological planner. Its conservative bootstrap rejects unsupported
poses, and an independent evaluator measures disjoint trajectory error rather than
equating more tracked frames with better navigation. A fresh 448-pixel route replay
with reference-resolution normalization passes 24 audit checks and preserves the
accepted memory; monocular tracking remains unachieved on that night/thunder route.
See `docs/visual-geometry-development.md` for positive/negative evidence and
`docs/quality-roadmap.md` for the full remaining project gates.

Local LoFTR correspondence, geometric sensitivity and bidirectional pixel
refinement are also evaluated as optional development tools. Independent audits
initially found only 3/8 correct movement estimates at 224 pixels and 1/8 at 448; conservative
checks reject unstable motion rather than promoting it into the planner. This is
a documented negative result, not working visual navigation. Runtime, checkpoint
identity, reproduction and evidence are in `docs/dense-correspondence-development.md`.
No API key or remote inference is used.

The subsequent public camera-resize repair corrects unequal focal axes and pixel
centers: dense matching now scores 3/8 movements at both resolutions, still with
wrong supported poses. Continuous identity-preserving RGB tracking and held-out
intermediate-frame geometry are implemented and independently replayed, but no
movement passes full reliability validation. These are guarded development
results, not working open-world navigation. See
`docs/temporal-motion-development.md` for corrected evidence and reproduction.

A fixed 2×2 noon/midnight × clear/thunder acquisition now supplies 804 consecutive
RGB frames with identical evaluator trajectories across all four natural sessions.
Independent replay finds 2/8 and 1/8 correct fully validated post-WAIT motions in
clear scenes, versus zero in thunder; all moving-endpoint estimates still fail.
This localizes a weather-related correspondence problem without proving robust
navigation. See `docs/controlled-motion-development.md` for the full trial/control
counts, replay evidence and remaining clear-scene coverage problem.

Past-context background selection, conditioned consensus refinement and an
explicit private view-bobbing comparison are now independently audited. None
establishes reliable navigation: moving-endpoint coverage remains zero, and the
bobbing pair scores 2/8 versus 1/8 fully validated settled movements. The shared
SDK runtime is untouched. See `docs/stationary-consensus-motion-development.md`.

An explicit plane/essential motion-competition variant now retains competing
homography branches, selects using independent off-plane rays and allows shadow
validation only to veto. Both clear and thunder development runs pass replay
audits but recover zero fully validated moving/settled motions, so the variant is
not promoted. See `docs/plane-motion-development.md`; positive navigation coverage
remains the unresolved requirement, not merely more green audit checks.

Chronological RGB-prefix selection now retains every trial and latches the first
supported local pose without private or future-based replacement. A fresh fixed
6/10-tick acquisition scores 3/8 correct settled **local** intervals versus 2/8
terminal full-window poses; moving intervals are 1/8 for both. The extra local pose
does not recover the final displacement, and most movements still fail. No memory
promotion or reliable navigation claim follows. Source-bound capture, all positive
and negative results, reproduction and remaining gates are documented in
`docs/keyframe-motion-development.md`.

Fresh distinct-handle two-leg acquisitions retain 450 RGB frames across the two
existing development worlds and independently replay all 328 prefix fits. Seed
200 supplies four correct combined-motion **local** intervals in eight trials,
but seed 201 supplies none; a seed-200 leg also passes internal geometry with a
26.18-degree direction error. The method is not promoted into navigation. Full
denominators, stationary controls, duplicate intervals and the retained failure
are in `docs/compound-motion-development.md`.

The scientific core has a portable offline release verifier:

```powershell
uv run python scripts/verify_core_release.py --seed 0 --seeds 30 --output runs/new-core-release
```

Windows and WSL each pass thirty worlds with two complete repeats and eighteen
identical artifacts per seed. Symbol-transfer probes now execute the actually
learned primitive and independently replay saved actions, preserving the seed-zero
3-versus-11 intervention result. Equation validation/crisis experiment design uses
only public observations. See `docs/core-release-verification.md` for boundaries
and reproduction; these results do not complete open-world scientific exploration.

`.github/workflows/core-ci.yml` defines locked Python 3.11 Windows/Linux default
and geometry test/build/release jobs. Its YAML is locally checked, but no Git
remote or hosted run exists yet; immutable action pins and hosted validation remain
open. Local release verification must not be presented as remote CI execution.

```powershell
uv run --extra geometry pytest
```

See `docs/architecture.md`. Mineflayer and structured simulators implement the same
agent-visible contract and never expose benchmark truth. The live observation
command expects an existing server; server download and EULA acceptance remain an
explicit user-controlled step.

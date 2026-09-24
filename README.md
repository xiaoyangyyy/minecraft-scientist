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

A concrete label-free `GridVisualController` maps opaque screen regions and hotbar
slots to MineStudio env actions. Tests verify action translation and semantic
isolation; the accepted live experiment additionally audits successful opaque
`SELECT` events and held equipment. It removes HUD bands and grounds outcome
evidence to anonymous cells over the remaining world viewport instead of using a
fixture-supplied central target crop.

## Architecture and extension boundary

See `docs/architecture.md`. Mineflayer and structured simulators implement the same
agent-visible contract and never expose benchmark truth. The live observation
command expects an existing server; server download and EULA acceptance remain an
explicit user-controlled step.

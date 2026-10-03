# Offline scientific-core release verification

`scripts/verify_core_release.py` supplies a portable, default-offline release
entry point for four-form structured discovery. It does not require Java,
MineStudio, an API key, OpenCV or an external model. A green result is explicitly
**not** completion of the visual/open-world project.

## Stronger evidence boundaries

The full equation benchmark now selects independent validation experiments and
crisis probes from public observations, not from evaluator object/height lists.
These lists previously repeated publicly visible information, rather than hidden
law parameters, but the boundary still needed correction. A regression test
substitutes an evaluator that raises on every access during experiment design;
separate real tests retain privileged accuracy scoring.

The MVP symbol-transfer probe previously used a boolean to select a fixed recipe.
It now takes the actual accepted, independently validated primitive and compiles
its learned body in order. Renaming a primitive does not change behavior; an
incorrect body is executed and fails instead of being replaced by a successful
recipe. Unknown operations, missing validation and candidate status reject before
world creation. Noncompliant success cannot become reuse evidence.

Each MVP run additionally preserves `transfer-evidence.json`,
`library-transfer-events.jsonl` and `no-library-transfer-events.jsonl`. They bind
the learned definition to real controlled actions in the same independent world.
The release verifier reconstructs that world through public observations,
reexecutes both programs/search policies and compares all saved transitions and
terminal outputs. It never substitutes success based on a symbol-library flag.

This comparison measures learned executable program reuse versus scratch protocol
search. It does not prove superiority to a fixed skill library carrying the same
program, or replace the separately scoped planner/component/LLM comparisons.

## Release contract

Before execution the verifier freezes all package Python sources, itself,
`pyproject.toml` and `uv.lock`, plus seed list, budget and two repetitions per seed.
Existing output directories are always refused. Each seed must independently:

- recover tool rules, machine protocols, transformation programs and an equation;
- pass equation shadow validation, unchanged-law transfer and crisis replacement;
- persist superseded/accepted tool and equation histories with real counterexamples;
- retain zero symbols in the no-library condition and positive compression/transfer
  savings in the library condition;
- independently replay the executable symbol transfer and its scratch control;
- validate schema-correct event/theory/symbol artifacts and replay the experience
  database without writing;
- reproduce all eighteen artifact byte hashes in a second full run.

The database must be closed and checkpointed, without pending WAL/SHM files,
before immutable read-only replay. The first development smoke used an ordinary
read-only SQLite connection, which created auxiliary WAL files and correctly
failed the exact artifact-set check. That failed directory is retained at
`runs/core-release-smoke-seed0-v1`; the verifier now avoids that mutation rather
than ignoring the files. Tests also reject pending WAL, forged theory ancestry,
unlearned primitive substitution, forged transfer events, changed repeat bytes,
invalid budgets and incomplete runs. A failure preserves partial evidence and
exits nonzero, not a smaller passing seed subset.

## Current results (2026-10-03)

`runs/core-release-seeds0-29-windows-v25/report.json` on Windows and
`runs/core-release-seeds0-29-wsl-v25/report.json` on WSL each pass all six release
checks for all thirty seeds. Each platform runs sixty complete benchmarks and
reproduces eighteen artifacts per seed, including the transfer traces and
experience database. All four learned-law accuracies, revised-law recovery and
task utility are 1.0 for these finite candidate-language worlds. Every seed learns
three symbols; reuse ranges from eleven to seventeen. Equation transfer saves
one experiment in every seed. The mean **reported** benchmark experiment count
is 53.9333; it is not the total cost of auxiliary transfer/revision/replay runs.

The v1 thirty-seed releases precede the executable-symbol transfer correction and
contain fifteen artifacts per repetition. The v2 eighteen-artifact runs precede
the added partial-output refusal checks. The v3 releases precede the addition of
the optional plane-motion source; v4 freezes that source, while v5 additionally
freezes the optional chronological keyframe policy, and v6 adds the public compound
acquisition schedule. Version v7 adds the optional subsequent-image veto; v8 adds
public provisional action selection and conditional parallax diagnosis. V9 adds
the optional bounded competing-map diagnostic and weak-anchor resampling. V10
adds original-map replay and the optional direction-constrained map search. V11
adds optional frozen-map action-switch validation. V12 adds a causal visibility
controller; V13 repairs its anchor-denominator definition, and V14 separates
second-phase new loss from original-anchor loss. V15 adds online fitting and
protection of the first supported map's immutable landmark partitions. V16 adds
optional disjoint per-ID tracking-loss accounting, with exact historical public
prefix replay and unchanged tracker output. V17 adds optional completed-second-leg
freezing and bounded different-control validation. V18 adds training-only joint
multi-view pose/landmark optimization with disjoint validation and frozen-map
future tests. V19 binds image-camera evaluation to renderer snapshots; V20 adds
chronological causal map windows, V21 dispatches first-eligible acquisition live,
V22 adds past-support switching, and V23 adds public frozen-map MOVE/WAIT settling.
V23 freezes the preceding 105 entries including 102 package files. V24 adds
public-image stability recovery and explicit fresh local-map epochs, freezing
106 entries including 103 package files. V25 adds bounded rolling maintenance,
explicit epoch journals and alternative opaque controls, freezing 107 entries
including 104 package files. Both V25 current/source-snapshot ledgers are rechecked
with zero mismatches. Core behavior is unchanged by these
optional modules; visual results remain separately scoped in
`docs/rolling-map-maintenance-development.md`.
These earlier runs are historical checks, not the current source-frozen release.
Neither local multi-seed verification
proves remote CI execution, fresh live-server gates, learned-planner release
comparisons, broad visual generalization or integrated long-horizon exploration.

The V25 full frozen regressions pass **1075 tests/two skips on Windows** and
**1077/two on WSL**, `runs/rolling-map-full-windows-v1.xml` and
`runs/rolling-map-full-wsl-v2.xml`. The isolated default-only WSL installation
passes **593/70 optional or platform skips**; the focused rolling/recovery/auditor
Windows profile passes 47 cases. Original MVP bytes, builds and changed-file
lint/format/whitespace checks pass. A failed initial WSL suite is retained, not
counted as a release pass: its temporary root disappeared during concurrent pytest
jobs. The unchanged rerun uses a verified-unused dedicated `--basetemp` and passes
the complete suite. See the rolling-map document for failure and identity details.

The historical adaptive-map audit checkpoint regression passes **718 tests/two
skips on Windows**, **720/two on WSL**, and **401/60 optional or platform skips**
in a clean default-only installation.
Targeted Ruff and both package builds pass. Default-only checks have no optional
visual/MineStudio/OpenAI runtime and still exercise the complete release verifier.
Those counts include the capture-wrapper tests and sixteen new adaptive-control/
map pipeline tests. Fresh policy/map replay now passes, but its retained direction
error prevents navigation promotion; see `docs/visibility-map-validation-development.md`.
Default-only checks pass offline after populating the locked dependency cache;
the initial isolated offline setup failed on a missing NumPy wheel. Installation
downloads are separate from the algorithm's offline/no-model runtime contract.

The subsequent decorator-source locator correction affects experiment scripts,
not the then-96-entry package/core ledger. V10 is historical after adding the
action-switch module. V11 matches the prior 97-entry checkpoint; V13 freezes the
prior 98-entry core after anchor-denominator refinement. V14 freezes the historical
phase-relative controller; V15 freezes the prior online-landmark checkpoint and
V16 freezes the optional loss tracing. Visual
inference/plan/diagnostic manifests have a separate historical
boundary after that correction, detailed in `docs/source-ledger-development.md`.
The scientific-core verifier does not substitute for their source checks.

The phase-relative V14 checkpoint passes **725/two skips on Windows**,
**727/two on WSL**, and **408/60 in the isolated default-only offline installation**.
All 98 V14 source entries were rechecked after execution on both platforms, with
zero mismatches. Targeted Ruff/format checks and the package build pass. Fresh
adaptive capture and independent map auditing complete, but two direction errors
remain; see `docs/phase-visibility-development.md`. These results are not reliable
navigation or completion of the broad project requirements.

The historical online-landmark V15 checkpoint passes **749/two skips on Windows**,
**751/two on WSL**, and **427/61 in the isolated default-only offline installation**.
Both 99-entry core ledgers matched that checkpoint's bytes, with no mismatch.
The optional acquisition change does not establish reliable navigation; see
`docs/online-landmark-acquisition-development.md` for fresh real-world evidence.

The V16 loss-tracing checkpoint passes **785/two skips on Windows**,
**787/two on WSL**, and **427/63 in the isolated default-only offline installation**.
Both 99-entry core ledgers matched that historical package/verifier/configuration checkpoint.
The later script-only in-memory historical loader refinement passes 45 targeted
tests, including input-tree immutability and forged bytecode refusal; the full
suites preceded that final script refinement. See
`docs/landmark-track-loss-development.md` for source boundaries and remaining work.

The V17 checkpoint passes **811 tests/two skips on Windows**, **813/two on WSL**,
and **443/64 optional or platform skips** in the isolated default-only offline
installation. Both 100-entry ledgers matched that checkpoint's bytes. Forty-seven focused
second-leg acquisition/map tests also pass. The independent fresh real-image audit
rejects the only second-leg candidate; these engineering checks do not establish
navigation. See `docs/second-leg-shadow-development.md`.

The V18 joint-map checkpoint freezes 101 source entries and passes both thirty-seed
release gates. Forty-seven new joint-module/diagnostic tests pass; full Windows
regression passes **858 tests/two skips**, WSL **860/two**, and isolated default-only
offline **443/66**. The real development replay improves one matched interval but
retains an incorrect short future direction despite consistent withheld pixels;
see `docs/joint-multiview-map-development.md`. Ruff/format checks and both builds
pass. Earlier V17 evidence remains historical after adding the new optional module.

The current seed-zero MVP report at
`runs/rolling-map-regression-mvp-seed0/benchmark.json` remains byte-identical
to the earlier regression: SHA-256
`B95AF2E5C152B9F06EDC63004EA87FB183C37E6A5A13BFF53F21A39A4F48AEB8`.
It retains 47 experiments, accuracy/task success 1.0, detection delay one, eight
repairs, three symbols/thirteen reuses and transfer **three versus eleven**
interventions. The new transfer evidence verifies the last comparison through
actual primitive execution rather than changing these legacy metric fields.

## CI definition versus executed evidence

`.github/workflows/core-ci.yml` now defines four Python 3.11 jobs: Windows/Linux,
each with default dependencies or the optional geometry extra. Each uses the lock,
runs the complete test suite, the thirty-seed/two-repeat core verifier, the original
MVP command and both package builds, retaining partial evidence even on failure.
Repository access is read-only and checkout credentials are not persisted.

The workflow YAML, triggers, matrix and permission declaration are checked locally;
its acceptance commands have current Windows/WSL evidence. **No Git remote is
configured and no hosted CI job has executed.** Public GitHub metadata access
timed out during setup, so the workflow currently uses conventional major action
refs; immutable SHA pinning and hosted-run validation remain publication gates.
Do not call this a verified remote CI release or treat local WSL as a hosted Linux
runner. Java/MineStudio/train extras and live/visual research gates are deliberately
not substituted by these offline jobs.

## Commands

From the repository root:

```powershell
uv run pytest
uv run python scripts/verify_core_release.py --seed 0 --seeds 30 --output runs/new-core-release
uv run minecraft-scientist benchmark --suite mvp --seed 0 --output runs/new-mvp
uv build
```

The same Python entry point runs in the dedicated WSL environment. Each run freezes
its executing source bytes; within-platform exact-repeat claims do not imply that
Windows CRLF output and Linux LF output have identical byte hashes. Core tests run
on both platforms and public scientific results agree, while runtime identity is
recorded separately. Use frozen historical source snapshots for old artifact
replay; current-source checks intentionally fail after substantive source changes.

# Lossless expanded development campaigns

Storage/orchestration adapters, not a new learner, mutation validator or reduced
benchmark. Full contract: `docs/compact-development-design.md`. They reuse the
unchanged four paid learners, public-first audits, raw scorers and challenger
policy; configs, budgets and every event/candidate/failure remain intact.

Canonical JSON is stored in deterministic-header gzip. Both stored-byte and
logical-payload hashes are checked, with bounded decompression, canonical JSON,
exact file coverage and lazy pinned private references. The logical payload
hash is the cross-platform scientific replay identity; zlib versions may encode
the same payload differently. Nothing is truncated or sampled to save space.
The revised writer streams canonical bytes in 64 KiB chunks and rejects final
certification after any failed write. Epoch view ownership is weakly cached to
avoid retaining completed worlds in the callback registry. These repairs change
source identities, not scientific budgets or mechanisms; compatibility must be
rechecked against the archived earlier source and complete logical payloads.

The structural verifier streams all public runs before any private content,
rather than holding the whole candidate corpus in RAM. The validation adapter
freezes all public learning/Wp prefixes, then really replays/pays them in fresh
worlds and replaces only the last three validation slots. Source scores/report
never select cases. No-anomaly and other ineligible prefixes are explicitly
categorized alongside the original ablation status, not relabeled successes.

Default selects ALL registered development groups. A `--max-groups 1` run is
only a compatibility probe. Sealed tuning/evaluation fixtures are never opened.

```powershell
uv run pytest research/compact_campaigns research/challenger_validation
uv run python scripts/run_compact_structural_development.py --corpus <pinned-corpus> --manifest-sha256 <certified-sha256> --output <new-directory>
uv run python scripts/run_compact_validation_development.py --source <completed-compact-campaign> --source-manifest-sha256 <verified-sha256> --corpus <pinned-corpus> --corpus-manifest-sha256 <certified-sha256> --output <new-directory>
```

Use `--verify-only` with the same inputs/output to independently replay and
rescore a completed run. Interrupted output is retained without a final
manifest; never overwrite/restart it merely because an observation timed out.
More development groups do not constitute sealed generalization, a formal
sample/statistics design, full strong-baseline fidelity, unknown control/noise,
H4, Minecraft/RGB, recovery, release or proof of novelty.

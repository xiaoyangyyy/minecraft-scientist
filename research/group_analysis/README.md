# Paired leakage-group development analysis

Development-only diagnostics after complete independent public-first/raw audits,
not a statistical significance test or formal method preregistration. Contract:
`docs/group-analysis-design.md`. This package never supplies evidence or private
metadata to a learner and changes no existing frozen mechanism/source file.

It retains every fixture/method/policy including wrong, unresolved, ineligible
and bounded-no-anomaly outcomes. Fixture totals and fixture-weighted summaries
are separate from equal-weight leakage-group means and paired group contrasts.
Case, H-change, renaming, public-prior behavioral coverage and construction
one-edit-family strata are all retained, including empty cells.
One group gives no between-group uncertainty interval. Exploratory resampling
assumes independent/exchangeable groups; grouping does not prove that premise.
No automatic superiority, novelty, causal uniqueness or full-project claim.

The runner always invokes real complete input verifiers, checks input/source pins
again after analysis and never overwrites an existing output. Runtime artifacts
are explicitly nondeterministic and exclude missing total scientific compute.

```powershell
uv run pytest research/group_analysis -q
uv run python scripts/analyze_development_groups.py --source <complete-source> --source-manifest-sha256 <pin> --corpus <certified-corpus> --corpus-manifest-sha256 <pin> --output <new-directory>
```

Add both `--validation-source` and `--validation-manifest-sha256` to analyze the
complete paired validation ablation alongside its structural discovery source.
Partial directories, missing methods/policies, group drift and unequal paid
prefixes are rejected, not fixed by silently filtering rows.

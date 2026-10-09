"""Post-hoc raw counterexamples for VERIFIED development promotions only."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.structural_campaigns.corpus import digest
from research.structural_holdouts.raw_tables import compose, trace
from research.validation_failures.witness import minimum_witness


def sources():
    paths = (
        list((ROOT / "research" / "validation_failures").glob("*.py"))
        + list((ROOT / "research" / "validation_failures").glob("*.md"))
        + list((ROOT / "research" / "structural_holdouts").glob("*.py"))
        + list((ROOT / "research" / "structural_holdouts").glob("*.md"))
        + [
            ROOT / "research" / "structural_campaigns" / "__init__.py",
            ROOT / "research" / "structural_campaigns" / "corpus.py",
            Path(__file__).resolve(),
        ]
    )
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def analyze(*, campaign, verification, output):
    campaign, verification, output = map(Path, (campaign, verification, output))
    proof = json.loads(verification.read_bytes())
    certified = proof["development_run"]
    if (
        proof["schema"]
        != "structural-campaign-development-cross-platform-verification-v1"
        or certified["state"] != "terminal"
        or certified["exit_code"] != 0
        or certified["full_audit"]["exit_code"] != 0
        or not certified["full_audit"]["public_before_private"]
        or not certified["full_audit"]["independent_raw_score"]
        or digest(campaign / "manifest.json") != certified["manifest_sha256"]
    ):
        raise ValueError("analysis requires completed pinned public/raw audit")
    manifest = json.loads((campaign / "manifest.json").read_bytes())
    for relative, expected in manifest["sha256"].items():
        path = campaign / relative
        if (
            not path.resolve().is_relative_to(campaign.resolve())
            or digest(path) != expected
        ):
            raise ValueError("verified campaign bytes drifted")
    registration = json.loads((campaign / "registration.json").read_bytes())
    report = json.loads((campaign / "report.json").read_bytes())
    if (
        registration["selection"]["split"] != "development"
        or report["registration"] != registration
    ):
        raise ValueError("analysis cannot consume sealed evaluation")
    source_hashes = sources()
    rows = []
    for row in report["rows"]:
        if row["raw_score"]["promoted_correct_at_H"] is not False:
            continue
        fid, method = row["fixture_id"], row["method"]
        if (
            fid not in registration["selection"]["fixture_ids"]
            or method not in registration["methods"]
        ):
            raise ValueError("unregistered failure row")
        directory = campaign / fid / method
        config = json.loads((directory / "config.json").read_bytes())
        result = json.loads((directory / "run.json").read_bytes())
        truth = json.loads((directory / "evaluator/world.json").read_bytes())
        repair = result["new"]["run"]
        candidate = repair["candidate_ir"]["parameters"]
        reference = compose(**truth["after"])
        witness = minimum_witness(
            candidate, reference, config["costs"], horizon=config["protocol"]["horizon"]
        )
        if witness is None:
            raise ValueError(
                "claimed wrong promotion has no bounded raw counterexample"
            )
        ids = set(repair["validation_query_ids"])
        plans = [
            event
            for event in repair["events"]
            if event["kind"] == "plan" and event["query_id"] in ids
        ]
        if len(plans) != len(ids):
            raise ValueError("validation plan coverage mismatch")
        passes = [
            trace(candidate, tuple(p["actions"]))
            == trace(reference, tuple(p["actions"]))
            for p in plans
        ]
        if not all(passes):
            raise ValueError(
                "wrong promotion contains a contradicting final validation word"
            )
        rows.append(
            {
                "fixture_id": fid,
                "method": method,
                "case": row["case"],
                "independent_group": row["independent_group"],
                "witness": witness,
                "final_validation_words": len(plans),
                "all_final_words_agree_with_wrong_candidate": all(passes),
                "witness_was_a_final_word": any(
                    p["actions"] == witness["actions"] for p in plans
                ),
                "actual_spent": result["spent"],
                "remaining_total_budget": result["budget"] - result["spent"],
                "oracle_witness_affordable": witness["reset_inclusive_cost"]
                <= result["budget"] - result["spent"],
                "limitation": "post-hoc truth-assisted witness, NOT an action available to Agent or a fair oracle baseline; grammar mismatch/rare-word coverage can coexist",
            }
        )
    if sources() != source_hashes:
        raise RuntimeError("diagnostic source drift")
    output.mkdir(parents=True, exist_ok=False)
    diagnostic = {
        "schema": "structural-promotion-failure-analysis-v1",
        "source_sha256": source_hashes,
        "campaign_manifest_sha256": certified["manifest_sha256"],
        "verification_sha256": digest(verification),
        "rows": rows,
        "wrong_promotions": len(rows),
        "independent_groups": len({r["independent_group"] for r in rows}),
        "scope": "development-only diagnostic oracle; no fix, prospective planner benefit or statistical/innovation claim",
    }
    (output / "analysis.json").write_text(
        json.dumps(diagnostic, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "wrong_promotions": len(rows),
                "witness_costs": [r["witness"]["reset_inclusive_cost"] for r in rows],
                "all_oracle_affordable": all(
                    r["oracle_witness_affordable"] for r in rows
                ),
            },
            sort_keys=True,
        )
    )
    return diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "verification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    analyze(**vars(parser.parse_args()))

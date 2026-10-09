"""Dataset construction audit, separate from Agent learning/private scoring."""

import hashlib
import itertools
import json
import re
from copy import deepcopy
from pathlib import Path

from .generator import CASES, describe, fixture
from .identity import canonical_json, partition_records
from .raw_tables import compose, validate


def validate_public(public):
    if type(public) is not dict or set(public) != {
        "actions",
        "outputs",
        "costs",
        "initial_family",
    }:
        raise ValueError("private/extra fields in public entry")
    for key, count in (("actions", 5), ("outputs", 2)):
        values = public[key]
        if (
            type(values) is not list
            or len(values) != count
            or any(
                type(v) is not str or re.fullmatch(r"id-[0-9a-f]{16}", v) is None
                for v in values
            )
            or len(set(values)) != count
        ):
            raise ValueError("invalid public opaque alphabet")
    if (
        type(public["costs"]) is not dict
        or set(public["costs"]) != set(public["actions"])
        or any(
            type(v) is not int or not 1 <= v <= 100 for v in public["costs"].values()
        )
    ):
        raise ValueError("invalid public costs")
    family = public["initial_family"]
    if (
        type(family) is not dict
        or set(family)
        != {"hypotheses", "calibration", "discovery_budget", "shadow_count"}
        or type(family["hypotheses"]) is not list
        or len(family["hypotheses"]) != 9
        or type(family["calibration"]) is not list
        or len(family["calibration"]) != 4
        or any(
            type(a) is not str or a not in public["actions"]
            for a in family["calibration"]
        )
        or type(family["discovery_budget"]) is not int
        or family["discovery_budget"] != 80
        or type(family["shadow_count"]) is not int
        or family["shadow_count"] != 3
    ):
        raise ValueError("invalid public family protocol")
    factors, pairs = {}, set()
    for h in family["hypotheses"]:
        if type(h) is not dict or set(h) != {
            "mechanism_id",
            "controller_id",
            "mechanism_ir",
            "controller_ir",
        }:
            raise ValueError("invalid public joint hypothesis")
        pair = h["mechanism_id"], h["controller_id"]
        if (
            any(
                type(v) is not str or re.fullmatch(r"id-[0-9a-f]{16}", v) is None
                for v in pair
            )
            or pair in pairs
        ):
            raise ValueError("duplicate/nonopaque public factor identity")
        pairs.add(pair)
        for role in ("mechanism", "controller"):
            ir = h[role + "_ir"]
            if (
                type(ir) is not dict
                or set(ir)
                != {
                    "mechanism",
                    "representation",
                    "scope",
                    "preconditions",
                    "latent_variables",
                    "effects",
                    "parameters",
                    "composition",
                }
                or ir["mechanism"] != "finite_transducer"
                or ir["representation"] != "state_machine"
                or ir["scope"]
                != {"language": "total-mealy-v1", "reset": "initial-state"}
                or ir["preconditions"] != []
                or ir["latent_variables"] != ["machine_state"]
                or ir["effects"] != {"output": "transition.output"}
                or ir["composition"] != []
            ):
                raise ValueError("unregistered public executable IR")
            raw = ir["parameters"]
            validate(raw)
            if (role == "controller" and raw["actions"] != public["actions"]) or (
                role == "mechanism"
                and (
                    not set(raw["actions"]).issubset(public["actions"])
                    or raw["outputs"] != public["outputs"]
                )
            ):
                raise ValueError("public factor/alphabet disagreement")
            key, body = (role, h[role + "_id"]), canonical_json(ir)
            if key in factors and factors[key] != body:
                raise ValueError("public factor ID redefinition")
            factors[key] = body
        compose(h["mechanism_ir"]["parameters"], h["controller_ir"]["parameters"])
    if (
        sum(k[0] == "mechanism" for k in factors) != 3
        or sum(k[0] == "controller" for k in factors) != 3
    ):
        raise ValueError("incomplete public family cross product")


def validate_fixture(public, private, metadata):
    validate_public(public)
    if (
        type(metadata) is not dict
        or set(metadata)
        != {"schema", "fixture_id", "inputs", "old_selection", "edits", "diagnostics"}
        or metadata["schema"] != "generated-structure-metadata-v1"
    ):
        raise ValueError("invalid evaluator metadata")
    if (
        type(private) is not dict
        or set(private) != {"schema", "boundary_episode", "before", "after"}
        or private["schema"] != "online-revision-private-v1"
        or type(private["boundary_episode"]) is not int
    ):
        raise ValueError("invalid generated private world")
    for epoch in ("before", "after"):
        if type(private[epoch]) is not dict or set(private[epoch]) != {
            "mechanism",
            "controller",
        }:
            raise ValueError("invalid private factors")
        compose(**private[epoch])
    actual = describe(public, private, metadata["edits"])
    if canonical_json(actual) != canonical_json(metadata["diagnostics"]):
        raise ValueError("raw identities/diagnostics disagree")
    generated = fixture(**metadata["inputs"])
    if canonical_json(generated) != canonical_json(
        {"public": public, "private": private, "metadata": metadata}
    ):
        raise ValueError("declared new-graph generation policy drift")
    return actual


def leakage_partition(metadata, *, salt):
    # Before-prior AND target identities share namespaces. Evaluation targets
    # therefore cannot equal development/tuning priors, even after renaming.
    return partition_records(
        tuple(
            {
                "id": m["fixture_id"],
                "leakage_keys": tuple(m["diagnostics"]["leakage_keys"]),
            }
            for m in metadata
        ),
        salt=salt,
    )


def summarize(metadata, partition):
    result = {}
    for split in ("development", "tuning", "evaluation"):
        rows = [
            m for m in metadata if partition["assignments"][m["fixture_id"]] == split
        ]
        result[split] = {
            "fixtures": len(rows),
            "independent_groups": len(
                {partition["groups"][m["fixture_id"]] for m in rows}
            ),
            "base_graphs": len({m["inputs"]["base_index"] for m in rows}),
            "behavior_change_at_H": sum(
                m["diagnostics"]["behavior_change_at_H"] for m in rows
            ),
            "behavior_change_unbounded": sum(
                m["diagnostics"]["behavior_change_unbounded"] for m in rows
            ),
            "target_exact_in_public_family": sum(
                m["diagnostics"]["target_exact_in_public_family"] for m in rows
            ),
            "outside_declared_mechanism_edit_family": sum(
                not m["diagnostics"]["within_declared_mechanism_edit_family"]
                for m in rows
            ),
            "cases": {
                case: sum(m["inputs"]["case"] == case for m in rows) for case in CASES
            },
        }
    return result


def audit_dataset(directory, *, current_sources=True):
    directory = Path(directory)
    registration = json.loads(
        (directory / "registration.json").read_text(encoding="utf-8")
    )
    if (
        type(registration) is not dict
        or set(registration)
        != {
            "schema",
            "config",
            "fixture_inputs",
            "source_sha256",
            "leakage_policy",
            "scope",
        }
        or registration["schema"] != "new-structure-corpus-registration-v1"
    ):
        raise ValueError("unexpected dataset registration")
    config = registration["config"]
    if (
        type(config) is not dict
        or set(config)
        != {"seed", "base_count", "renamings", "states", "boundary", "salt"}
        or type(config["seed"]) is not int
        or config["seed"] < 0
        or type(config["base_count"]) is not int
        or not 1 <= config["base_count"] <= 512
        or type(config["renamings"]) is not int
        or not 1 <= config["renamings"] <= 4
        or type(config["states"]) is not list
        or not config["states"]
        or any(type(s) is not int or not 2 <= s <= 5 for s in config["states"])
        or len(set(config["states"])) != len(config["states"])
        or type(config["boundary"]) is not int
        or not 0 <= config["boundary"] <= 1000
        or type(config["salt"]) is not str
        or not 1 <= len(config["salt"]) <= 256
    ):
        raise ValueError("invalid frozen dataset configuration")
    expected_inputs = [
        {
            "seed": config["seed"],
            "base_index": i,
            "states": config["states"][i % len(config["states"])],
            "case": case,
            "renaming_index": r,
            "boundary": config["boundary"],
        }
        for i, case, r in itertools.product(
            range(config["base_count"]), CASES, range(config["renamings"])
        )
    ]
    if canonical_json(registration["fixture_inputs"]) != canonical_json(
        expected_inputs
    ):
        raise ValueError("dataset private grid omission/reorder/selection")
    if (
        registration["leakage_policy"]
        != "cost-colored target and all supplied-before-prior structure/behavior share namespaces; before/after pair structure/behavior also grouped. Transitive closure, not seed-based splitting"
    ):
        raise ValueError("unregistered leakage policy")
    records, public_entries = [], {}
    fixture_ids = []
    for inputs in expected_inputs:
        fixture_id = hashlib.sha256(canonical_json(inputs).encode()).hexdigest()[:24]
        if fixture_id in public_entries:
            raise ValueError("duplicate dataset fixture")
        public = json.loads(
            (directory / fixture_id / "public.json").read_text(encoding="utf-8")
        )
        validate_public(public)  # All public entries checked BEFORE private I/O.
        fixture_ids.append(fixture_id)
        public_entries[fixture_id] = public
    for fixture_id in fixture_ids:
        metadata = json.loads(
            (directory / fixture_id / "metadata.json").read_text(encoding="utf-8")
        )
        private = json.loads(
            (directory / fixture_id / "evaluator" / "world.json").read_text(
                encoding="utf-8"
            )
        )
        if metadata["fixture_id"] != fixture_id:
            raise ValueError("fixture path/identity disagreement")
        validate_fixture(public_entries[fixture_id], private, metadata)
        records.append(metadata)
    partition = json.loads((directory / "partitions.json").read_text(encoding="utf-8"))
    expected = leakage_partition(records, salt=registration["config"]["salt"])
    if canonical_json(partition) != canonical_json(expected):
        raise ValueError("leakage-connected partition mismatch")
    # Independently check that no actual shared key crosses splits, including
    # transitive components. Check canonical bodies on digest reuse as well.
    owners, bodies = {}, {}
    for record in records:
        split = partition["assignments"][record["fixture_id"]]
        for key in record["diagnostics"]["leakage_keys"]:
            if key in owners and owners[key] != split:
                raise ValueError("cross-split prior/target/pair leakage")
            owners[key] = split
        values = deepcopy(record["diagnostics"]["prior_identity_bodies"])
        for name, identity in record["diagnostics"]["identities"].items():
            key = name.replace("target-", "") + ":" + identity["sha256"]
            values[key] = identity["canonical"]
        for key, body in values.items():
            if hashlib.sha256(body.encode()).hexdigest() != key.rsplit(":", 1)[1] or (
                key in bodies and bodies[key] != body
            ):
                raise ValueError("canonical body/digest collision or forgery")
            bodies[key] = body
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    summary = summarize(records, partition)
    if (
        report["schema"] != "new-structure-corpus-report-v1"
        or report["summary"] != summary
        or report["fixture_count"] != len(records)
        or report["independent_group_count"] != partition["independent_group_count"]
    ):
        raise ValueError("dataset report denominators disagree")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_files = {"registration.json", "partitions.json", "report.json"}
    expected_files.update(
        f"{fid}/{name}"
        for fid in fixture_ids
        for name in ("public.json", "metadata.json", "evaluator/world.json")
    )
    if (
        manifest["schema"] != "new-structure-corpus-manifest-v1"
        or set(manifest["sha256"]) != expected_files
        or manifest["nondeterministic_artifacts"] != ["timing.json"]
    ):
        raise ValueError("dataset manifest coverage mismatch")
    actual_files = {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }
    if actual_files != expected_files | {"manifest.json", "timing.json"}:
        raise ValueError("missing/extra dataset files")
    for relative, expected_hash in manifest["sha256"].items():
        if (
            hashlib.sha256((directory / relative).read_bytes()).hexdigest()
            != expected_hash
        ):
            raise ValueError("dataset byte drift")
    if current_sources:
        root = Path(__file__).resolve().parents[2]
        source_files = (
            list((root / "research" / "structural_holdouts").glob("*.py"))
            + list((root / "research" / "structural_holdouts").glob("*.md"))
            + [root / "scripts" / "build_structural_corpus.py"]
        )
        if type(registration["source_sha256"]) is not dict or set(
            registration["source_sha256"]
        ) != {p.relative_to(root).as_posix() for p in source_files}:
            raise ValueError("dataset source inventory changed or omitted")
        for relative, expected_hash in registration["source_sha256"].items():
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or hashlib.sha256((root / relative).read_bytes()).hexdigest()
                != expected_hash
            ):
                raise ValueError("dataset source drift")
    return {
        "schema": "new-structure-corpus-audit-v1",
        "valid": True,
        "fixture_count": len(records),
        "independent_group_count": partition["independent_group_count"],
        "summary": summary,
        "deterministic_artifacts": len(expected_files),
        "scope": "dataset construction/leakage only; no method evaluation, unique causal recovery or formal statistical result",
    }

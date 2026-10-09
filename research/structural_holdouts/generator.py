"""New graph/edit-combination fixtures; benchmark construction, never Agent code."""

import hashlib
import itertools
import random
from copy import deepcopy
from functools import lru_cache

from .identity import canonical_json, fingerprint
from .raw_tables import (
    NO_DELIVERY,
    apply_edit,
    compose,
    equivalent,
    in_one_edit_family,
    validate,
)

CASES = (
    "unchanged",
    "controller_change",
    "redirect",
    "output",
    "insert",
    "invisible_insert",
    "composed",
    "out_of_family",
)


def _rng(seed, index, purpose):
    return random.Random(
        int(
            hashlib.sha256(
                f"structural-corpus-v1:{seed}:{index}:{purpose}".encode()
            ).hexdigest(),
            16,
        )
    )


def _base(seed, index, states):
    rng = _rng(seed, index, "base-graph")
    names, actions, outputs = (
        [f"s-{i}" for i in range(states)],
        [f"a-{i}" for i in range(4)],
        ["o-0", "o-1"],
    )
    rows = []
    for i, state in enumerate(names):
        for action in actions:
            rows.append(
                {
                    "source": state,
                    "action": action,
                    "target": names[(i + 1) % states]
                    if action == actions[0]
                    else rng.choice(names),
                    "output": outputs[1] if rng.randrange(4) == 0 else outputs[0],
                }
            )
    # A reachable positive output is guaranteed, without supplying its location.
    rows[-1]["output"] = outputs[1]
    return {
        "language": "total-mealy-v1",
        "states": names,
        "actions": actions,
        "outputs": outputs,
        "initial_state": names[0],
        "transitions": rows,
    }


def _controllers(actions):
    issued = [*actions, "alternate"]
    result = []
    for mode in range(3):
        states = ["c-0", "c-1"] if mode == 2 else ["c-0"]
        rows = []
        for state in states:
            for index, command in enumerate(issued):
                target = state
                delivery = actions[index] if index < 4 else actions[1]
                if index == 1 and mode:
                    if mode == 1 or state == "c-0":
                        delivery = NO_DELIVERY
                    if mode == 2:
                        target = "c-1" if state == "c-0" else "c-0"
                rows.append(
                    {
                        "source": state,
                        "action": command,
                        "target": target,
                        "output": delivery,
                    }
                )
        result.append(
            {
                "language": "total-mealy-v1",
                "states": states,
                "actions": issued,
                "outputs": [*actions, NO_DELIVERY],
                "initial_state": states[0],
                "transitions": rows,
            }
        )
    return result


def _edit(row, field, value):
    return {
        "operation": "redirect_transition" if field == "target" else "change_output",
        "source": row["source"],
        "action": row["action"],
        "from": row[field],
        "to": value,
    }


def _reachable_rows(program):
    table = validate(program)
    seen, pending = {program["initial_state"]}, [program["initial_state"]]
    for state in pending:
        for action in program["actions"]:
            target, _ = table[state, action]
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return [row for row in program["transitions"] if row["source"] in seen]


def _changes(program, rng, case):
    rows = _reachable_rows(program)
    row = rng.choice(rows)
    if case in ("unchanged", "controller_change"):
        return []
    if case == "redirect":
        return [
            _edit(
                row,
                "target",
                rng.choice([s for s in program["states"] if s != row["target"]]),
            )
        ]
    if case == "output":
        return [
            _edit(
                row, "output", next(o for o in program["outputs"] if o != row["output"])
            )
        ]
    if case in ("insert", "invisible_insert"):
        if case == "insert":
            row = rng.choice([r for r in rows if r["target"] != r["source"]] or rows)
        template = row["target"] if case == "invisible_insert" else row["source"]
        return [
            {
                "operation": "add_state_from_template",
                "template": template,
                "state": "added-state",
            },
            _edit(row, "target", "added-state"),
        ]
    chosen = rng.sample(rows, 2)
    field = "target" if case == "composed" else "output"
    domain = program["states"] if field == "target" else program["outputs"]
    return [
        _edit(r, field, rng.choice([v for v in domain if v != r[field]]))
        for r in chosen
    ]


def _to_ir(program):
    return {
        "mechanism": "finite_transducer",
        "representation": "state_machine",
        "scope": {"language": "total-mealy-v1", "reset": "initial-state"},
        "preconditions": [],
        "latent_variables": ["machine_state"],
        "effects": {"output": "transition.output"},
        "parameters": deepcopy(program),
        "composition": [],
    }


def _rename(program, names):
    result = deepcopy(program)
    result["states"] = [names[v] for v in program["states"]]
    result["actions"] = [names[v] for v in program["actions"]]
    result["outputs"] = [
        names[v] if v != NO_DELIVERY else v for v in program["outputs"]
    ]
    result["initial_state"] = names[program["initial_state"]]
    for row in result["transitions"]:
        for field in ("source", "action", "target", "output"):
            if row[field] != NO_DELIVERY:
                row[field] = names[row[field]]
    # Order is public alphabet contract, not a private action semantic annotation.
    validate(result)
    return result


def public_entry(fixture):
    """Only this detached entry is supplied to learning/planning."""
    return deepcopy(fixture["public"])


def instantiate(private):
    """Trusted world factory; returns only the capability, never fixture metadata."""
    from minecraft_scientist.protocols import AgentWorldView
    from minecraft_scientist.theory.finite_state import FiniteStateProgram
    from minecraft_scientist.world.revision import _SwitchingWorld

    if (
        type(private) is not dict
        or set(private) != {"schema", "before", "after", "boundary_episode"}
        or private["schema"] != "online-revision-private-v1"
        or type(private["boundary_episode"]) is not int
        or private["boundary_episode"] < 0
    ):
        raise ValueError("invalid generated private schedule")
    factors = []
    for epoch in ("before", "after"):
        if set(private[epoch]) != {"mechanism", "controller"}:
            raise ValueError("invalid generated factors")
        compose(private[epoch]["mechanism"], private[epoch]["controller"])
        factors.append(
            tuple(
                FiniteStateProgram.model_validate(private[epoch][key])
                for key in ("mechanism", "controller")
            )
        )
    world = _SwitchingWorld(*factors, private["boundary_episode"])
    return AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=world.step,
        begin_trial=world.begin_trial,
    )


def describe(public, private, edits):
    """Raw diagnostic facts and leakage keys; this function is evaluator-only."""
    before = compose(**private["before"])
    after = compose(**private["after"])
    costs = public["costs"]
    reset = public["outputs"][0]
    identities = {}
    for name, programs in (("target", (after,)), ("pair", (before, after))):
        for mode in ("structure", "behavior"):
            identity = _identity(programs, costs, reset, mode)
            identities[f"{name}-{mode}"] = {
                "sha256": identity.sha256,
                "canonical": identity.canonical,
                "enumerations": identity.enumerations,
            }
    keys = {
        f"{kind}:{identities['target-' + kind]['sha256']}"
        for kind in ("structure", "behavior")
    }
    keys.update(
        f"pair-{kind}:{identities['pair-' + kind]['sha256']}"
        for kind in ("structure", "behavior")
    )
    priors, covered = {}, False
    for hypothesis in public["initial_family"]["hypotheses"]:
        candidate = compose(
            hypothesis["mechanism_ir"]["parameters"],
            hypothesis["controller_ir"]["parameters"],
        )
        covered |= equivalent(candidate, after)
        for mode in ("structure", "behavior"):
            identity = _identity((candidate,), costs, reset, mode)
            keys.add(f"{mode}:{identity.sha256}")
            priors[f"{mode}:{identity.sha256}"] = identity.canonical
    changed = deepcopy(private["before"]["mechanism"])
    for edit in edits:
        changed = apply_edit(changed, edit)
    if canonical_json(changed) != canonical_json(private["after"]["mechanism"]):
        raise ValueError("edit journal does not reconstruct generated after mechanism")
    return {
        "identities": identities,
        "prior_identity_bodies": priors,
        "leakage_keys": sorted(keys),
        "behavior_change_unbounded": not equivalent(before, after),
        "behavior_change_at_H": not equivalent(before, after, horizon=5),
        "target_exact_in_public_family": covered,
        "within_declared_mechanism_edit_family": in_one_edit_family(
            private["before"]["mechanism"], private["after"]["mechanism"]
        ),
        "mechanism_state_count_before": len(private["before"]["mechanism"]["states"]),
        "mechanism_state_count_after": len(private["after"]["mechanism"]["states"]),
        "scope": "raw deterministic reset issued behavior; prior/target/pair graph keys, not physical causal uniqueness or statistical method results",
    }


def fixture(
    *, seed=0, base_index=0, states=4, case="redirect", renaming_index=0, boundary=20
):
    if (
        any(
            type(v) is not int or v < 0
            for v in (seed, base_index, renaming_index, boundary)
        )
        or type(states) is not int
        or not 2 <= states <= 5
        or case not in CASES
    ):
        raise ValueError("invalid new-graph fixture protocol")
    base = _base(seed, base_index, states)
    rng = _rng(seed, base_index, "public-family")
    row = rng.choice(base["transitions"])
    alternatives = [
        base,
        apply_edit(
            base,
            _edit(
                row, "output", next(o for o in base["outputs"] if o != row["output"])
            ),
        ),
    ]
    row = rng.choice(base["transitions"])
    alternatives.append(
        apply_edit(
            base,
            _edit(
                row,
                "target",
                rng.choice([s for s in base["states"] if s != row["target"]]),
            ),
        )
    )
    controllers = _controllers(base["actions"])
    truth_rng = _rng(seed, base_index, "private-old-selection")
    mi, ci = truth_rng.randrange(3), truth_rng.randrange(3)
    old = alternatives[mi]
    edits = _changes(old, _rng(seed, base_index, "change-" + case), case)
    after = deepcopy(old)
    for edit in edits:
        after = apply_edit(after, edit)
    after_ci = (ci + 1) % 3 if case == "controller_change" else ci
    roles = [
        *base["states"],
        "added-state",
        *base["actions"],
        *base["outputs"],
        "alternate",
        "c-0",
        "c-1",
        *(f"mechanism-{i}" for i in range(3)),
        *(f"controller-{i}" for i in range(3)),
    ]
    names = {
        role: "id-"
        + hashlib.sha256(
            f"structural-opaque-v1:{seed}:{base_index}:{renaming_index}:{role}".encode()
        ).hexdigest()[:16]
        for role in roles
    }
    renamed = [_rename(p, names) for p in alternatives]
    controls = [_rename(p, names) for p in controllers]
    hypotheses = [
        {
            "mechanism_id": names[f"mechanism-{m}"],
            "controller_id": names[f"controller-{c}"],
            "mechanism_ir": _to_ir(renamed[m]),
            "controller_ir": _to_ir(controls[c]),
        }
        for m, c in itertools.product(range(3), range(3))
    ]
    public = {
        "actions": list(controls[0]["actions"]),
        "outputs": list(renamed[0]["outputs"]),
        "costs": {
            names[a]: 3 if a == "alternate" else 1 for a in controllers[0]["actions"]
        },
        "initial_family": {
            "hypotheses": hypotheses,
            "calibration": [names[a] for a in base["actions"]],
            "discovery_budget": 80,
            "shadow_count": 3,
        },
    }
    private = {
        "schema": "online-revision-private-v1",
        "boundary_episode": boundary,
        "before": {"mechanism": renamed[mi], "controller": controls[ci]},
        "after": {"mechanism": _rename(after, names), "controller": controls[after_ci]},
    }
    renamed_edits = [
        {
            key: names.get(value, value) if key != "operation" else value
            for key, value in edit.items()
        }
        for edit in edits
    ]
    inputs = {
        "seed": seed,
        "base_index": base_index,
        "states": states,
        "case": case,
        "renaming_index": renaming_index,
        "boundary": boundary,
    }
    metadata = {
        "schema": "generated-structure-metadata-v1",
        "fixture_id": hashlib.sha256(canonical_json(inputs).encode()).hexdigest()[:24],
        "inputs": inputs,
        "old_selection": [mi, ci],
        "edits": renamed_edits,
        "diagnostics": describe(public, private, renamed_edits),
    }
    return {"public": public, "private": private, "metadata": metadata}


@lru_cache(maxsize=8192)
def _cached_identity(program_json, costs_json, reset, mode):
    import json

    return fingerprint(
        tuple(json.loads(program_json)),
        json.loads(costs_json),
        reset_output=reset,
        mode=mode,
    )


def _identity(programs, costs, reset, mode):
    return _cached_identity(
        canonical_json(programs), canonical_json(costs), reset, mode
    )

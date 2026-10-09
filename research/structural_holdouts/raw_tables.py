"""Independent raw-table construction/scoring, never a learner equivalence oracle."""

import itertools
from collections import deque
from copy import deepcopy

NO_DELIVERY = "_no_delivery_"


def validate(raw, *, output_cap=8):
    if (
        type(raw) is not dict
        or set(raw)
        != {"language", "states", "actions", "outputs", "initial_state", "transitions"}
        or raw["language"] != "total-mealy-v1"
    ):
        raise ValueError("invalid raw Mealy schema")
    for key, limit in (("states", 16), ("actions", 6), ("outputs", output_cap)):
        values = raw[key]
        if (
            type(values) is not list
            or not 1 <= len(values) <= limit
            or any(type(v) is not str or not 1 <= len(v) <= 80 for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("invalid raw alphabet")
    if (
        type(raw["initial_state"]) is not str
        or raw["initial_state"] not in raw["states"]
        or type(raw["transitions"]) is not list
        or len(raw["transitions"]) > 128
    ):
        raise ValueError("invalid raw initial state/table")
    table = {}
    for row in raw["transitions"]:
        if (
            type(row) is not dict
            or set(row) != {"source", "action", "target", "output"}
            or any(type(v) is not str for v in row.values())
        ):
            raise ValueError("invalid raw transition")
        if (
            row["source"] not in raw["states"]
            or row["target"] not in raw["states"]
            or row["action"] not in raw["actions"]
            or row["output"] not in raw["outputs"]
            or (row["source"], row["action"]) in table
        ):
            raise ValueError("invalid/duplicate raw transition label")
        table[row["source"], row["action"]] = row["target"], row["output"]
    if set(table) != set(itertools.product(raw["states"], raw["actions"])):
        raise ValueError("raw table must be total")
    return table


def compose(mechanism, controller):
    m, c = validate(mechanism, output_cap=3), validate(controller)
    if any(
        o not in (*mechanism["actions"], NO_DELIVERY) for o in controller["outputs"]
    ):
        raise ValueError("controller delivered unknown action")
    pairs = list(itertools.product(mechanism["states"], controller["states"]))
    if len(pairs) > 16:
        raise ValueError("raw product exceeds declared state bound")
    names = {pair: f"product-{index}" for index, pair in enumerate(pairs)}
    rows = []
    for ms, cs in pairs:
        for action in controller["actions"]:
            cn, delivery = c[cs, action]
            mn, output = (
                (ms, mechanism["outputs"][0])
                if delivery == NO_DELIVERY
                else m[ms, delivery]
            )
            rows.append(
                {
                    "source": names[ms, cs],
                    "action": action,
                    "target": names[mn, cn],
                    "output": output,
                }
            )
    result = {
        "language": "total-mealy-v1",
        "states": list(names.values()),
        "actions": list(controller["actions"]),
        "outputs": list(mechanism["outputs"]),
        "initial_state": names[mechanism["initial_state"], controller["initial_state"]],
        "transitions": rows,
    }
    validate(result, output_cap=3)
    return result


def trace(program, actions):
    table, state = validate(program), program["initial_state"]
    if (
        type(actions) is not tuple
        or len(actions) > 256
        or any(type(a) is not str or a not in program["actions"] for a in actions)
    ):
        raise ValueError("invalid raw trace word")
    outputs = []
    for action in actions:
        state, output = table[state, action]
        outputs.append(output)
    return tuple(outputs)


def equivalent(left, right, *, horizon=None):
    l, r = validate(left), validate(right)
    if set(left["actions"]) != set(right["actions"]) or (
        horizon is not None and (type(horizon) is not int or not 0 <= horizon <= 8)
    ):
        raise ValueError("invalid raw equivalence protocol")
    if left["outputs"][0] != right["outputs"][0]:
        return False
    pending, seen = deque([(left["initial_state"], right["initial_state"], 0)]), set()
    while pending:
        ls, rs, depth = pending.popleft()
        if (ls, rs) in seen or (horizon is not None and depth >= horizon):
            continue
        seen.add((ls, rs))
        for action in left["actions"]:
            ln, lo = l[ls, action]
            rn, ro = r[rs, action]
            if lo != ro:
                return False
            pending.append((ln, rn, depth + 1))
    return True


def apply_edit(program, edit):
    table = validate(program)
    changed = deepcopy(program)
    operation = edit.get("operation") if type(edit) is dict else None
    if operation in ("redirect_transition", "change_output"):
        field = "target" if operation == "redirect_transition" else "output"
        if (
            set(edit) != {"operation", "source", "action", "from", "to"}
            or (edit["source"], edit["action"]) not in table
        ):
            raise ValueError("invalid raw row edit")
        row = next(
            r
            for r in changed["transitions"]
            if (r["source"], r["action"]) == (edit["source"], edit["action"])
        )
        if row[field] != edit["from"] or edit["from"] == edit["to"]:
            raise ValueError("raw edit old-value mismatch/no-op")
        row[field] = edit["to"]
    elif operation == "add_state_from_template":
        if (
            set(edit) != {"operation", "template", "state"}
            or edit["template"] not in program["states"]
            or type(edit["state"]) is not str
            or edit["state"] in program["states"]
        ):
            raise ValueError("invalid raw state insertion")
        changed["states"].append(edit["state"])
        for action in program["actions"]:
            target, output = table[edit["template"], action]
            changed["transitions"].append(
                {
                    "source": edit["state"],
                    "action": action,
                    "target": edit["state"] if target == edit["template"] else target,
                    "output": output,
                }
            )
    else:
        raise ValueError("unknown raw edit")
    validate(changed)
    return changed


def in_one_edit_family(before, after):
    """Raw original-state alignment, not behavior/causal identifiability."""
    old, new = validate(before), validate(after)
    if (
        before["actions"] != after["actions"]
        or before["outputs"] != after["outputs"]
        or before["initial_state"] != after["initial_state"]
    ):
        return False
    if set(before["states"]) == set(after["states"]):
        return (
            sum(
                sum(a != b for a, b in zip(old[key], new[key], strict=True))
                for key in old
            )
            <= 1
        )
    added = set(after["states"]) - set(before["states"])
    if len(added) != 1 or set(before["states"]) - set(after["states"]):
        return False
    name = next(iter(added))
    differences = [key for key in old if old[key] != new[key]]
    if len(differences) != 1 or new[differences[0]] != (name, old[differences[0]][1]):
        return False
    return any(
        all(
            new[name, a]
            == (
                name if old[template, a][0] == template else old[template, a][0],
                old[template, a][1],
            )
            for a in before["actions"]
        )
        for template in before["states"]
    )

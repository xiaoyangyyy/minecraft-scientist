"""Independent certificate replay from raw IR branches, no search invocation.

For equality: verify actual reachable basis, initial inclusion, zero terminal
values and closure for EVERY allowed (action,output). Batch Gauss-Jordan span
checks do not trust generator vectors, rank, queue, pivots or status.
For inequality: raw probability propagation verifies the concrete output event.
Shared IR schema validation is reused, but not compiled prediction/filtering.
"""

import hashlib
import json
from fractions import Fraction
from math import gcd

from research.stochastic_ir.program import StochasticProgram

MAX_BITS = 8192
MAX_OPERATIONS = 10_000_000
SCHEMA = "public-stochastic-equivalence-certificate-v1"
SCOPE = "Declared total public stochastic TheoryIR programs, fixed explicit state beliefs and allowed action subset; all finite output-prefix probabilities, not actual-world identifiability, calibration, minimal cost, experiment execution or theory acceptance."


def fraction(value):
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        raise ValueError("canonical exact signed rational required")
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator < 1
        or gcd(numerator, denominator) != 1
        or max(numerator.bit_length(), denominator.bit_length()) > MAX_BITS
    ):
        raise ValueError("noncanonical or oversized certificate rational")
    return Fraction(numerator, denominator)


def model_hash(ir):
    return hashlib.sha256(
        json.dumps(
            ir.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class AuditWork:
    def __init__(self):
        self.operations = 0

    def check(self, value):
        self.operations += 1
        if (
            self.operations > MAX_OPERATIONS
            or max(value.numerator.bit_length(), value.denominator.bit_length())
            > MAX_BITS
        ):
            raise ValueError("independent certificate audit bound exhausted")
        return value


def verify_certificate(
    left_ir,
    right_ir,
    certificate,
    *,
    left_belief=None,
    right_belief=None,
    allowed_actions=None,
):
    keys = {
        "schema",
        "scope",
        "status",
        "left_ir_sha256",
        "right_ir_sha256",
        "left_belief",
        "right_belief",
        "allowed_actions",
        "dimension",
        "basis",
        "witness",
        "arithmetic",
    }
    if (
        type(certificate) is not dict
        or set(certificate) != keys
        or certificate["schema"] != SCHEMA
        or certificate["scope"] != SCOPE
    ):
        raise ValueError("invalid exact certificate shape/scope")
    left, right = (
        StochasticProgram.from_ir(left_ir),
        StochasticProgram.from_ir(right_ir),
    )
    if (
        set(left.actions) != set(right.actions)
        or set(left.outputs) != set(right.outputs)
        or certificate["left_ir_sha256"] != model_hash(left_ir)
        or certificate["right_ir_sha256"] != model_hash(right_ir)
    ):
        raise ValueError("certificate public models/alphabets mismatch")
    actions = certificate["allowed_actions"]
    expected_actions = left.actions if allowed_actions is None else allowed_actions
    if (
        type(expected_actions) is not tuple
        or len(set(expected_actions)) != len(expected_actions)
        or any(type(a) is not str or a not in left.actions for a in expected_actions)
        or actions != list(expected_actions)
    ):
        raise ValueError("certificate changed caller-pinned allowed action scope")
    if (
        type(actions) is not list
        or len(set(actions)) != len(actions)
        or any(type(a) is not str or a not in left.actions for a in actions)
    ):
        raise ValueError("invalid allowed action subset")
    beliefs = []
    for model, key in ((left, "left_belief"), (right, "right_belief")):
        payload = certificate[key]
        if type(payload) is not dict:
            raise ValueError("invalid certificate belief")
        values = {s: fraction(v) for s, v in payload.items()}
        if any(v <= 0 for v in values.values()):
            raise ValueError("belief uses positive sparse canonical mass")
        beliefs.append(model.belief(values))
    if beliefs != [left.belief(left_belief), right.belief(right_belief)]:
        raise ValueError("certificate changed caller-pinned state beliefs")
    dimension = len(left.states) + len(right.states)
    if (
        type(certificate["dimension"]) is not int
        or certificate["dimension"] != dimension
    ):
        raise ValueError("certificate dimension mismatch")
    arithmetic = certificate["arithmetic"]
    if (
        type(arithmetic) is not dict
        or set(arithmetic) != {"operations", "operation_limit", "bit_limit"}
        or any(type(v) is not int for v in arithmetic.values())
        or not 1 <= arithmetic["bit_limit"] <= MAX_BITS
        or not 0
        <= arithmetic["operations"]
        <= arithmetic["operation_limit"]
        <= MAX_OPERATIONS
        or arithmetic["operation_limit"] < 1
    ):
        raise ValueError("invalid reported search arithmetic bounds")
    work = AuditWork()
    # These loops use original branch dictionaries, not generator operators.
    specs = [ir.parameters for ir in (left_ir, right_ir)]
    indices = [{s: i for i, s in enumerate(spec["states"])} for spec in specs]
    initial = tuple(beliefs[0].get(s, Fraction(0)) for s in left.states) + tuple(
        -beliefs[1].get(s, Fraction(0)) for s in right.states
    )

    def step(vector, action, output):
        following = [Fraction(0)] * dimension
        offset = 0
        for spec, index in zip(specs, indices, strict=True):
            for row in spec["transitions"]:
                if row["action"] == action and row["output"] == output:
                    probability = Fraction(
                        row["probability"]["numerator"],
                        row["probability"]["denominator"],
                    )
                    target = offset + index[row["target"]]
                    product = work.check(
                        vector[offset + index[row["source"]]] * probability
                    )
                    following[target] = work.check(following[target] + product)
            offset += len(index)
        return tuple(following)

    def trace_vector(trace):
        if type(trace) is not list or len(trace) > dimension - 1:
            raise ValueError("invalid reachable trace bound")
        vector = initial
        for pair in trace:
            if (
                type(pair) is not dict
                or set(pair) != {"action", "output"}
                or pair["action"] not in actions
                or pair["output"] not in left.outputs
            ):
                raise ValueError("trace uses undeclared public symbols")
            vector = step(vector, pair["action"], pair["output"])
        return vector

    if certificate["status"] == "distinguishable":
        witness = certificate["witness"]
        if (
            certificate["basis"] is not None
            or type(witness) is not dict
            or set(witness)
            != {"actions", "outputs", "left_probability", "right_probability"}
            or type(witness["actions"]) is not list
            or type(witness["outputs"]) is not list
            or not witness["actions"]
            or len(witness["actions"]) != len(witness["outputs"])
        ):
            raise ValueError("invalid concrete witness")
        vector = trace_vector(
            [
                {"action": a, "output": o}
                for a, o in zip(witness["actions"], witness["outputs"], strict=True)
            ]
        )
        lp, rp = (
            sum(vector[: len(left.states)], Fraction(0)),
            -sum(vector[len(left.states) :], Fraction(0)),
        )
        if (
            lp == rp
            or lp != fraction(witness["left_probability"])
            or rp != fraction(witness["right_probability"])
        ):
            raise ValueError("witness raw probability mismatch")
        return {
            "valid": True,
            "status": "distinguishable",
            "witness_length": len(witness["actions"]),
            "audit_operations": work.operations,
            "search_operations_independently_replayed": False,
        }
    if certificate["status"] != "equivalent" or certificate["witness"] is not None:
        raise ValueError("unrecognized certificate result")
    basis = certificate["basis"]
    if type(basis) is not list or not 1 <= len(basis) <= dimension:
        raise ValueError("invalid complete basis size")
    vectors = []
    for entry in basis:
        if (
            type(entry) is not dict
            or set(entry) != {"trace", "vector"}
            or type(entry["vector"]) is not list
            or len(entry["vector"]) != dimension
        ):
            raise ValueError("invalid basis record")
        vector = trace_vector(entry["trace"])
        if vector != tuple(fraction(v) for v in entry["vector"]) or sum(
            vector, Fraction(0)
        ):
            raise ValueError("basis is not reachable/zero-terminal")
        vectors.append(vector)
    # Independent batch full row reduction, unlike incremental search echelon.
    reduced = [list(v) for v in vectors]
    pivots, rank = [], 0
    for col in range(dimension):
        selected = next((i for i in range(rank, len(reduced)) if reduced[i][col]), None)
        if selected is None:
            continue
        reduced[rank], reduced[selected] = reduced[selected], reduced[rank]
        scale = reduced[rank][col]
        reduced[rank] = [work.check(v / scale) for v in reduced[rank]]
        for i in range(len(reduced)):
            if i != rank and reduced[i][col]:
                scale = reduced[i][col]
                reduced[i] = [
                    work.check(x - work.check(scale * y))
                    for x, y in zip(reduced[i], reduced[rank], strict=True)
                ]
        pivots.append(col)
        rank += 1
    if rank != len(vectors):
        raise ValueError("basis vectors are linearly dependent")

    def in_span(vector):
        residual = list(vector)
        for row, pivot in zip(reduced, pivots, strict=True):
            scale = residual[pivot]
            if scale:
                residual = [
                    work.check(x - work.check(scale * y))
                    for x, y in zip(residual, row, strict=True)
                ]
        return not any(residual)

    if not in_span(initial):
        raise ValueError("basis omits initial reachable vector")
    for vector in vectors:
        for action in actions:
            for output in left.outputs:
                if not in_span(step(vector, action, output)):
                    raise ValueError(
                        "basis omits reachable operator image; no full equivalence"
                    )
    return {
        "valid": True,
        "status": "equivalent",
        "reachable_rank": rank,
        "audit_operations": work.operations,
        "search_operations_independently_replayed": False,
    }

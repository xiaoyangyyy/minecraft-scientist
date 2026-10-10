"""Standard rational weighted-automaton forward-space equivalence.

The alphabet is (issued action, observed output), not hidden delivery/state.
An output-prefix probability is initial @ product(M[action,output]) @ ones.
Direct-sum subtraction makes equality a zero-series question. Basis closure
proves all finite words; bounded horizon search alone cannot do that.
See Kiefer 2020, Algorithm 2 / Theorem 2.3; no novelty claim.
"""

import hashlib
import json
from collections import deque
from fractions import Fraction

from research.stochastic_ir.program import StochasticProgram

MAX_OPERATIONS = 10_000_000
MAX_BITS = 8192
SCHEMA = "public-stochastic-equivalence-certificate-v1"
SCOPE = "Declared total public stochastic TheoryIR programs, fixed explicit state beliefs and allowed action subset; all finite output-prefix probabilities, not actual-world identifiability, calibration, minimal cost, experiment execution or theory acceptance."


class EquivalenceBound(ValueError):
    """No answer/certificate when exact computation exceeds its declared bound."""


def encoded(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def ir_hash(ir):
    return hashlib.sha256(
        json.dumps(
            ir.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class Work:
    def __init__(self, limit, bits):
        if (
            type(limit) is not int
            or not 1 <= limit <= MAX_OPERATIONS
            or type(bits) is not int
            or not 1 <= bits <= MAX_BITS
        ):
            raise ValueError("strict bounded operation/bit limits required")
        self.limit, self.bits, self.operations = limit, bits, 0

    def checked(self, value):
        if (
            max(value.numerator.bit_length(), value.denominator.bit_length())
            > self.bits
        ):
            raise EquivalenceBound(
                "exact coefficient bit bound; no floating approximation"
            )
        return value

    def operation(self, value):
        self.operations += 1
        if self.operations > self.limit:
            raise EquivalenceBound(
                "arithmetic bound; no partial equivalence certificate"
            )
        return self.checked(value)


class Span:
    """Incremental echelon rows; retain actual reachable vectors separately."""

    def __init__(self, work):
        self.rows, self.work = {}, work

    def add(self, vector):
        residual = list(vector)
        for pivot, row in sorted(self.rows.items()):
            factor = residual[pivot]
            if factor:
                for index in range(pivot, len(residual)):
                    if row[index]:
                        product = self.work.operation(factor * row[index])
                        residual[index] = self.work.operation(residual[index] - product)
        pivot = next((i for i, value in enumerate(residual) if value), None)
        if pivot is None:
            return False
        factor = residual[pivot]
        self.rows[pivot] = tuple(self.work.operation(v / factor) for v in residual)
        return True


def compare(
    left_ir,
    right_ir,
    *,
    left_belief=None,
    right_belief=None,
    allowed_actions=None,
    operation_limit=MAX_OPERATIONS,
    bit_limit=MAX_BITS,
):
    left, right = (
        StochasticProgram.from_ir(left_ir),
        StochasticProgram.from_ir(right_ir),
    )
    if set(left.actions) != set(right.actions) or set(left.outputs) != set(
        right.outputs
    ):
        raise ValueError("shared public action/output alphabets required")
    actions = left.actions if allowed_actions is None else allowed_actions
    if (
        type(actions) is not tuple
        or len(set(actions)) != len(actions)
        or any(type(a) is not str or a not in left.actions for a in actions)
    ):
        raise ValueError("allowed actions must be a declared unique tuple subset")
    lb, rb = left.belief(left_belief), right.belief(right_belief)
    work = Work(operation_limit, bit_limit)
    nleft, dimension = len(left.states), len(left.states) + len(right.states)
    initial = tuple(work.checked(lb.get(s, Fraction(0))) for s in left.states) + tuple(
        work.checked(-rb.get(s, Fraction(0))) for s in right.states
    )
    operators = {}
    for action in actions:
        for output in left.outputs:
            edges = []
            for program, offset in ((left, 0), (right, nleft)):
                indices = {s: i + offset for i, s in enumerate(program.states)}
                for row in program.transitions:
                    if row.action == action and row.output == output:
                        edges.append(
                            (
                                indices[row.source],
                                indices[row.target],
                                work.checked(row.probability.fraction()),
                            )
                        )
            operators[action, output] = tuple(edges)

    def apply(vector, key):
        following = [Fraction(0)] * dimension
        for source, target, probability in operators[key]:
            if vector[source]:
                product = work.operation(vector[source] * probability)
                following[target] = work.operation(following[target] + product)
        return tuple(following)

    span, queue, reachable = Span(work), deque(), []
    span.add(initial)
    queue.append(((), initial))
    reachable.append(((), initial))
    witness = None
    while queue and witness is None:
        trace, vector = queue.popleft()
        for key in operators:
            following = apply(vector, key)
            difference = Fraction(0)
            for coefficient in following:
                difference = work.operation(difference + coefficient)
            trace_next = trace + (key,)
            if difference:
                if len(trace_next) > dimension - 1:
                    raise ArithmeticError(
                        "counterexample exceeds weighted-automaton dimension bound"
                    )
                witness = {
                    "actions": [a for a, _ in trace_next],
                    "outputs": [o for _, o in trace_next],
                    "left_probability": encoded(
                        work.checked(sum(following[:nleft], Fraction(0)))
                    ),
                    "right_probability": encoded(
                        work.checked(-sum(following[nleft:], Fraction(0)))
                    ),
                }
                break
            if span.add(following):
                if len(reachable) >= dimension:
                    raise ArithmeticError("reachable basis exceeds ambient dimension")
                reachable.append((trace_next, following))
                queue.append((trace_next, following))
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "status": "distinguishable" if witness else "equivalent",
        "left_ir_sha256": ir_hash(left_ir),
        "right_ir_sha256": ir_hash(right_ir),
        "left_belief": {s: encoded(p) for s, p in sorted(lb.items())},
        "right_belief": {s: encoded(p) for s, p in sorted(rb.items())},
        "allowed_actions": list(actions),
        "dimension": dimension,
        "basis": [
            {
                "trace": [{"action": a, "output": o} for a, o in trace],
                "vector": [encoded(v) for v in vector],
            }
            for trace, vector in reachable
        ]
        if witness is None
        else None,
        "witness": witness,
        "arithmetic": {
            "operations": work.operations,
            "operation_limit": operation_limit,
            "bit_limit": bit_limit,
        },
    }

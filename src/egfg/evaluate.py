"""Numeric evaluation of a Dag under a pluggable semiring.

Tables keep their variables in sorted order. Semirings with several
components (the expectation semiring) store them on a leading axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cost import node_cost, reachable
from .ir import Dag, dag_scopes
from .model import Factor, FactorGraph


@dataclass
class Table:
    vars: tuple[str, ...]
    data: np.ndarray


class Semiring(Protocol):
    def lift(self, factor: Factor) -> Table: ...
    def mul(self, x: Table, y: Table) -> Table: ...
    def sum_out(self, x: Table, var: str) -> Table: ...


def _sorted_table(factor: Factor, data: np.ndarray, lead: int) -> Table:
    """Reorder the factor axes (after `lead` component axes) into sorted variable order."""
    order = sorted(range(len(factor.scope)), key=lambda i: factor.scope[i])
    perm = list(range(lead)) + [lead + i for i in order]
    return Table(tuple(factor.scope[i] for i in order), np.transpose(data, perm))


def _broadcast(x: Table, out_vars: tuple[str, ...], lead: int) -> np.ndarray:
    """View x.data with one axis per out_vars entry (size 1 where x lacks the variable)."""
    shape = list(x.data.shape[:lead])
    for v in out_vars:
        shape.append(x.data.shape[lead + x.vars.index(v)] if v in x.vars else 1)
    return x.data.reshape(shape)  # x.vars is a sorted subsequence of out_vars


def _union(x: Table, y: Table) -> tuple[str, ...]:
    return tuple(sorted(set(x.vars) | set(y.vars)))


class _Scalar:
    """Semiring on plain non-negative tables; `reduce` is the additive operation."""

    def __init__(self, reduce):
        self.reduce = reduce

    def lift(self, factor: Factor) -> Table:
        return _sorted_table(factor, np.asarray(factor.table, dtype=float), 0)

    def mul(self, x: Table, y: Table) -> Table:
        out = _union(x, y)
        return Table(out, _broadcast(x, out, 0) * _broadcast(y, out, 0))

    def sum_out(self, x: Table, var: str) -> Table:
        i = x.vars.index(var)
        return Table(x.vars[:i] + x.vars[i + 1 :], self.reduce(x.data, axis=i))


SUM_PRODUCT = _Scalar(np.sum)
MAX_PRODUCT = _Scalar(np.max)


class ExpectationSemiring:
    """Second-order expectation semiring: elements (p, r, s) = (φ, φh, φh²).

    For f(x) = Σ_k h_k(x_k), summing the product over all variables gives
    (Z, Z·E[f], Z·E[f²]).
    """

    def __init__(self, feature: dict[int, np.ndarray]):
        self.feature = feature

    def lift(self, factor: Factor) -> Table:
        phi = np.asarray(factor.table, dtype=float)
        h = self.feature.get(factor.id, np.zeros_like(phi))
        return _sorted_table(factor, np.stack([phi, phi * h, phi * h * h]), 1)

    def mul(self, x: Table, y: Table) -> Table:
        out = _union(x, y)
        p1, r1, s1 = _broadcast(x, out, 1)
        p2, r2, s2 = _broadcast(y, out, 1)
        return Table(out, np.stack([p1 * p2, p1 * r2 + r1 * p2, p1 * s2 + s1 * p2 + 2 * r1 * r2]))

    def sum_out(self, x: Table, var: str) -> Table:
        i = x.vars.index(var)
        return Table(x.vars[:i] + x.vars[i + 1 :], x.data.sum(axis=1 + i))


def value_feature(fg: FactorGraph, var: str, values: np.ndarray) -> dict[int, np.ndarray]:
    """Feature h(x) = values[x_var], attached to the lowest-id factor containing var."""
    f = min((f for f in fg.factors if var in f.scope), key=lambda f: f.id)
    shape = [1] * len(f.scope)
    shape[f.scope.index(var)] = len(values)
    return {f.id: np.broadcast_to(np.asarray(values, float).reshape(shape), f.table.shape).copy()}


def evaluate(
    dag: Dag, fg: FactorGraph, semiring, inputs: dict[str, Table] | None = None
) -> tuple[dict[str, Table], int]:
    """Evaluate each reachable node once (memoized); return root tables and the FLOP count.

    Input leaves take their value from `inputs` (already in the semiring's form);
    an input leaf without a value is an error.
    """
    inputs = inputs or {}
    scopes = dag_scopes(dag, fg)
    order = reachable(dag)
    unresolved = sorted(
        {dag.nodes[nid].arg for nid in order if dag.nodes[nid].op == "input"} - set(inputs)
    )
    if unresolved:
        raise ValueError(f"Dag still contains input leaves: {unresolved}")
    done: dict[str, Table] = {}
    flops = 0
    # children before parents: process in reverse DFS-discovery order with an explicit check
    pending = list(reversed(order))
    while pending:
        progressed = []
        for nid in pending:
            node = dag.nodes[nid]
            if any(c not in done for c in node.children):
                progressed.append(nid)
                continue
            if node.op == "leaf":
                done[nid] = semiring.lift(fg.factor(node.arg))
            elif node.op == "input":
                done[nid] = inputs[node.arg]
            elif node.op == "mul":
                done[nid] = semiring.mul(done[node.children[0]], done[node.children[1]])
            else:
                done[nid] = semiring.sum_out(done[node.children[0]], node.arg)
            flops += node_cost(node, scopes[nid], [scopes[c] for c in node.children], fg)
        if len(progressed) == len(pending):
            raise ValueError("Dag has a cycle")
        pending = progressed
    return {name: done[nid] for name, nid in dag.roots.items()}, flops

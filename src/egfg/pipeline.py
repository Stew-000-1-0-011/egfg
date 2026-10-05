"""Public API: optimize the computation of all marginals, then query it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .egraph import SaturationResult, saturate
from .evaluate import MAX_PRODUCT, SUM_PRODUCT, ExpectationSemiring, Table, evaluate, value_feature
from .extract import Extraction, extract_dag_ilp, extract_tree
from .ir import all_marginal_queries
from .model import Factor, FactorGraph


@dataclass
class OptimizeResult:
    saturation: SaturationResult
    extraction: Extraction


def optimize(
    fg: FactorGraph,
    extractor: Literal["ilp", "tree"] = "ilp",
    max_iters: int = 30,
    node_limit: int = 50_000,
) -> OptimizeResult:
    sat = saturate(fg, all_marginal_queries(fg), max_iters=max_iters, node_limit=node_limit)
    ex = extract_dag_ilp(sat.graph, fg) if extractor == "ilp" else extract_tree(sat.graph, fg)
    return OptimizeResult(sat, ex)


def marginals(fg: FactorGraph, res: OptimizeResult) -> dict[str, np.ndarray]:
    tables, _ = evaluate(res.extraction.dag, fg, SUM_PRODUCT)
    return {v: t.data / t.data.sum() for v, t in tables.items()}


class _Clamped:
    """Max-product with some variables clamped: inconsistent factor entries become 0."""

    def __init__(self, fixed: dict[str, int]):
        self.fixed = fixed

    def lift(self, factor: Factor) -> Table:
        table = np.array(factor.table, dtype=float)
        for axis, v in enumerate(factor.scope):
            if v in self.fixed:
                mask = np.zeros(table.shape[axis], dtype=bool)
                mask[self.fixed[v]] = True
                shape = [1] * table.ndim
                shape[axis] = -1
                table = np.where(mask.reshape(shape), table, 0.0)
        return MAX_PRODUCT.lift(Factor(factor.id, factor.scope, table))

    def mul(self, x: Table, y: Table) -> Table:
        return MAX_PRODUCT.mul(x, y)

    def sum_out(self, x: Table, var: str) -> Table:
        return MAX_PRODUCT.sum_out(x, var)


def mode(fg: FactorGraph, res: OptimizeResult) -> dict[str, int]:
    """A maximizing joint assignment (MAP).

    Variables are fixed one at a time in name order: each takes the argmax of
    its max-marginal given the values already fixed. This stays a true
    maximizer even when several assignments tie, which independent per-variable
    argmaxes would not guarantee.
    """
    fixed: dict[str, int] = {}
    for v in fg.variables():
        tables, _ = evaluate(res.extraction.dag, fg, _Clamped(fixed))
        fixed[v] = int(np.argmax(tables[v].data))
    return fixed


def moments(fg: FactorGraph, res: OptimizeResult, var: str, values: np.ndarray) -> tuple[float, float]:
    """Mean and variance of values[x_var] under the normalized distribution."""
    sr = ExpectationSemiring(value_feature(fg, var, values))
    tables, _ = evaluate(res.extraction.dag, fg, sr)
    p, r, s = (float(tables[var].data[i].sum()) for i in range(3))
    E = r / p
    return E, s / p - E * E

"""Public API: optimize the computation of all marginals, then query it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cost import dag_cost
from .decompose import clusters, local_problems, stitch
from .egraph import SaturationResult
from .search import run_strategy
from .evaluate import MAX_PRODUCT, SUM_PRODUCT, ExpectationSemiring, Table, evaluate, value_feature
from .extract import Extraction, extract_dag_greedy, extract_dag_ilp, extract_tree
from .jtree import junction_tree
from .model import Factor, FactorGraph


@dataclass
class OptimizeResult:
    saturations: list[SaturationResult]  # one per cluster
    extraction: Extraction  # the stitched computation of all marginals
    clusters: list[frozenset[str]]  # variables of each cluster
    local_extractions: list[Extraction]
    eval_fg: FactorGraph | None = None  # the graph to evaluate on when it differs from the input
    sum_product_only: bool = False  # True when the computation relies on sum-only equalities
    structured: object | None = None  # the structure.Structured used (low-rank splits), if any
    partition: object | None = None  # partition.PartitionInfo when the partition was searched

    @property
    def saturation(self) -> SaturationResult:
        if len(self.saturations) != 1:
            raise ValueError("decomposed result has one saturation per cluster; use .saturations")
        return self.saturations[0]

    @property
    def hit_limit(self) -> bool:
        return any(s.hit_limit for s in self.saturations)

    @property
    def num_nodes(self) -> int:
        return sum(s.num_nodes for s in self.saturations)

    @property
    def max_cluster_nodes(self) -> int:
        return max(s.num_nodes for s in self.saturations)

    @property
    def saturate_s(self) -> float:
        return sum(s.seconds for s in self.saturations)

    @property
    def extract_s(self) -> float:
        return sum(e.seconds for e in self.local_extractions)


def _extract(g, fg: FactorGraph, extractor: str, time_limit_s: float, weights=None, overhead: int = 0,
             shape_penalty: float = 0.0) -> Extraction:
    if shape_penalty and extractor != "greedy":
        raise ValueError("shape_penalty needs the greedy extractor")
    if extractor == "ilp":
        return extract_dag_ilp(g, fg, time_limit_s=time_limit_s, weights=weights, overhead=overhead)
    if extractor == "greedy":
        return extract_dag_greedy(g, fg, time_limit_s=time_limit_s, weights=weights, overhead=overhead,
                                  shape_penalty=shape_penalty)
    if extractor == "tree":
        return extract_tree(g, fg, weights=weights, overhead=overhead)
    raise ValueError(f"unknown extractor {extractor!r}")


def optimize(
    fg: FactorGraph,
    extractor: Literal["ilp", "greedy", "tree"] = "ilp",
    max_iters: int = 30,
    node_limit: int = 50_000,
    rules: Literal["full", "no_reverse", "minimal"] = "full",
    cluster_budget: int | None = None,
    seed: bool = False,
    time_limit_s: float = 60,
    strategy: str = "seeds+staged",
    structure: tuple[str, ...] = (),
    call_overhead: int = 0,
    shape_penalty: float = 0.0,
    partition: Literal["budget", "search"] = "budget",
    partition_time_s: float = 120,
    max_cluster_vars: int = 12,
    exchange: bool = True,
    boundary: bool = True,
) -> OptimizeResult:
    """Search for a cheap computation of all marginals.

    `strategy` orders the rewrites (see search.py); "bfs" with the other defaults
    reproduces phase 1. `cluster_budget` splits the junction tree
    into clusters of at most that many variables (larger cliques stay alone);
    `seed` unions the junction tree computation into each query first.
    `time_limit_s` bounds each ILP / greedy extraction. `shape_penalty` (greedy only) adds α per
    distinct node shape, for shorter compact C code. `call_overhead` adds a fixed cost to
    every operation in the extraction objective (fewer, larger steps run faster in numpy). `structure=("lowrank",)` adds
    the low-rank factorizations of the tables as equalities (sum-product only: the
    result can give marginals, not modes or moments).

    `partition="search"` chooses the clusters by measured cost instead of `cluster_budget`
    (see partition.py): merges, then `exchange` moves, then `boundary` choices (messages
    sent as pieces, which cluster computes a marginal), within `partition_time_s` and at most
    `max_cluster_vars` variables per cluster.
    """
    if structure:
        return _optimize_structured(fg, extractor, max_iters, node_limit, rules, cluster_budget, seed,
                                    time_limit_s, strategy, structure, call_overhead, shape_penalty)
    if partition == "search":
        return _optimize_searched(fg, extractor, max_iters, node_limit, rules, time_limit_s, strategy, call_overhead,
                                  shape_penalty, partition_time_s, max_cluster_vars, exchange, boundary)
    if partition != "budget":
        raise ValueError(f"unknown partition {partition!r}")
    jt = junction_tree(fg)
    problems = local_problems(fg, jt, clusters(jt, cluster_budget), seed)
    sats, exs = [], []
    for p in problems:
        sat, best = run_strategy(
            fg, p.queries, strategy, max_iters=max_iters, node_limit=node_limit, rules=rules,
            inputs=p.inputs, seeds=p.seeds,
        )
        sats.append(sat)
        ex = _extract(sat.graph, fg, extractor, time_limit_s, overhead=call_overhead, shape_penalty=shape_penalty)
        if best is not None and best.cost < ex.cost:  # restart remembers its best round
            ex = best
        exs.append(ex)
    dag = stitch(problems, [e.dag for e in exs])
    optimal = None if any(e.optimal is None for e in exs) else all(e.optimal for e in exs)
    ex = Extraction(dag, dag_cost(dag, fg), optimal, sum(e.seconds for e in exs))
    return OptimizeResult(sats, ex, [p.variables for p in problems], exs)


def _optimize_searched(fg, extractor, max_iters, node_limit, rules, time_limit_s, strategy, call_overhead,
                       shape_penalty, partition_time_s, max_cluster_vars, exchange, boundary) -> OptimizeResult:
    from .partition import search_partition

    solve_kw = dict(strategy=strategy, max_iters=max_iters, node_limit=node_limit, rules=rules,
                    time_limit_s=time_limit_s)
    extract = lambda g: _extract(g, fg, extractor, time_limit_s, overhead=call_overhead,  # noqa: E731
                                 shape_penalty=shape_penalty)
    dag, cur, info = search_partition(fg, solve_kw, extract, partition_time_s, max_cluster_vars, boundary, exchange)
    exs = [s.extraction for s in cur.solved]
    sats = [SaturationResult(None, 0, s.hit_limit, s.seconds, s.nodes) for s in cur.solved]
    ex = Extraction(dag, dag_cost(dag, fg), None, sum(e.seconds for e in exs))
    return OptimizeResult(sats, ex, [p.variables for p in cur.problems], exs, partition=info)


def _optimize_structured(fg, extractor, max_iters, node_limit, rules, cluster_budget, seed, time_limit_s,
                         strategy, structure, call_overhead=0, shape_penalty=0.0) -> OptimizeResult:
    from .baselines import junction_tree_terms
    from .ir import all_marginal_queries
    from .structure import low_rank

    unknown = set(structure) - {"lowrank"}
    if unknown:
        raise ValueError(f"unknown structure {sorted(unknown)}")
    if cluster_budget is not None:
        raise ValueError("structure equalities are not combined with cluster_budget")
    st = low_rank(fg)
    queries = all_marginal_queries(fg)
    seed_fgs = [fg] + ([st.replaced] if st.replaced is not None else [])
    seeds = junction_tree_terms(fg)[0] if seed else None
    sat, best = run_strategy(st.fg, queries, strategy, max_iters=max_iters, node_limit=node_limit, rules=rules,
                             seeds=seeds, equalities=st.equalities, seed_fgs=seed_fgs)
    ex = _extract(sat.graph, st.fg, extractor, time_limit_s, overhead=call_overhead, shape_penalty=shape_penalty)
    if best is not None and best.cost < ex.cost:
        ex = best
    return OptimizeResult([sat], ex, [frozenset(fg.variables())], [ex], st.fg, bool(st.equalities), st)


def _eval_fg(fg: FactorGraph, res: OptimizeResult) -> FactorGraph:
    return res.eval_fg if res.eval_fg is not None else fg


def _require_general(res: OptimizeResult, what: str) -> None:
    if res.sum_product_only:
        raise ValueError(f"{what} needs a computation valid in every semiring; this one uses sum-only equalities")


def marginals(fg: FactorGraph, res: OptimizeResult) -> dict[str, np.ndarray]:
    tables, _ = evaluate(res.extraction.dag, _eval_fg(fg, res), SUM_PRODUCT)
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
    _require_general(res, "mode")
    fixed: dict[str, int] = {}
    for v in fg.variables():
        tables, _ = evaluate(res.extraction.dag, fg, _Clamped(fixed))
        fixed[v] = int(np.argmax(tables[v].data))
    return fixed


def moments(fg: FactorGraph, res: OptimizeResult, var: str, values: np.ndarray) -> tuple[float, float]:
    """Mean and variance of values[x_var] under the normalized distribution."""
    _require_general(res, "moments")
    sr = ExpectationSemiring(value_feature(fg, var, values))
    tables, _ = evaluate(res.extraction.dag, fg, sr)
    p, r, s = (float(tables[var].data[i].sum()) for i in range(3))
    E = r / p
    return E, s / p - E * E

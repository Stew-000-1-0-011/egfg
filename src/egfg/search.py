"""Search strategies: in which order and how often the rewrite rules are applied.

`saturate` (in egraph.py) applies every rule once per iteration (breadth first).
Here the same rules are scheduled differently:

- bfs: the same as `saturate`.
- staged: per round, apply the sum-pushing rule a few times (scope analysis
  saturated in between), then the reordering rules (commutativity, associativity,
  sum swap) once, then the reverse distributivity once (rule set "full" only).
- backoff: egglog's backoff scheduler (rules that match too often rest for a while).
- seeds: union the junction trees of several elimination orders with the queries.
- restart: grow a small e-graph, extract the cheapest computation, rebuild the
  e-graph from it, repeat (a local search around the current best).

Strategies can be combined with seeds ("seeds+staged", "seeds+restart").
Optionally a trace records (nodes, seconds, cost of a greedy extraction) after
every step, to measure how good the e-graph is when the search is stopped early.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from egglog import back_off, ruleset, run

from .baselines import candidate_orders, junction_tree_terms
from .egraph import EGraphData, _export, _rules, _size, build_egraph, rule_groups
from .extract import Extraction, extract_dag_greedy, extract_tree
from .ir import Dag, Input, Leaf, Mul, Sum, Term
from .model import FactorGraph

STRATEGIES = ("bfs", "staged", "backoff", "seeds", "restart", "seeds+staged", "seeds+restart")


@dataclass
class SearchConfig:
    strategy: str = "bfs"
    rules: str = "full"
    node_limit: int = 50_000
    max_steps: int = 30  # iterations (bfs, backoff) or rounds (staged, restart)
    time_limit_s: float = 120
    match_limit: int = 1000  # backoff
    ban_length: int = 5  # backoff
    push_steps: int = 3  # staged: sum-pushing steps per round
    backoff_chunk: int = 8  # backoff steps per scheduled run (the backoff state lives within one run)
    restart_fraction: float = 0.2  # restart: node budget of one round, as a fraction of node_limit
    n_random_orders: int = 4  # seeds


@dataclass
class TracePoint:
    step: int
    nodes: int
    seconds: float  # search time so far (tracing excluded)
    cost: int  # greedy DAG extraction of the e-graph at that point


@dataclass
class SearchResult:
    graph: EGraphData
    hit_limit: bool
    steps: int
    seconds: float
    num_nodes: int
    trace: list[TracePoint] = field(default_factory=list)
    best: Extraction | None = None  # restart keeps the best extraction over its rounds


def jt_seed_sets(fg: FactorGraph, n_random: int = 4) -> list[dict[str, Term]]:
    """Junction-tree terms for several elimination orders (one seed set per order)."""
    return [junction_tree_terms(fg, order)[0] for order in candidate_orders(fg, n_random)]


def dag_to_terms(dag: Dag) -> dict[str, Term]:
    memo: dict[str, Term] = {}

    def go(nid: str) -> Term:
        if nid not in memo:
            n = dag.nodes[nid]
            if n.op == "leaf":
                memo[nid] = Leaf(n.arg)
            elif n.op == "input":
                memo[nid] = Input(n.arg)
            elif n.op == "mul":
                memo[nid] = Mul(go(n.children[0]), go(n.children[1]))
            else:
                memo[nid] = Sum(n.arg, go(n.children[0]))
        return memo[nid]

    return {q: go(nid) for q, nid in dag.roots.items()}


class _Clock:
    """Search time, excluding the time spent on tracing."""

    def __init__(self):
        self.start = time.perf_counter()
        self.paused = 0.0

    def now(self) -> float:
        return time.perf_counter() - self.start - self.paused


def search(
    fg: FactorGraph,
    queries: dict[str, Term],
    config: SearchConfig | None = None,
    inputs: dict[str, frozenset[str]] | None = None,
    seeds: dict[str, Term] | None = None,
    trace: bool = False,
) -> SearchResult:
    cfg = config or SearchConfig()
    if cfg.strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {cfg.strategy!r}; expected one of {STRATEGIES}")
    clock = _Clock()
    seed_sets: list[dict[str, Term]] = [seeds] if seeds else []
    base = cfg.strategy
    if cfg.strategy.startswith("seeds"):
        seed_sets += jt_seed_sets(fg, cfg.n_random_orders)
        base = cfg.strategy.split("+")[1] if "+" in cfg.strategy else "bfs"
    points: list[TracePoint] = []

    def record(eg, step) -> None:
        if not trace:
            return
        t0 = time.perf_counter()
        g = _export(eg, queries, inputs or {}, seed_sets)
        points.append(TracePoint(step, sum(len(v) for v in g.classes.values()), round(clock.now(), 4),
                                 extract_dag_greedy(g, fg).cost))
        clock.paused += time.perf_counter() - t0

    if base == "restart":
        return _restart(fg, queries, cfg, inputs, seed_sets, clock, trace, points)

    eg = build_egraph(fg, queries, inputs, seed_sets)
    record(eg, 0)
    steps, hit = _grow(eg, base, cfg, clock, lambda s: record(eg, s), cfg.node_limit)
    g = _export(eg, queries, inputs or {}, seed_sets)
    return SearchResult(g, hit, steps, clock.now(), sum(len(v) for v in g.classes.values()), points)


def _grow(eg, base: str, cfg: SearchConfig, clock: _Clock, after_step, node_limit: int) -> tuple[int, bool]:
    """Apply the rules to `eg` with schedule `base` until saturation or a limit; (steps, hit_limit)."""
    steps = 0
    if base == "bfs":
        eg.register(*_rules(cfg.rules))
        while steps < cfg.max_steps and clock.now() < cfg.time_limit_s:
            report = eg.run(1)
            steps += 1
            after_step(steps)
            if _size(eg) > node_limit:
                return steps, True
            if not report.updated:
                break
        return steps, False
    groups = rule_groups(cfg.rules)
    scope_rs = ruleset(*groups["scope"])
    if base == "staged":
        push_rs = ruleset(*groups["push"])
        reorder_rs = ruleset(*groups["reorder"])
        pull_rs = ruleset(*groups["pull"]) if groups["pull"] else None
        push = run(scope_rs).saturate() + run(push_rs)
        phases = [push] * cfg.push_steps + [run(scope_rs).saturate() + run(reorder_rs)]
        if pull_rs is not None:
            phases.append(run(scope_rs).saturate() + run(pull_rs))
        while steps < cfg.max_steps and clock.now() < cfg.time_limit_s:
            before = _size(eg)
            for ph in phases:  # check the limit after every phase, not only after the round
                eg.run(ph)
                if _size(eg) > node_limit:
                    eg.run(run(scope_rs).saturate())
                    steps += 1
                    after_step(steps)
                    return steps, True
            eg.run(run(scope_rs).saturate())
            steps += 1
            after_step(steps)
            if _size(eg) == before:
                break
        return steps, False
    if base == "backoff":
        rest = ruleset(*(groups["push"] + groups["reorder"] + groups["pull"]))
        bo = back_off(match_limit=cfg.match_limit, ban_length=cfg.ban_length)
        unchanged = 0
        while steps < cfg.max_steps and clock.now() < cfg.time_limit_s:
            before = _size(eg)
            chunk = (run(scope_rs).saturate() + run(rest, scheduler=bo)) * cfg.backoff_chunk
            eg.run(chunk + run(scope_rs).saturate())
            steps += cfg.backoff_chunk
            after_step(steps)
            if _size(eg) > node_limit:
                return steps, True
            # banned rules can leave a whole chunk unchanged: stop only after two such chunks
            unchanged = unchanged + 1 if _size(eg) == before else 0
            if unchanged >= 2:
                break
        return steps, False
    raise ValueError(f"unknown schedule {base!r}")


def _restart(fg, queries, cfg, inputs, seed_sets, clock, trace, points) -> SearchResult:
    budget = max(1, int(cfg.node_limit * cfg.restart_fraction))
    current: list[dict[str, Term]] = list(seed_sets)
    best: Extraction | None = None
    g = None
    hit_any = False
    rounds = 0
    while rounds < cfg.max_steps and clock.now() < cfg.time_limit_s:
        eg = build_egraph(fg, queries, inputs, current)
        _, hit = _grow(eg, "bfs", SearchConfig(rules=cfg.rules, max_steps=30, time_limit_s=cfg.time_limit_s),
                       clock, lambda s: None, budget)
        hit_any = hit_any or hit
        g = _export(eg, queries, inputs or {}, current)
        ex = extract_dag_greedy(g, fg, time_limit_s=max(1.0, cfg.time_limit_s - clock.now()))
        rounds += 1
        if trace:
            points.append(TracePoint(rounds, sum(len(v) for v in g.classes.values()), round(clock.now(), 4),
                                     ex.cost))
        improved = best is None or ex.cost < best.cost
        if improved:
            best = ex
        if not improved or not hit:
            break  # no progress, or the e-graph saturated within the budget
        # rebuild from the current best computation (and the original seeds)
        current = list(seed_sets) + [dag_to_terms(best.dag)]
    return SearchResult(g, hit_any, rounds, clock.now(), sum(len(v) for v in g.classes.values()), points, best)


def extract_result(res: SearchResult, fg: FactorGraph, extractor: str = "greedy", time_limit_s: float = 60):
    """Final extraction; for restart, never worse than the best computation found in its rounds."""
    from .extract import extract_dag_ilp

    if extractor == "greedy":
        ex = extract_dag_greedy(res.graph, fg, time_limit_s=time_limit_s)
    elif extractor == "tree":
        ex = extract_tree(res.graph, fg)
    else:
        ex = extract_dag_ilp(res.graph, fg, time_limit_s=time_limit_s)
    if res.best is not None and res.best.cost < ex.cost:
        return res.best
    return ex


DEFAULT_STRATEGY = "seeds+staged"


def run_strategy(
    fg: FactorGraph,
    queries: dict[str, Term],
    strategy: str = DEFAULT_STRATEGY,
    max_iters: int = 30,
    node_limit: int = 50_000,
    rules: str = "full",
    inputs: dict[str, frozenset[str]] | None = None,
    seeds: dict[str, Term] | None = None,
    time_limit_s: float = 120,
):
    """Saturate with a strategy; returns (SaturationResult, best extraction found on the way or None).

    "bfs" is exactly `saturate`. The junction-tree seeds of the "seeds" strategies are
    built from the factor graph, which only describes the whole problem when there are
    no Input leaves; for local problems (clusters, time steps) "seeds+X" therefore
    becomes X with the problem's own seeds.
    """
    from .egraph import SaturationResult, saturate

    if inputs and strategy.startswith("seeds"):
        strategy = strategy.split("+")[1] if "+" in strategy else "bfs"
    if strategy == "bfs":
        return saturate(fg, queries, max_iters=max_iters, node_limit=node_limit, rules=rules,
                        inputs=inputs, seeds=seeds), None
    cfg = SearchConfig(strategy=strategy, rules=rules, node_limit=node_limit, max_steps=max_iters,
                       time_limit_s=time_limit_s)
    res = search(fg, queries, cfg, inputs=inputs, seeds=seeds)
    return SaturationResult(res.graph, res.steps, res.hit_limit, res.seconds, res.num_nodes), res.best

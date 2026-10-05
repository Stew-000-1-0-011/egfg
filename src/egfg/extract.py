"""Extraction from an e-graph: greedy tree extraction and ILP DAG extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pulp

from .cost import dag_cost, node_cost
from .egraph import EGraphData
from .ir import Dag, ENode
from .model import FactorGraph


@dataclass
class Extraction:
    dag: Dag
    cost: int
    optimal: bool | None
    seconds: float


def class_scopes(g: EGraphData, fg: FactorGraph) -> dict[str, frozenset[str]]:
    scopes: dict[str, frozenset[str]] = {}
    changed = True
    while changed:
        changed = False
        for cid, nodes in g.classes.items():
            if cid in scopes:
                continue
            for n in nodes:
                if all(c in scopes for c in n.children):
                    if n.op == "leaf":
                        s = frozenset(fg.factor(n.arg).scope)
                    elif n.op == "mul":
                        s = scopes[n.children[0]] | scopes[n.children[1]]
                    else:
                        s = scopes[n.children[0]] - {n.arg}
                    scopes[cid] = s
                    changed = True
                    break
    return scopes


def _costs(g: EGraphData, fg: FactorGraph, scopes) -> dict[tuple[str, int], int]:
    return {
        (cid, i): node_cost(n, scopes[cid], [scopes[c] for c in n.children], fg)
        for cid, nodes in g.classes.items()
        for i, n in enumerate(nodes)
    }


def _reachable_classes(g: EGraphData) -> list[str]:
    seen: set[str] = set()
    stack = list(g.roots.values())
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        for n in g.classes[cid]:
            stack.extend(n.children)
    return sorted(seen)


def _build(g: EGraphData, fg: FactorGraph, choice: dict[str, ENode], optimal, start) -> Extraction:
    nodes: dict[str, ENode] = {}
    stack = list(g.roots.values())
    while stack:
        cid = stack.pop()
        if cid in nodes:
            continue
        nodes[cid] = choice[cid]
        stack.extend(choice[cid].children)
    dag = Dag(nodes, dict(g.roots))
    return Extraction(dag, dag_cost(dag, fg), optimal, time.perf_counter() - start)


def extract_tree(g: EGraphData, fg: FactorGraph) -> Extraction:
    """Pick, per e-class, the node minimizing own cost + children's tree costs (fixpoint)."""
    start = time.perf_counter()
    scopes = class_scopes(g, fg)
    costs = _costs(g, fg, scopes)
    best: dict[str, tuple[float, ENode]] = {}
    changed = True
    while changed:
        changed = False
        for cid, nodes in g.classes.items():
            for i, n in enumerate(nodes):
                if not all(c in best for c in n.children):
                    continue
                total = costs[(cid, i)] + sum(best[c][0] for c in n.children)
                if cid not in best or total < best[cid][0]:
                    best[cid] = (total, n)
                    changed = True
    return _build(g, fg, {cid: n for cid, (_, n) in best.items()}, None, start)


def extract_dag_ilp(g: EGraphData, fg: FactorGraph, time_limit_s: float = 60) -> Extraction:
    """Choose one node per needed e-class minimizing the shared (DAG) cost."""
    start = time.perf_counter()
    scopes = class_scopes(g, fg)
    costs = _costs(g, fg, scopes)
    cids = _reachable_classes(g)
    prob = pulp.LpProblem("dag_extraction", pulp.LpMinimize)
    x = {
        (cid, i): prob.add_variable(f"x_{k}_{i}", cat="Binary")
        for k, cid in enumerate(cids)
        for i in range(len(g.classes[cid]))
    }
    a = {cid: prob.add_variable(f"a_{k}", cat="Binary") for k, cid in enumerate(cids)}
    prob += pulp.lpSum(costs[key] * var for key, var in x.items())
    for cid in set(g.roots.values()):
        prob += a[cid] == 1
    for cid in cids:
        prob += pulp.lpSum(x[(cid, i)] for i in range(len(g.classes[cid]))) == a[cid]
        for i, n in enumerate(g.classes[cid]):
            for ch in n.children:
                prob += x[(cid, i)] <= a[ch]
    # The tree extraction is acyclic and always valid: it is the warm start for
    # every solve and the fallback whenever the solver runs out of time.
    fallback = extract_tree(g, fg)
    warm = fallback.dag.nodes

    def set_warm_start() -> None:
        for (cid, i), var in x.items():
            var.setInitialValue(1 if cid in warm and warm[cid] == g.classes[cid][i] else 0)
        for cid, var in a.items():
            var.setInitialValue(1 if cid in warm else 0)

    def give_up() -> Extraction:
        return Extraction(fallback.dag, fallback.cost, False, time.perf_counter() - start)

    # Acyclicity by lazy cuts: solve, and while the chosen nodes form a cycle,
    # forbid choosing that whole cycle again (instead of big-M ordering variables).
    optimal = True
    while True:
        remaining = time_limit_s - (time.perf_counter() - start)
        if remaining <= 0:
            return give_up()
        set_warm_start()
        stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=remaining, warmStart=True))
        if not stats.has_solution:
            return give_up()
        optimal = optimal and stats.status == pulp.LpSolveStatus.Optimal
        picked = {
            cid: i
            for cid in cids
            for i in range(len(g.classes[cid]))
            if x[(cid, i)].value() is not None and x[(cid, i)].value() > 0.5
        }
        cycle = _find_cycle(g, picked)
        if cycle is None:
            break
        prob += pulp.lpSum(x[(cid, picked[cid])] for cid in cycle) <= len(cycle) - 1
    choice = {cid: g.classes[cid][i] for cid, i in picked.items()}
    result = _build(g, fg, choice, optimal, start)
    return result if result.cost <= fallback.cost else give_up()


def _find_cycle(g: EGraphData, picked: dict[str, int]) -> list[str] | None:
    """Return the e-classes of one cycle among the picked nodes, or None."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {cid: WHITE for cid in picked}
    for root in picked:
        if color[root] != WHITE:
            continue
        path: list[str] = []
        stack = [(root, iter(g.classes[root][picked[root]].children))]
        color[root] = GREY
        path.append(root)
        while stack:
            cid, it = stack[-1]
            ch = next(it, None)
            if ch is None:
                color[cid] = BLACK
                stack.pop()
                path.pop()
            elif color.get(ch) == GREY:
                return path[path.index(ch) :]
            elif color.get(ch) == WHITE:
                color[ch] = GREY
                path.append(ch)
                stack.append((ch, iter(g.classes[ch][picked[ch]].children)))
    return None

"""Representation-aware extraction.

The e-graph says which computations are equal; this module also chooses how each
chosen value is represented. The unit of choice is a state (e-class, representation).
A state is produced by an implementation: an e-node computed from child states by
some physical operation, or a conversion from another state of the same e-class.
The caller lists every implementation with its cost; tree, greedy-DAG and ILP
extraction then work on states exactly as the plain extractors work on e-classes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pulp

from .egraph import EGraphData
from .ir import ENode

State = tuple[str, str]  # (e-class, representation)


@dataclass(frozen=True)
class Impl:
    cid: str
    out: str
    node: ENode | None  # None for a conversion
    kids: tuple[State, ...]
    cost: int
    op: str  # name of the physical operation (for reports and evaluation)

    @property
    def state(self) -> State:
        return (self.cid, self.out)


@dataclass
class RepChoice:
    choice: dict[State, Impl]  # the implementation of every state reachable from the roots
    roots: dict[str, State]
    cost: int
    weighted_cost: int
    optimal: bool | None
    seconds: float


def _w(impl: Impl, weights) -> int:
    return impl.cost * (weights.get(impl.cid, 1) if weights else 1)


def _reach(roots, choice: dict[State, Impl], weights) -> int | None:
    """Weighted DAG cost of the states reachable from `roots`; None on a cycle or a missing state."""
    state: dict[State, int] = {}
    total = 0
    for root in roots:
        if root in state:
            continue
        if root not in choice:
            return None
        state[root] = 1
        stack = [(root, iter(choice[root].kids))]
        while stack:
            s, it = stack[-1]
            k = next(it, None)
            if k is None:
                state[s] = 2
                total += _w(choice[s], weights)
                stack.pop()
            elif k not in state:
                if k not in choice:
                    return None
                state[k] = 1
                stack.append((k, iter(choice[k].kids)))
            elif state[k] == 1:
                return None
    return total


def _restrict(choice: dict[State, Impl], roots) -> dict[State, Impl]:
    out: dict[State, Impl] = {}
    stack = list(roots)
    while stack:
        s = stack.pop()
        if s in out:
            continue
        out[s] = choice[s]
        stack.extend(choice[s].kids)
    return out


def _tree(impls_of: dict[State, list[Impl]], weights, allowed=None, deadline: float | None = None) -> dict[State, Impl]:
    """Per state, the implementation minimizing own cost + children's tree costs (fixpoint)."""
    best: dict[State, tuple[int, Impl]] = {}
    changed = True
    while changed:
        changed = False
        if deadline is not None and time.perf_counter() > deadline and best:
            break  # out of time: keep what the passes so far found (every state reached keeps a valid choice)
        for s, impls in impls_of.items():
            for im in impls:
                if allowed is not None and not allowed(im):
                    continue
                if not all(k in best for k in im.kids):
                    continue
                total = _w(im, weights) + sum(best[k][0] for k in im.kids)
                if s not in best or total < best[s][0]:
                    best[s] = (total, im)
                    changed = True
    return {s: im for s, (_, im) in best.items()}


def _start(g: EGraphData, impls_of, roots, weights, deadline: float | None = None) -> dict[State, Impl]:
    tree = _tree(impls_of, weights, deadline=deadline)
    if not g.seed:
        return tree
    # the seed: in seed e-classes use only the seed node (or conversions)
    seeded = _tree(impls_of, weights, lambda im: im.node is None or im.cid not in g.seed or im.node == g.seed[im.cid],
                   deadline=deadline)
    seeded = {**tree, **seeded}
    a, b = _reach(roots, tree, weights), _reach(roots, seeded, weights)
    if b is not None and (a is None or b <= a):
        return seeded
    return tree


def _result(choice, roots_named, weights, optimal, start) -> RepChoice:
    roots = list(roots_named.values())
    choice = _restrict(choice, roots)
    return RepChoice(choice, dict(roots_named), _reach(roots, choice, None), _reach(roots, choice, weights),
                     optimal, time.perf_counter() - start)


def extract_rep(
    g: EGraphData,
    impls_of: dict[State, list[Impl]],
    roots: dict[str, State],
    method: str = "tree",
    weights: dict[str, int] | None = None,
    time_limit_s: float = 60,
) -> RepChoice:
    start = time.perf_counter()
    rs = list(dict.fromkeys(roots.values()))
    choice = _start(g, impls_of, rs, weights, start + time_limit_s)
    if _reach(rs, choice, weights) is None:
        raise ValueError("no implementation reaches every root (a required representation is unavailable)")
    if method == "tree":
        return _result(choice, roots, weights, None, start)
    if method == "greedy":
        return _result(_greedy(impls_of, rs, choice, weights, start, time_limit_s), roots, weights, None, start)
    if method == "ilp":
        return _ilp(impls_of, roots, choice, weights, start, time_limit_s)
    raise ValueError(f"unknown method {method!r}")


def _greedy(impls_of, roots, choice, weights, start, time_limit_s) -> dict[State, Impl]:
    choice = dict(choice)
    current = _reach(roots, choice, weights)
    improved = True
    while improved and time.perf_counter() - start < time_limit_s:
        improved = False
        for s in list(_restrict(choice, roots)):
            if time.perf_counter() - start >= time_limit_s:
                break
            keep = choice[s]
            for im in impls_of.get(s, []):
                if im == keep or not all(k in choice for k in im.kids):
                    continue
                choice[s] = im
                c = _reach(roots, choice, weights)
                if c is not None and c < current:
                    current, keep, improved = c, im, True
                choice[s] = keep
    return choice


def _ilp(impls_of, roots_named, start_choice, weights, start, time_limit_s) -> RepChoice:
    roots = list(dict.fromkeys(roots_named.values()))
    # states reachable through any implementation
    seen: set[State] = set()
    stack = list(roots)
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        for im in impls_of.get(s, []):
            stack.extend(im.kids)
    states = sorted(seen)
    impls = [im for s in states for im in impls_of.get(s, []) if all(k in seen for k in im.kids)]
    prob = pulp.LpProblem("rep_extraction", pulp.LpMinimize)
    x = {im: prob.add_variable(f"x_{k}", cat="Binary") for k, im in enumerate(impls)}
    a = {s: prob.add_variable(f"a_{k}", cat="Binary") for k, s in enumerate(states)}
    prob += pulp.lpSum(_w(im, weights) * v for im, v in x.items())
    for s in roots:
        prob += a[s] == 1
    by_state: dict[State, list[Impl]] = {s: [] for s in states}
    for im in impls:
        by_state[im.state].append(im)
    for s in states:
        prob += pulp.lpSum(x[im] for im in by_state[s]) == a[s]
    for im in impls:
        for k in im.kids:
            prob += x[im] <= a[k]
    fallback = _restrict(start_choice, roots)

    def give_up() -> RepChoice:
        r = _result(fallback, roots_named, weights, False, start)
        return r

    optimal = True
    while True:
        remaining = time_limit_s - (time.perf_counter() - start)
        if remaining <= 0:
            return give_up()
        for im, v in x.items():
            v.setInitialValue(1 if fallback.get(im.state) == im else 0)
        for s, v in a.items():
            v.setInitialValue(1 if s in fallback else 0)
        stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=remaining, warmStart=True))
        if not stats.has_solution:
            return give_up()
        optimal = optimal and stats.status == pulp.LpSolveStatus.Optimal
        picked = {im.state: im for im, v in x.items() if v.value() is not None and v.value() > 0.5}
        cycle = _cycle(picked)
        if cycle is None:
            break
        prob += pulp.lpSum(x[im] for im in cycle) <= len(cycle) - 1
    res = _result(picked, roots_named, weights, optimal, start)
    fb = _reach(roots, fallback, weights)
    return res if res.weighted_cost <= fb else give_up()


def _cycle(picked: dict[State, Impl]) -> list[Impl] | None:
    color: dict[State, int] = {}
    for root in picked:
        if root in color:
            continue
        path = [root]
        color[root] = 1
        stack = [(root, iter(picked[root].kids))]
        while stack:
            s, it = stack[-1]
            k = next(it, None)
            if k is None:
                color[s] = 2
                stack.pop()
                path.pop()
            elif k not in picked:
                continue
            elif color.get(k) == 1:
                return [picked[t] for t in path[path.index(k) :]]
            elif k not in color:
                color[k] = 1
                path.append(k)
                stack.append((k, iter(picked[k].kids)))
    return None

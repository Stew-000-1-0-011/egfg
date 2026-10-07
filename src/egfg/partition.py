"""Choose the partition by measured cost: merges, exchanges and the boundary messages.

A state is a partition of the clique tree into connected clusters, the clique each factor is
multiplied at, the clique (hence cluster) computing each marginal, and the crossing edges
whose message is sent as several pieces. Everything is keyed by cliques, so a state stays
valid whatever the clusters are. The cost of a state is the sum of the costs of its local
problems (the stitched computation shares nothing across clusters); every local problem is
solved once and remembered by its content.

Search (first improvement, priority order, strictly decreasing cost, so it always stops):

1. for the candidate clique trees (several elimination orders), merges from one cluster per
   clique (the junction tree computation); successive halving keeps the better half of the
   trees after every round of a few merges, until one is left;
2. exchanges: merges, moving a leaf clique of a cluster to its neighbour, moving a factor of a
   crossing separator to the other side (optionally: any factor to any cluster containing its
   scope, splitting a cluster along an inner edge);
3. boundary (optional): sending a message as pieces, moving the marginal of a variable to
   another cluster containing it.

With `jobs` > 1, the local problems of the next few candidates are solved in parallel first;
the candidates are then taken in the same order, so the accepted moves are the same.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context

from .cost import dag_cost
from .decompose import LocalProblem, local_problems, stitch
from .extract import Extraction
from .ir import to_dag
from .jtree import JunctionTree, junction_tree
from .model import FactorGraph
from .search import run_strategy

EXCHANGE = ("merge", "move", "factor")
PHASE_K_EXCHANGE = ("merge", "move", "factor_any", "split")  # phase K's moves
BOUNDARY = ("pieces", "owner")


@dataclass(frozen=True)
class State:
    parts: tuple[frozenset[int], ...]  # sorted by lowest clique
    home: tuple[tuple[int, int], ...]  # (factor id, clique)
    owner: tuple[tuple[str, int], ...]  # (variable, clique computing its marginal)
    split: frozenset[tuple[int, int]] = frozenset()  # crossing edges (i, k) sent as pieces

    def with_parts(self, parts) -> State:
        return replace(self, parts=tuple(sorted((frozenset(p) for p in parts), key=min)))


@dataclass
class Solved:
    extraction: Extraction
    hit_limit: bool
    nodes: int
    seconds: float


def solve_local(fg: FactorGraph, p: LocalProblem, solve_kw: dict, extract_kw: dict) -> Solved:
    """Saturate, extract (`extract_kw` for pipeline._extract), and never return worse than the
    junction tree terms (or the query itself where there is no seed)."""
    from .pipeline import _extract

    t0 = time.perf_counter()
    sat, best = run_strategy(fg, p.queries, inputs=p.inputs, seeds=p.seeds, **solve_kw)
    ex = _extract(sat.graph, fg, **extract_kw)
    if best is not None and best.cost < ex.cost:
        ex = best
    fallback = to_dag({q: p.seeds.get(q, t) for q, t in p.queries.items()}, p.inputs)
    fcost = dag_cost(fallback, fg)
    if fcost < ex.cost:
        ex = Extraction(fallback, fcost, None, ex.seconds)
    return Solved(ex, sat.hit_limit, sat.num_nodes, time.perf_counter() - t0)


_WORKER: dict = {}


def _init_worker(fg, solve_kw, extract_kw) -> None:
    _WORKER.update(fg=fg, solve_kw=solve_kw, extract_kw=extract_kw)


def _solve_in_worker(p: LocalProblem) -> Solved:
    return solve_local(_WORKER["fg"], p, _WORKER["solve_kw"], _WORKER["extract_kw"])


def _key(p: LocalProblem):
    return p.cliques, tuple(sorted(p.inputs.items())), tuple(sorted(p.queries.items(), key=lambda kv: kv[0]))


@dataclass
class Evaluated:
    cost: int
    hits: int
    problems: list[LocalProblem]
    solved: list[Solved]


@dataclass
class PartitionInfo:
    trees: int
    tree_costs: list[int]  # merge-phase cost of each candidate tree when it stopped
    phase_costs: dict[str, int]  # cost after each phase ("jt", "merge", "exchange", "boundary")
    tries: int = 0  # candidate states evaluated
    solves: int = 0  # local problems solved (not remembered)
    accepted: list[tuple[str, int]] = field(default_factory=list)  # (move kind, cost after it)
    seconds: float = 0.0
    state: State | None = None


class PartitionSearch:
    def __init__(self, fg: FactorGraph, solve_kw: dict, extract_kw: dict, max_cluster_vars: int = 12, jobs: int = 1):
        self.fg = fg
        self.solve_kw = solve_kw
        self.extract_kw = extract_kw
        self.max_vars = max_cluster_vars
        self.jobs = jobs
        self.pool = None
        if jobs > 1:
            self.pool = ProcessPoolExecutor(jobs, mp_context=get_context("spawn"), initializer=_init_worker,
                                            initargs=(fg, solve_kw, extract_kw))
        self.memo: dict = {}
        self.tries = 0
        self.solves = 0

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()

    # ---- one state -------------------------------------------------------------------
    def tree(self, jt: JunctionTree, state: State) -> JunctionTree:
        assigned: dict[int, list[int]] = {i: [] for i in range(len(jt.cliques))}
        for f, i in state.home:
            assigned[i].append(f)
        return JunctionTree(jt.cliques, jt.nbrs, {i: sorted(v) for i, v in assigned.items()})

    def problems(self, jt: JunctionTree, state: State) -> list[LocalProblem]:
        return local_problems(self.fg, self.tree(jt, state), list(state.parts), seed=True,
                              owner=dict(state.owner), split=state.split)

    def solve(self, p: LocalProblem) -> Solved:
        key = _key(p)
        if key not in self.memo:
            self.solves += 1
            self.memo[key] = solve_local(self.fg, p, self.solve_kw, self.extract_kw)
        return self.memo[key]

    def prefetch(self, jt: JunctionTree, states: list[State]) -> None:
        """Solve, in parallel, the local problems of `states` not solved yet."""
        todo: dict = {}
        for st in states:
            for p in self.problems(jt, st):
                k = _key(p)
                if k not in self.memo and k not in todo:
                    todo[k] = p
        if self.pool is None or len(todo) < 2:
            return
        for k, solved in zip(todo, self.pool.map(_solve_in_worker, todo.values())):
            self.solves += 1
            self.memo[k] = solved

    def evaluate(self, jt: JunctionTree, state: State) -> Evaluated:
        self.tries += 1
        problems = self.problems(jt, state)
        solved = [self.solve(p) for p in problems]
        return Evaluated(sum(s.extraction.cost for s in solved), sum(s.hit_limit for s in solved), problems, solved)

    # ---- moves -------------------------------------------------------------------------
    def nvars(self, jt: JunctionTree, cliques) -> int:
        return len(frozenset().union(*(jt.cliques[i] for i in cliques)))

    def moves(self, jt: JunctionTree, st: State, kinds) -> list[tuple[str, State]]:
        """Candidate states in priority order: heavy crossing separators first."""
        of = {i: c for c, cl in enumerate(st.parts) for i in cl}
        sep = lambda i, k: self.fg.size(jt.cliques[i] & jt.cliques[k])  # noqa: E731
        cross = sorted(((i, k) for i in range(len(jt.cliques)) for k in jt.nbrs[i] if of[i] != of[k]),
                       key=lambda e: (-sep(*e), e))
        out: list[tuple[int, tuple, str, State]] = []  # (rank group, priority, kind, state)
        parts = list(st.parts)
        if "merge" in kinds:
            seen = set()
            for i, k in cross:
                a, b = sorted((of[i], of[k]))
                if (a, b) in seen or self.nvars(jt, parts[a] | parts[b]) > self.max_vars:
                    continue
                seen.add((a, b))
                rest = [p for n, p in enumerate(parts) if n not in (a, b)]
                out.append((0, (-sep(i, k), i, k), "merge", st.with_parts(rest + [parts[a] | parts[b]])))
        if "move" in kinds:
            for i, k in cross:  # move clique i from its cluster to k's
                a, b = of[i], of[k]
                inner = [x for x in jt.nbrs[i] if of[x] == a]
                if len(parts[a]) < 2 or len(inner) > 1 or self.nvars(jt, parts[b] | {i}) > self.max_vars:
                    continue
                new = list(parts)
                new[a], new[b] = parts[a] - {i}, parts[b] | {i}
                out.append((1, (-sep(i, k), i, k), "move", st.with_parts(new)))
        if "factor" in kinds:  # a factor of a crossing separator, to the other side
            home = dict(st.home)
            seen = set()
            for i, k in cross:
                for f, h in st.home:
                    scope = set(self.fg.factor(f).scope)
                    if of[h] == of[i] and scope <= jt.cliques[i] & jt.cliques[k] and (f, of[k]) not in seen:
                        seen.add((f, of[k]))
                        new = dict(home)
                        new[f] = k
                        out.append((2, (-self.fg.size(scope), f, of[k]), "factor",
                                    replace(st, home=tuple(sorted(new.items())))))
        if "factor_any" in kinds:  # any factor, to any cluster containing its scope (phase K)
            home = dict(st.home)
            for f, h in st.home:
                scope = set(self.fg.factor(f).scope)
                for c, cl in enumerate(parts):
                    if c == of[h]:
                        continue
                    cands = [i for i in sorted(cl) if scope <= jt.cliques[i]]
                    if cands:
                        new = dict(home)
                        new[f] = cands[0]
                        out.append((2, (-self.fg.size(scope), f, c), "factor_any",
                                    replace(st, home=tuple(sorted(new.items())))))
        if "split" in kinds:
            for i in range(len(jt.cliques)):
                for k in jt.nbrs[i]:
                    if i < k and of[i] == of[k]:
                        side = _side(jt, parts[of[i]], k, i)
                        rest = [p for n, p in enumerate(parts) if n != of[i]]
                        out.append((3, (sep(i, k), i, k), "split", st.with_parts(rest + [side, parts[of[i]] - side])))
        if "pieces" in kinds:
            for i, k in cross:
                out.append((4, (-sep(i, k), i, k), "pieces", replace(st, split=st.split ^ {(i, k)})))
        if "owner" in kinds:
            owner = dict(st.owner)
            for v, o in st.owner:
                for c, cl in enumerate(parts):
                    cands = [i for i in sorted(cl) if v in jt.cliques[i]]
                    if c != of[o] and cands:
                        new = dict(owner)
                        new[v] = cands[0]
                        out.append((5, (v, c), "owner", replace(st, owner=tuple(sorted(new.items())))))
        # merges and moves by separator (heaviest first), then the other kinds
        out.sort(key=lambda x: (0, x[1][0], x[0], x[1]) if x[0] < 2 else (x[0], 0, 0, x[1]))
        return [(kind, s) for _, _, kind, s in out]

    # ---- local search ----------------------------------------------------------------
    def descend(self, jt: JunctionTree, st: State, cur: Evaluated, kinds, deadline: float, info: PartitionInfo,
                max_accept: int | None = None):
        """First improvement in priority order until no move improves, the deadline, or
        `max_accept` accepted moves. With a pool, the candidates are prefetched in windows."""
        window = 2 * self.jobs if self.pool is not None else 1
        accepted = 0
        improved = True
        while improved and time.perf_counter() < deadline and (max_accept is None or accepted < max_accept):
            improved = False
            cands = self.moves(jt, st, kinds)
            for start in range(0, len(cands), window):
                if time.perf_counter() >= deadline:
                    break
                chunk = cands[start:start + window]
                if window > 1:
                    self.prefetch(jt, [c for _, c in chunk])
                for kind, cand in chunk:
                    r = self.evaluate(jt, cand)
                    if r.cost < cur.cost and r.hits <= cur.hits:
                        st, cur, improved = cand, r, True
                        accepted += 1
                        info.accepted.append((kind, r.cost))
                        break
                if improved:
                    break
        return st, cur


def _side(jt: JunctionTree, members: frozenset[int], start: int, other: int) -> frozenset[int]:
    """The cliques of `members` reached from `start` without crossing to `other`."""
    seen, stack = {start}, [start]
    while stack:
        i = stack.pop()
        for k in jt.nbrs[i]:
            if k in members and k not in seen and not (i == start and k == other):
                seen.add(k)
                stack.append(k)
    return frozenset(seen)


def initial_state(fg: FactorGraph, jt: JunctionTree) -> State:
    """One cluster per clique: the junction tree computation."""
    home = tuple(sorted((f, i) for i, fs in jt.assigned.items() for f in fs))
    owner = tuple(sorted((v, min(i for i, c in enumerate(jt.cliques) if v in c)) for v in fg.variables()))
    return State(tuple(frozenset({i}) for i in range(len(jt.cliques))), home, owner)


def candidate_trees(fg: FactorGraph, n_random: int) -> list[JunctionTree]:
    from .baselines import candidate_orders

    out, seen = [], set()
    for order in candidate_orders(fg, n_random):
        jt = junction_tree(fg, order)
        key = frozenset(jt.cliques)
        if key not in seen:
            seen.add(key)
            out.append(jt)
    return out


def search_partition(fg: FactorGraph, solve_kw: dict, extract_kw: dict, time_s: float = 120,
                     max_cluster_vars: int = 12, boundary: bool = True, exchange: bool = True, n_random: int = 1,
                     trees: str = "halving", jobs: int = 1, moves: tuple[str, ...] = EXCHANGE, merges_per_round: int = 2):
    """Returns (Dag of all marginals, the final Evaluated, PartitionInfo).

    `trees`: "halving" (successive halving over the candidate trees), "all" (every tree to
    the end, phase K) or "first" (the min-fill tree only)."""
    if trees not in ("halving", "all", "first"):
        raise ValueError(f"unknown trees {trees!r}")
    t0 = time.perf_counter()
    deadline = t0 + time_s
    ps = PartitionSearch(fg, solve_kw, extract_kw, max_cluster_vars, jobs)
    try:
        cands = candidate_trees(fg, n_random)
        if trees == "first":
            cands = cands[:1]
        info = PartitionInfo(len(cands), [], {})
        merge_end = t0 + time_s * (0.5 if exchange or boundary else 1.0)
        runs = []  # [tree index, jt, state, evaluated, still merging]
        for n, jt in enumerate(cands):
            st = initial_state(fg, jt)
            ps.prefetch(jt, [st])
            runs.append([n, jt, st, ps.evaluate(jt, st), True])
        info.phase_costs["jt"] = runs[0][3].cost
        every = list(runs)
        if trees == "all":
            for n, run in enumerate(runs):
                share = t0 + (merge_end - t0) * (n + 1) / len(runs)
                run[2], run[3] = ps.descend(run[1], run[2], run[3], ("merge",), share, info)
        else:
            while len(runs) > 1 and time.perf_counter() < merge_end:
                left = merge_end - time.perf_counter()
                for n, run in enumerate(runs):
                    if run[4]:
                        before = len(info.accepted)
                        end = time.perf_counter() + left / len(runs)
                        run[2], run[3] = ps.descend(run[1], run[2], run[3], ("merge",), end, info, merges_per_round)
                        run[4] = len(info.accepted) - before == merges_per_round
                runs.sort(key=lambda r: (r[3].cost, r[0]))
                runs = runs[: math.ceil(len(runs) / 2)] if len(runs) > 1 else runs
            run = runs[0]
            run[2], run[3] = ps.descend(run[1], run[2], run[3], ("merge",), merge_end, info)
        info.tree_costs = [r[3].cost for r in every]  # where each tree stopped (dropped trees early)
        _, jt, st, cur, _ = min(runs, key=lambda r: (r[3].cost, r[0]))
        info.phase_costs["merge"] = cur.cost
        if exchange:
            end = deadline if not boundary else t0 + time_s * 0.8
            st, cur = ps.descend(jt, st, cur, moves, end, info)
            info.phase_costs["exchange"] = cur.cost
        if boundary:
            st, cur = ps.descend(jt, st, cur, moves + BOUNDARY, deadline, info)
            info.phase_costs["boundary"] = cur.cost
    finally:
        ps.close()
    info.tries, info.solves, info.seconds, info.state = ps.tries, ps.solves, time.perf_counter() - t0, st
    dag = stitch(cur.problems, [s.extraction.dag for s in cur.solved])
    return dag, cur, info

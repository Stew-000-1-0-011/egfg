"""Reference computations: brute force (this section) and comparison baselines."""

from __future__ import annotations

import numpy as np

from .model import FactorGraph



def brute_force_joint(fg: FactorGraph) -> np.ndarray:
    """Unnormalized joint; axes follow fg.variables()."""
    vs = fg.variables()
    operands = []
    for f in fg.factors:
        operands += [f.table, [vs.index(v) for v in f.scope]]
    return np.einsum(*operands, list(range(len(vs))))


def brute_force_marginals(fg: FactorGraph) -> dict[str, np.ndarray]:
    J = brute_force_joint(fg)
    Z = J.sum()
    vs = fg.variables()
    return {
        v: J.sum(axis=tuple(j for j in range(len(vs)) if j != i)) / Z for i, v in enumerate(vs)
    }


def brute_force_max_value(fg: FactorGraph) -> float:
    return float(brute_force_joint(fg).max())


def joint_value(fg: FactorGraph, assignment: dict[str, int]) -> float:
    val = 1.0
    for f in fg.factors:
        val *= float(f.table[tuple(assignment[v] for v in f.scope)])
    return val


def brute_force_moments(fg: FactorGraph, var: str, values: np.ndarray) -> tuple[float, float]:
    p = brute_force_marginals(fg)[var]
    E = float((p * values).sum())
    return E, float((p * values**2).sum() - E**2)


# ---------------------------------------------------------------------------
# Junction tree (on trees of pairwise factors this is Shafer-Shenoy BP)
# ---------------------------------------------------------------------------

from .ir import Dag, Leaf, Mul, Sum, Term, term_scope, to_dag  # noqa: E402

MessageSig = tuple[frozenset[int], frozenset[str]]


def min_fill_order(fg: FactorGraph) -> list[str]:
    adj: dict[str, set[str]] = {v: set() for v in fg.variables()}
    for f in fg.factors:
        for u in f.scope:
            adj[u] |= set(f.scope) - {u}
    order: list[str] = []
    while adj:
        def fill(v: str) -> int:
            nb = sorted(adj[v])
            return sum(1 for i, x in enumerate(nb) for y in nb[i + 1 :] if y not in adj[x])

        v = min(adj, key=lambda u: (fill(u), u))
        nb = adj.pop(v)
        for x in nb:
            adj[x] |= nb - {x}
            adj[x].discard(v)
        order.append(v)
    return order


def _cliques(fg: FactorGraph, order: list[str]) -> list[frozenset[str]]:
    adj: dict[str, set[str]] = {v: set() for v in fg.variables()}
    for f in fg.factors:
        for u in f.scope:
            adj[u] |= set(f.scope) - {u}
    raw: list[frozenset[str]] = []
    for v in order:
        nb = adj.pop(v)
        raw.append(frozenset(nb | {v}))
        for x in nb:
            adj[x] |= nb - {x}
            adj[x].discard(v)
    cliques: list[frozenset[str]] = []
    for i, c in enumerate(raw):
        if any(c < d for d in raw) or any(c == d for d in raw[:i]):
            continue
        cliques.append(c)
    return cliques


def _clique_tree(cliques: list[frozenset[str]]) -> dict[int, list[int]]:
    """Maximum-weight spanning tree on separator sizes (Kruskal, ties by index)."""
    edges = sorted(
        ((len(cliques[i] & cliques[j]), i, j) for i in range(len(cliques)) for j in range(i + 1, len(cliques))),
        key=lambda e: (-e[0], e[1], e[2]),
    )
    parent = list(range(len(cliques)))

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    nbrs: dict[int, list[int]] = {i: [] for i in range(len(cliques))}
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            nbrs[i].append(j)
            nbrs[j].append(i)
    return {i: sorted(n) for i, n in nbrs.items()}


def _product(terms: list[Term]) -> Term:
    if not terms:
        raise ValueError("empty product: a clique received no factors and no messages")
    out = terms[0]
    for t in terms[1:]:
        out = Mul(out, t)
    return out


def _sum_out(vars_: set[str], body: Term) -> Term:
    for x in sorted(vars_, reverse=True):
        body = Sum(x, body)
    return body


def junction_tree_dag(fg: FactorGraph, order: list[str] | None = None) -> tuple[Dag, list[MessageSig]]:
    order = min_fill_order(fg) if order is None else order
    cliques = _cliques(fg, order)
    nbrs = _clique_tree(cliques)
    assigned: dict[int, list[int]] = {i: [] for i in range(len(cliques))}
    for f in sorted(fg.factors, key=lambda f: f.id):
        i = min(i for i, c in enumerate(cliques) if set(f.scope) <= c)
        assigned[i].append(f.id)

    msgs: dict[tuple[int, int], tuple[Term, frozenset[int]]] = {}

    def message(i: int, j: int) -> tuple[Term, frozenset[int]]:
        if (i, j) not in msgs:
            parts: list[Term] = [Leaf(fid) for fid in assigned[i]]
            fids = set(assigned[i])
            for k in nbrs[i]:
                if k != j:
                    t, fk = message(k, i)
                    parts.append(t)
                    fids |= fk
            body = _product(parts)
            msgs[(i, j)] = (_sum_out(set(cliques[i]) - cliques[j], body), frozenset(fids))
        return msgs[(i, j)]

    terms: dict[str, Term] = {}
    for v in fg.variables():
        i = min(i for i, c in enumerate(cliques) if v in c)
        parts = [Leaf(fid) for fid in assigned[i]] + [message(k, i)[0] for k in nbrs[i]]
        terms[v] = _sum_out(set(cliques[i]) - {v}, _product(parts))
    for i in nbrs:  # make sure every directed message exists for the signature list
        for j in nbrs[i]:
            message(i, j)
    sigs = [(fids, term_scope(t, fg)) for (t, fids) in msgs.values()]
    return to_dag(terms), sigs


def opt_einsum_cost(fg: FactorGraph) -> int:
    """Σ over variables of opt_einsum's optimal per-marginal contraction cost (reference only)."""
    import opt_einsum

    letters = {v: chr(ord("a") + i) if i < 26 else chr(ord("A") + i - 26) for i, v in enumerate(fg.variables())}
    inputs = ",".join("".join(letters[v] for v in f.scope) for f in fg.factors)
    shapes = [f.table.shape for f in fg.factors]
    strategy = "optimal" if len(fg.factors) <= 8 else "dp"
    total = 0
    for v in fg.variables():
        _, info = opt_einsum.contract_path(f"{inputs}->{letters[v]}", *shapes, shapes=True, optimize=strategy)
        total += int(info.opt_cost)
    return total

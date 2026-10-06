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

from .ir import Dag, Leaf, Mul, Sum, Term, to_dag  # noqa: E402
from .jtree import TreeTerms, junction_tree, min_fill_order  # noqa: E402,F401

MessageSig = tuple[frozenset[int], frozenset[str]]


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


def junction_tree_terms(fg: FactorGraph, order: list[str] | None = None) -> tuple[dict[str, Term], list[MessageSig]]:
    """Marginal terms of the junction tree, and the signature of every directed message."""
    jt = junction_tree(fg, order)
    calc = TreeTerms(fg, jt)
    terms = {v: calc.marginal(v)[0] for v in fg.variables()}
    sigs = []
    for i in jt.nbrs:
        for j in jt.nbrs[i]:
            m = calc.message(i, j)
            if m is not None:
                sigs.append((m[2], m[1]))
    return terms, sigs


def junction_tree_dag(fg: FactorGraph, order: list[str] | None = None) -> tuple[Dag, list[MessageSig]]:
    terms, sigs = junction_tree_terms(fg, order)
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


def factor_graph_bp_dag(fg: FactorGraph) -> tuple[Dag, list[MessageSig]]:
    """Sum-product BP on an acyclic factor graph, written as shared terms.

    m(f→x) = Σ_{scope(f)∖x} f · Π_{y∈scope(f)∖x} μ(y→f),  μ(y→f) = Π_{g∋y, g≠f} m(g→y),
    p(x) ∝ Π_{f∋x} m(f→x). Products follow factor id / variable name order.
    Returns the Dag and the (factor ids, scope) signature of every m(f→x).
    """
    nvars = len(fg.cards)
    parent = list(range(nvars + len(fg.factors)))
    vidx = {v: i for i, v in enumerate(fg.variables())}

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for k, f in enumerate(fg.factors):
        for v in f.scope:
            ru, rv = find(nvars + k), find(vidx[v])
            if ru == rv:
                raise ValueError("factor graph has a loop; BP would not be exact")
            parent[ru] = rv

    factors_of = {v: sorted(f.id for f in fg.factors if v in f.scope) for v in fg.variables()}
    m_memo: dict[tuple[int, str], tuple[Term, frozenset[int]]] = {}

    def mu(y: str, fid: int) -> tuple[Term | None, frozenset[int]]:
        parts, fids = [], frozenset()
        for g in factors_of[y]:
            if g != fid:
                t, fg_ids = m(g, y)
                parts.append(t)
                fids |= fg_ids
        return (_product(parts) if parts else None), fids

    def m(fid: int, x: str) -> tuple[Term, frozenset[int]]:
        if (fid, x) not in m_memo:
            scope = fg.factor(fid).scope
            parts: list[Term] = [Leaf(fid)]
            fids = frozenset({fid})
            for y in sorted(set(scope) - {x}):
                t, ids = mu(y, fid)
                if t is not None:
                    parts.append(t)
                    fids |= ids
            m_memo[(fid, x)] = (_sum_out(set(scope) - {x}, _product(parts)), fids)
        return m_memo[(fid, x)]

    terms = {v: _product([m(fid, v)[0] for fid in factors_of[v]]) for v in fg.variables()}
    sigs = [(ids, frozenset({x})) for (fid, x), (_, ids) in m_memo.items()]
    return to_dag(terms), sigs

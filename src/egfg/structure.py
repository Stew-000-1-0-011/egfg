"""Equalities that elimination orders cannot express: low-rank factorization of factor tables.

If a factor's table, viewed as a matrix (left variables × right variables), has
rank r, then φ(left, right) = Σ_z U(left, z) · V(z, right) with a new variable z
of r states. The equality `Leaf(φ) = Σ_z (Leaf(U) · Leaf(V))` lets the e-graph
compute through z when that is cheaper. It holds for sum-product only (it sums
over z), and U, V may have negative entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .ir import Leaf, Mul, Sum, Term
from .model import Factor, FactorGraph


@dataclass
class Split:
    fid: int
    left: tuple[str, ...]
    right: tuple[str, ...]
    z: str
    rank: int
    u: int  # factor id of U(left, z)
    v: int  # factor id of V(z, right)


@dataclass
class Structured:
    fg: FactorGraph  # the original factors plus every U, V, and every new variable
    equalities: list[tuple[Term, Term]]
    splits: list[Split]
    replaced: FactorGraph | None  # each splittable factor replaced by its most compact split


def _bipartitions(scope: tuple[str, ...]):
    """(left, right) with both non-empty; `left` always holds scope[0] (no mirrored duplicates)."""
    rest = scope[1:]
    for k in range(0, len(rest)):
        for extra in combinations(rest, k):
            left = (scope[0],) + extra
            right = tuple(v for v in scope if v not in left)
            if right:
                yield left, right


def low_rank(fg: FactorGraph, tol: float = 1e-9) -> Structured:
    cards = dict(fg.cards)
    factors = list(fg.factors)
    next_id = max(f.id for f in fg.factors) + 1
    splits: list[Split] = []
    best_for: dict[int, tuple[int, Split]] = {}
    for f in fg.factors:
        if len(f.scope) < 2:
            continue
        for j, (left, right) in enumerate(_bipartitions(f.scope)):
            perm = [f.scope.index(v) for v in left + right]
            rows = int(np.prod([cards[v] for v in left]))
            cols = int(np.prod([cards[v] for v in right]))
            M = np.transpose(f.table, perm).reshape(rows, cols)
            u, s, vt = np.linalg.svd(M, full_matrices=False)
            r = int(np.sum(s > tol * s[0])) if s[0] > 0 else 0
            if r == 0 or r >= min(rows, cols):
                continue
            z = f"_z{f.id}_{j}"
            cards[z] = r
            U = (u[:, :r] * s[:r]).reshape([cards[v] for v in left] + [r])
            V = vt[:r, :].reshape([r] + [cards[v] for v in right])
            sp = Split(f.id, left, right, z, r, next_id, next_id + 1)
            factors += [Factor(sp.u, left + (z,), U), Factor(sp.v, (z,) + right, V)]
            next_id += 2
            splits.append(sp)
            size = (rows + cols) * r
            if f.id not in best_for or size < best_for[f.id][0]:
                best_for[f.id] = (size, sp)
    ext = FactorGraph(cards, factors, allow_negative=True)
    eqs = [(Leaf(sp.fid), Sum(sp.z, Mul(Leaf(sp.u), Leaf(sp.v)))) for sp in splits]
    replaced = None
    if best_for:
        chosen = {fid: sp for fid, (_, sp) in best_for.items()}
        keep = [f for f in fg.factors if f.id not in chosen]
        extra = [ext.factor(i) for sp in chosen.values() for i in (sp.u, sp.v)]
        rcards = dict(fg.cards) | {sp.z: sp.rank for sp in chosen.values()}
        replaced = FactorGraph(rcards, keep + extra, allow_negative=True)
    return Structured(ext, eqs, splits, replaced)


def low_rank_tables(fg: FactorGraph, rank: int, seed: int = 0) -> FactorGraph:
    """The same graph with every 2-variable table replaced by a non-negative rank-`rank` product."""
    rng = np.random.default_rng(seed + 110_000)
    out = []
    for f in fg.factors:
        if len(f.scope) == 2:
            a, b = (fg.cards[v] for v in f.scope)
            table = rng.uniform(0.1, 1.0, (a, rank)) @ rng.uniform(0.1, 1.0, (rank, b))
            out.append(Factor(f.id, f.scope, table))
        else:
            out.append(f)
    return FactorGraph(fg.cards, out)

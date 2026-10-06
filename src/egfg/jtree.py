"""Junction trees: min-fill cliques, the clique tree, and Shafer-Shenoy message terms.

`TreeTerms` writes messages and marginals as terms on a connected part of a
clique tree. Neighbours outside that part are represented by Input leaves, so
the same code builds the whole-graph junction tree baseline, the seeds for a
whole graph, and the seeds for one cluster of a decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Input, Leaf, Mul, Sum, Term
from .model import FactorGraph


def _adjacency(fg: FactorGraph) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {v: set() for v in fg.variables()}
    for f in fg.factors:
        for u in f.scope:
            adj[u] |= set(f.scope) - {u}
    return adj


def min_fill_order(fg: FactorGraph) -> list[str]:
    return elimination_order(fg, "min_fill")


ORDER_CRITERIA = ("min_fill", "min_degree", "min_weight")


def elimination_order(fg: FactorGraph, criterion: str = "min_fill", rng=None) -> list[str]:
    """Greedy elimination order. Ties break by name, or at random when `rng` is given.

    min_fill: fewest new edges; min_degree: fewest neighbours; min_weight: smallest
    table created (product of the cardinalities of the variable and its neighbours).
    """
    if criterion not in ORDER_CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")
    adj = _adjacency(fg)
    order: list[str] = []
    while adj:
        def fill(v: str) -> int:
            nb = sorted(adj[v])
            return sum(1 for i, x in enumerate(nb) for y in nb[i + 1 :] if y not in adj[x])

        def score(u: str):
            if criterion == "min_fill":
                return fill(u)
            if criterion == "min_degree":
                return len(adj[u])
            return fg.size(adj[u] | {u})

        best = min(score(u) for u in adj)
        ties = sorted(u for u in adj if score(u) == best)
        v = ties[0] if rng is None else ties[int(rng.integers(len(ties)))]
        nb = adj.pop(v)
        for x in nb:
            adj[x] |= nb - {x}
            adj[x].discard(v)
        order.append(v)
    return order


def _cliques(fg: FactorGraph, order: list[str]) -> list[frozenset[str]]:
    adj = _adjacency(fg)
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


@dataclass
class JunctionTree:
    cliques: list[frozenset[str]]
    nbrs: dict[int, list[int]]  # clique tree adjacency, sorted
    assigned: dict[int, list[int]]  # clique -> factor ids (each factor to its lowest clique)


def junction_tree(fg: FactorGraph, order: list[str] | None = None) -> JunctionTree:
    order = min_fill_order(fg) if order is None else order
    cliques = _cliques(fg, order)
    assigned: dict[int, list[int]] = {i: [] for i in range(len(cliques))}
    for f in sorted(fg.factors, key=lambda f: f.id):
        i = min(i for i, c in enumerate(cliques) if set(f.scope) <= c)
        assigned[i].append(f.id)
    return JunctionTree(cliques, _clique_tree(cliques), assigned)


# (term, scope, factor ids below it)
Piece = tuple[Term, frozenset[str], frozenset[int]]


def product(parts: list[Piece]) -> Piece | None:
    """Left-to-right product; None for an empty product (the constant 1)."""
    if not parts:
        return None
    term, scope, fids = parts[0]
    for t, s, f in parts[1:]:
        term, scope, fids = Mul(term, t), scope | s, fids | f
    return term, scope, fids


def sum_out(piece: Piece, keep: frozenset[str] | set[str]) -> Piece:
    """Sum out every variable of the piece's scope not in `keep`, first in name order outermost."""
    term, scope, fids = piece
    for x in sorted(scope - set(keep), reverse=True):
        term = Sum(x, term)
    return term, scope & frozenset(keep), fids


class TreeTerms:
    """Shafer-Shenoy terms on the cliques `members` (a connected part of the clique tree).

    `external[(k, i)]` is the Input leaf (with its scope) standing for the message
    from clique k outside `members` into clique i inside; a missing entry means
    that message is the constant 1. Products take the clique's factors in id
    order, then its neighbours' messages in clique order.
    """

    def __init__(
        self,
        fg: FactorGraph,
        jt: JunctionTree,
        members: set[int] | None = None,
        external: dict[tuple[int, int], tuple[Input, frozenset[str]]] | None = None,
        share_products: bool = False,
    ):
        """With `share_products`, the products at a clique that leave out one neighbour
        reuse prefix and suffix products of the neighbours' messages."""
        self.share_products = share_products
        self.fg, self.jt = fg, jt
        self.members = set(range(len(jt.cliques))) if members is None else set(members)
        self.external = dict(external or {})
        self._msgs: dict[tuple[int, int], Piece | None] = {}

    def _parts(self, i: int, exclude: int | None) -> list[Piece]:
        parts: list[Piece] = [
            (Leaf(fid), frozenset(self.fg.factor(fid).scope), frozenset({fid})) for fid in self.jt.assigned[i]
        ]
        for k in self.jt.nbrs[i]:
            if k == exclude:
                continue
            if k in self.members:
                m = self.message(k, i)
            elif (k, i) in self.external:
                t, s = self.external[(k, i)]
                m = (t, s, frozenset())
            else:
                m = None
            if m is not None:
                parts.append(m)
        return parts

    def _shared_product(self, i: int, exclude: int | None) -> Piece | None:
        """factors · (prefix product of the messages before `exclude`) · (suffix product after it)."""
        facs = [(Leaf(fid), frozenset(self.fg.factor(fid).scope), frozenset({fid})) for fid in self.jt.assigned[i]]
        msgs: list[tuple[int, Piece | None]] = []
        for k in self.jt.nbrs[i]:
            if k == exclude:
                msgs.append((k, None))  # only its position matters
                continue
            if k in self.members:
                m = self.message(k, i)
            elif (k, i) in self.external:
                t, sc = self.external[(k, i)]
                m = (t, sc, frozenset())
            else:
                m = None
            if m is not None:
                msgs.append((k, m))
        if exclude is None:
            return product(facs + [m for _, m in msgs])
        pos = next((n for n, (k, _) in enumerate(msgs) if k == exclude), None)
        before = [m for _, m in msgs[:pos]] if pos is not None else [m for _, m in msgs]
        after = [m for _, m in msgs[pos + 1 :]] if pos is not None else []
        before, after = [m for m in before if m is not None], [m for m in after if m is not None]
        pre = product(before)  # left-to-right: the prefix products are shared between messages
        suf = product(list(reversed(after)))  # right-to-left: the suffix products are shared too
        return product(facs + [p for p in (pre, suf) if p is not None])

    def message(self, i: int, j: int) -> Piece | None:
        """Message from member clique i to its neighbour j (inside or outside `members`)."""
        if (i, j) not in self._msgs:
            p = self._shared_product(i, j) if self.share_products else product(self._parts(i, exclude=j))
            self._msgs[(i, j)] = None if p is None else sum_out(p, self.jt.cliques[j])
        return self._msgs[(i, j)]

    def marginal(self, v: str) -> Piece:
        """Unnormalized marginal of v at the lowest member clique containing v."""
        i = min(i for i in self.members if v in self.jt.cliques[i])
        p = product(self._parts(i, exclude=None))
        if p is None or v not in p[1]:
            raise ValueError(f"no factor or message mentions {v!r} at clique {i}")
        return sum_out(p, {v})

    def joint(self, keep: frozenset[str] | set[str]) -> Piece:
        """Unnormalized joint of `keep` at the lowest member clique containing all of it."""
        cands = [i for i in self.members if set(keep) <= self.jt.cliques[i]]
        if not cands:
            raise ValueError(f"no clique contains all of {sorted(keep)}")
        p = product(self._parts(min(cands), exclude=None))
        if p is None:
            raise ValueError(f"nothing to multiply at clique {min(cands)}")
        return sum_out(p, keep)

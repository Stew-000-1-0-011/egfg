"""Split the problem along the junction tree, solve clusters separately, stitch the results.

A cluster is a connected set of cliques of the clique tree. Each cluster gets a
local problem whose queries are its outgoing messages and the marginals of the
variables it owns; incoming messages are Input leaves. After extraction, every
Input leaf is replaced by the computation of that message in the sending
cluster, giving one Dag for all marginals.

With no budget, the whole clique tree is one cluster and the local queries are
exactly the phase 1 marginal queries.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Dag, ENode, Input, Leaf, Term
from .jtree import JunctionTree, TreeTerms, product, sum_out
from .model import FactorGraph


@dataclass
class LocalProblem:
    index: int
    cliques: frozenset[int]
    variables: frozenset[str]
    owned: list[str]  # variables whose marginal this cluster computes
    inputs: dict[str, frozenset[str]]  # input name -> scope
    queries: dict[str, Term]  # marginals are named by variable, messages by message_name
    seeds: dict[str, Term]


def message_name(i: int, k: int, piece: int | None = None) -> str:
    """The message along the clique tree edge from clique i to clique k (named by cliques, not
    clusters, so that a local problem keeps its name when other clusters change)."""
    return f"msg:{i}>{k}" if piece is None else f"msg:{i}>{k}.{piece}"


def clusters(jt: JunctionTree, budget: int | None) -> list[frozenset[int]]:
    """Merge adjacent clusters, smallest merged variable count first, while within budget.

    Starts from one cluster per clique. A clique larger than the budget stays
    alone. Clusters are returned in order of their lowest clique.
    """
    n = len(jt.cliques)
    if budget is None:
        return [frozenset(range(n))]
    groups: dict[int, set[int]] = {i: {i} for i in range(n)}  # keyed by lowest clique
    gvars: dict[int, frozenset[str]] = {i: jt.cliques[i] for i in range(n)}
    of = list(range(n))
    while True:
        best = None
        for i in range(n):
            for k in jt.nbrs[i]:
                a, b = of[i], of[k]
                if a >= b:
                    continue
                size = len(gvars[a] | gvars[b])
                if size <= budget and (best is None or (size, a, b) < best):
                    best = (size, a, b)
        if best is None:
            break
        _, a, b = best
        groups[a] |= groups.pop(b)
        gvars[a] = gvars[a] | gvars.pop(b)
        for i in groups[a]:
            of[i] = a
    return [frozenset(groups[g]) for g in sorted(groups)]


def local_problems(
    fg: FactorGraph,
    jt: JunctionTree,
    parts: list[frozenset[int]],
    seed: bool = False,
    owner: dict[str, int] | None = None,
    split: frozenset[tuple[int, int]] | set[tuple[int, int]] = frozenset(),
) -> list[LocalProblem]:
    """`owner[v]` is a clique containing v whose cluster computes v's marginal (default: the
    lowest cluster containing v). A message along a crossing edge (i, k) in `split` is sent as
    several pieces, one per group of the sender's factors and incoming pieces linked by the
    variables summed out (the product of the pieces is the message)."""
    of = {i: c for c, cl in enumerate(parts) for i in cl}
    cvars = [frozenset().union(*(jt.cliques[i] for i in cl)) for cl in parts]
    fids = [sorted(f for i in cl for f in jt.assigned[i]) for cl in parts]
    crossing: dict[tuple[int, int], tuple[int, int]] = {}  # (c, j) -> (clique in c, clique in j)
    for i in range(len(jt.cliques)):
        for k in jt.nbrs[i]:
            if of[i] != of[k]:
                crossing[(of[i], of[k])] = (i, k)
    cnbrs = {c: sorted(j for (a, j) in crossing if a == c) for c in range(len(parts))}
    owned: dict[int, list[str]] = {c: [] for c in range(len(parts))}
    for v in fg.variables():
        if owner is not None and v in owner:
            c = of[owner[v]]
            if v not in cvars[c]:
                raise ValueError(f"owner clique {owner[v]} of {v!r} does not contain it")
        else:
            c = min(c for c in range(len(parts)) if v in cvars[c])
        owned[c].append(v)

    side_memo: dict[tuple[int, int], bool] = {}

    def side_owns(j: int, c: int) -> bool:
        """Does the side of the cluster tree containing j (seen from c) own a variable?"""
        if (j, c) not in side_memo:
            side_memo[(j, c)] = bool(owned[j]) or any(side_owns(k, j) for k in cnbrs[j] if k != c)
        return side_memo[(j, c)]

    # items of a cluster: its factors and the pieces of its incoming messages, as (term, scope)
    def factor_items(c: int) -> list[tuple[Term, frozenset[str]]]:
        return [(Leaf(f), frozenset(fg.factor(f).scope)) for f in fids[c]]

    pieces_memo: dict[tuple[int, int], list[tuple[str, frozenset[str], list[tuple[Term, frozenset[str]]]]]] = {}

    def pieces(c: int, j: int) -> list[tuple[str, frozenset[str], list[tuple[Term, frozenset[str]]]]]:
        """The pieces (name, scope, items multiplied) of the message c -> j; [] if it is the constant 1."""
        if (c, j) not in pieces_memo:
            items = factor_items(c)
            for k in cnbrs[c]:
                if k != j:
                    items += [(Input(n), s) for n, s, _ in pieces(k, c)]
            i, k = crossing[(c, j)]
            if not items:
                out = []
            else:
                groups = _components(items, cvars[j]) if (i, k) in split else []
                if len(groups) > 1:
                    out = [(message_name(i, k, n), s, g) for n, (s, g) in enumerate(groups)]
                else:  # one table (also when splitting finds a single group)
                    out = [(message_name(i, k), frozenset().union(*(s for _, s in items)) & cvars[j], items)]
            pieces_memo[(c, j)] = out
        return pieces_memo[(c, j)]

    def exists(c: int, j: int) -> bool:
        return side_owns(j, c) and bool(pieces(c, j))

    def naive(items: list[tuple[Term, frozenset[str]]], keep: frozenset[str] | set[str]) -> Term:
        return sum_out(product([(t, s, frozenset()) for t, s in items]), keep)[0]

    problems = []
    for c, cl in enumerate(parts):
        incoming = [(n, s) for k in cnbrs[c] if exists(k, c) for n, s, _ in pieces(k, c)]
        inputs = dict(incoming)
        queries: dict[str, Term] = {}
        for j in cnbrs[c]:
            if exists(c, j):
                for name, s, items in pieces(c, j):  # its incoming pieces all exist (j's side owns something)
                    queries[name] = naive(items, s)
        allitems = factor_items(c) + [(Input(n), s) for n, s in incoming]
        for v in owned[c]:
            queries[v] = naive(allitems, {v})
        if not queries:
            continue
        seeds: dict[str, Term] = {}
        if seed:
            external = {}
            for k in cnbrs[c]:
                if exists(k, c):
                    ps = [(Input(n), s, frozenset()) for n, s, _ in pieces(k, c)]
                    t, s, _ = product(ps)
                    i_clique, k_clique = crossing[(c, k)]
                    external[(k_clique, i_clique)] = (t, s)
            calc = TreeTerms(fg, jt, set(cl), external)
            for j in cnbrs[c]:
                if exists(c, j) and len(pieces(c, j)) == 1:
                    seeds[pieces(c, j)[0][0]] = calc.message(*crossing[(c, j)])[0]
            for v in owned[c]:
                seeds[v] = calc.marginal(v)[0]
        problems.append(LocalProblem(c, cl, cvars[c], owned[c], inputs, queries, seeds))
    return problems


def _components(items: list[tuple[Term, frozenset[str]]], keep: frozenset[str]) -> list[tuple[frozenset[str], list]]:
    """Group the items linked by variables not in `keep` (summed out); (kept scope, items) per group.
    Groups whose kept scope is empty (constants) join the first other group."""
    parent = list(range(len(items)))

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    first: dict[str, int] = {}
    for n, (_, s) in enumerate(items):
        for x in sorted(s - keep):
            if x in first:
                parent[find(n)] = find(first[x])
            else:
                first[x] = n
    groups: dict[int, list[int]] = {}
    for n in range(len(items)):
        groups.setdefault(find(n), []).append(n)
    out = [(frozenset().union(*(items[n][1] for n in g)) & keep, [items[n] for n in g]) for g in groups.values()]
    consts = [g for s, g in out if not s]
    out = [(s, g) for s, g in out if s]
    if not out:
        return [(frozenset(), [it for g in consts for it in g])]
    for g in consts:
        out[0] = (out[0][0], out[0][1] + g)
    return out


def stitch(problems: list[LocalProblem], dags: list[Dag]) -> Dag:
    """Join the per-cluster Dags, replacing each Input leaf by the message's computation."""
    nodes: dict[str, ENode] = {}
    produced: dict[str, str] = {}  # query name -> global node id
    marginals: dict[str, str] = {}
    for p, dag in zip(problems, dags):
        for nid, node in dag.nodes.items():
            nodes[f"{p.index}:{nid}"] = ENode(node.op, node.arg, tuple(f"{p.index}:{c}" for c in node.children))
        for name, nid in dag.roots.items():
            produced[name] = f"{p.index}:{nid}"
        for v in p.owned:
            marginals[v] = produced[v]

    def resolve(nid: str) -> str:
        seen = set()
        while nodes[nid].op == "input":
            if nid in seen or nodes[nid].arg not in produced:
                raise ValueError(f"input leaf {nodes[nid].arg!r} has no producing computation")
            seen.add(nid)
            nid = produced[nodes[nid].arg]
        return nid

    out: dict[str, ENode] = {}
    roots = {v: resolve(nid) for v, nid in marginals.items()}
    stack = list(roots.values())
    while stack:
        nid = stack.pop()
        if nid in out:
            continue
        node = nodes[nid]
        out[nid] = ENode(node.op, node.arg, tuple(resolve(c) for c in node.children))
        stack.extend(out[nid].children)
    left = sorted(n.arg for n in out.values() if n.op == "input")
    if left:
        raise ValueError(f"stitched Dag still contains input leaves: {left}")
    return Dag(out, roots)

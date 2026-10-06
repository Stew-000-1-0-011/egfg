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
from .jtree import JunctionTree, Piece, TreeTerms, product, sum_out
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


def message_name(c: int, j: int) -> str:
    return f"msg:{c}>{j}"


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
    fg: FactorGraph, jt: JunctionTree, parts: list[frozenset[int]], seed: bool = False
) -> list[LocalProblem]:
    of = {i: c for c, cl in enumerate(parts) for i in cl}
    cvars = [frozenset().union(*(jt.cliques[i] for i in cl)) for cl in parts]
    fids = [sorted(f for i in cl for f in jt.assigned[i]) for cl in parts]
    fscope = [frozenset().union(*(fg.factor(f).scope for f in fs)) for fs in fids]
    crossing: dict[tuple[int, int], tuple[int, int]] = {}  # (c, j) -> (clique in c, clique in j)
    for i in range(len(jt.cliques)):
        for k in jt.nbrs[i]:
            if of[i] != of[k]:
                crossing[(of[i], of[k])] = (i, k)
    cnbrs = {c: sorted(j for (a, j) in crossing if a == c) for c in range(len(parts))}
    owned: dict[int, list[str]] = {c: [] for c in range(len(parts))}
    for v in fg.variables():
        owned[min(c for c in range(len(parts)) if v in cvars[c])].append(v)

    side_memo: dict[tuple[int, int], bool] = {}

    def side_owns(j: int, c: int) -> bool:
        """Does the side of the cluster tree containing j (seen from c) own a variable?"""
        if (j, c) not in side_memo:
            side_memo[(j, c)] = bool(owned[j]) or any(side_owns(k, j) for k in cnbrs[j] if k != c)
        return side_memo[(j, c)]

    scope_memo: dict[tuple[int, int], frozenset[str] | None] = {}

    def msg_scope(c: int, j: int) -> frozenset[str] | None:
        """Scope of the message c -> j, or None if it is the constant 1 (nothing to multiply)."""
        if (c, j) not in scope_memo:
            s, nonempty = set(fscope[c]), bool(fids[c])
            for k in cnbrs[c]:
                if k != j and (sk := msg_scope(k, c)) is not None:
                    s |= sk
                    nonempty = True
            scope_memo[(c, j)] = frozenset(s & cvars[j]) if nonempty else None
        return scope_memo[(c, j)]

    def exists(c: int, j: int) -> bool:
        return side_owns(j, c) and msg_scope(c, j) is not None

    problems = []
    for c, cl in enumerate(parts):
        inputs = {message_name(k, c): msg_scope(k, c) for k in cnbrs[c] if exists(k, c)}
        pieces: list[tuple[str | None, Piece]] = [
            (None, (Leaf(f), frozenset(fg.factor(f).scope), frozenset({f}))) for f in fids[c]
        ] + [(name, (Input(name), s, frozenset())) for name, s in sorted(inputs.items())]

        def naive(skip: str | None, keep: frozenset[str] | set[str]) -> Term:
            p = product([pc for name, pc in pieces if name is None or name != skip])
            return sum_out(p, keep)[0]

        queries: dict[str, Term] = {}
        for j in cnbrs[c]:
            if exists(c, j):
                queries[message_name(c, j)] = naive(message_name(j, c), cvars[c] & cvars[j])
        for v in owned[c]:
            queries[v] = naive(None, {v})
        if not queries:
            continue
        seeds: dict[str, Term] = {}
        if seed:
            external = {
                (k_clique, i_clique): (Input(message_name(k, c)), inputs[message_name(k, c)])
                for k in cnbrs[c]
                if message_name(k, c) in inputs
                for (i_clique, k_clique) in [crossing[(c, k)]]
            }
            calc = TreeTerms(fg, jt, set(cl), external)
            for j in cnbrs[c]:
                if exists(c, j):
                    seeds[message_name(c, j)] = calc.message(*crossing[(c, j)])[0]
            for v in owned[c]:
                seeds[v] = calc.marginal(v)[0]
        problems.append(LocalProblem(c, cl, cvars[c], owned[c], inputs, queries, seeds))
    return problems


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

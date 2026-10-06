"""Computation terms (Leaf / Mul / Sum / Input) and the shared DAG representation.

A `Dag` is the common currency of the project: queries, extraction results and
baselines all become a `Dag`, so cost and evaluation are defined once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

from .model import FactorGraph


@dataclass(frozen=True)
class Leaf:
    fid: int


@dataclass(frozen=True)
class Mul:
    a: "Term"
    b: "Term"


@dataclass(frozen=True)
class Sum:
    var: str
    a: "Term"


@dataclass(frozen=True)
class Input:
    """A placeholder for a value computed elsewhere (e.g. a message from another cluster).

    Its scope is not stored in the term; it comes from an `inputs` table
    (name -> scope) carried alongside, such as `Dag.inputs`.
    """

    name: str


Term = Union[Leaf, Mul, Sum, Input]


@dataclass(frozen=True)
class ENode:
    """op 'leaf': arg = factor id; op 'sum': arg = variable; op 'mul': arg = None;
    op 'input': arg = input name."""

    op: Literal["leaf", "mul", "sum", "input"]
    arg: int | str | None
    children: tuple[str, ...]


@dataclass
class Dag:
    nodes: dict[str, ENode]
    roots: dict[str, str]
    inputs: dict[str, frozenset[str]] = field(default_factory=dict)  # input name -> scope


def term_scope(t: Term, fg: FactorGraph, inputs: dict[str, frozenset[str]] | None = None) -> frozenset[str]:
    if isinstance(t, Leaf):
        return frozenset(fg.factor(t.fid).scope)
    if isinstance(t, Input):
        return frozenset((inputs or {})[t.name])
    if isinstance(t, Mul):
        return term_scope(t.a, fg, inputs) | term_scope(t.b, fg, inputs)
    return term_scope(t.a, fg, inputs) - {t.var}


def marginal_query(fg: FactorGraph, v: str) -> Term:
    """Unnormalized marginal of v: Σ_{others, name order, first outermost} Π_{factors by id}."""
    fids = sorted(f.id for f in fg.factors)
    body: Term = Leaf(fids[0])
    for fid in fids[1:]:
        body = Mul(body, Leaf(fid))
    for x in reversed([u for u in fg.variables() if u != v]):
        body = Sum(x, body)
    return body


def all_marginal_queries(fg: FactorGraph) -> dict[str, Term]:
    return {v: marginal_query(fg, v) for v in fg.variables()}


def to_dag(terms: dict[str, Term], inputs: dict[str, frozenset[str]] | None = None) -> Dag:
    """Hash-cons the terms into one Dag; identical subterms become one node."""
    nodes: dict[str, ENode] = {}
    memo: dict[ENode, str] = {}

    def go(t: Term) -> str:
        if isinstance(t, Leaf):
            node = ENode("leaf", t.fid, ())
        elif isinstance(t, Input):
            node = ENode("input", t.name, ())
        elif isinstance(t, Mul):
            node = ENode("mul", None, (go(t.a), go(t.b)))
        else:
            node = ENode("sum", t.var, (go(t.a),))
        if node not in memo:
            nid = f"n{len(nodes)}"
            memo[node] = nid
            nodes[nid] = node
        return memo[node]

    roots = {name: go(t) for name, t in terms.items()}
    return Dag(nodes, roots, dict(inputs or {}))


def dag_scopes(dag: Dag, fg: FactorGraph) -> dict[str, frozenset[str]]:
    scopes: dict[str, frozenset[str]] = {}

    def go(nid: str) -> frozenset[str]:
        if nid in scopes:
            return scopes[nid]
        stack = [(nid, False)]
        while stack:  # iterative post-order to avoid recursion limits
            cur, done = stack.pop()
            if cur in scopes:
                continue
            node = dag.nodes[cur]
            if not done:
                stack.append((cur, True))
                stack.extend((c, False) for c in node.children if c not in scopes)
                continue
            if node.op == "leaf":
                scopes[cur] = frozenset(fg.factor(node.arg).scope)
            elif node.op == "input":
                scopes[cur] = dag.inputs[node.arg]
            elif node.op == "mul":
                scopes[cur] = scopes[node.children[0]] | scopes[node.children[1]]
            else:
                scopes[cur] = scopes[node.children[0]] - {node.arg}
        return scopes[nid]

    for nid in dag.nodes:
        go(nid)
    return scopes

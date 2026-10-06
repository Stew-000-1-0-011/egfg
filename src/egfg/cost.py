"""FLOP cost model shared by every module.

leaf = input = 0; mul = size of the result scope; sum = size of the child scope.
DAG cost counts each node reachable from the roots exactly once.
"""

from __future__ import annotations

from .ir import Dag, ENode, dag_scopes
from .model import FactorGraph


def node_cost(
    node: ENode,
    own_scope: frozenset[str],
    child_scopes: list[frozenset[str]],
    fg: FactorGraph,
) -> int:
    if node.op in ("leaf", "input"):
        return 0
    if node.op == "mul":
        return fg.size(own_scope)
    return fg.size(child_scopes[0])


def reachable(dag: Dag) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    stack = list(dag.roots.values())
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        order.append(nid)
        stack.extend(dag.nodes[nid].children)
    return order


def dag_cost(dag: Dag, fg: FactorGraph) -> int:
    scopes = dag_scopes(dag, fg)
    total = 0
    for nid in reachable(dag):
        node = dag.nodes[nid]
        total += node_cost(node, scopes[nid], [scopes[c] for c in node.children], fg)
    return total


def max_intermediate_size(dag: Dag, fg: FactorGraph) -> int:
    scopes = dag_scopes(dag, fg)
    return max(fg.size(scopes[nid]) for nid in reachable(dag))


def depends_on(dag: Dag, after: set[int] | frozenset[int]) -> dict[str, bool]:
    """For each reachable node: is one of the factors `after` below it?"""
    dep: dict[str, bool] = {}
    for root in reachable(dag):
        stack = [root]
        while stack:
            cur = stack[-1]
            if cur in dep:
                stack.pop()
                continue
            node = dag.nodes[cur]
            todo = [c for c in node.children if c not in dep]
            if todo:
                stack.extend(todo)
                continue
            dep[cur] = (node.op == "leaf" and node.arg in after) or any(dep[c] for c in node.children)
            stack.pop()
    return dep


def split_cost(dag: Dag, fg: FactorGraph, after: set[int] | frozenset[int]) -> tuple[int, int]:
    """(cost of the nodes not depending on the factors `after`, cost of those that do)."""
    scopes = dag_scopes(dag, fg)
    dep = depends_on(dag, after)
    before = later = 0
    for nid in reachable(dag):
        node = dag.nodes[nid]
        c = node_cost(node, scopes[nid], [scopes[x] for x in node.children], fg)
        if dep[nid]:
            later += c
        else:
            before += c
    return before, later

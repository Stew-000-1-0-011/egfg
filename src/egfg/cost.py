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

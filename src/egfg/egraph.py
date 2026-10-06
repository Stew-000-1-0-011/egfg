"""Equality saturation of inference terms with egglog, and export to plain Python data.

This is the only module that touches egglog. It relies on the private
`EGraph._serialize()` (verified on egglog 14.0.0); if egglog changes, only
`_export` needs updating.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from egglog import (
    EGraph,
    Expr,
    Set,
    String,
    StringLike,
    eq,
    function,
    i64Like,
    rewrite,
    rule,
    set_,
    union,
    var,
    vars_,
)

from .ir import ENode, Input, Leaf, Mul, Sum, Term
from .model import FactorGraph


class T(Expr):
    @classmethod
    def factor(cls, i: i64Like) -> T: ...

    @classmethod
    def input(cls, name: StringLike) -> T: ...

    def __mul__(self, other: T) -> T: ...

    @classmethod
    def sum(cls, v: StringLike, t: T) -> T: ...


@function(merge=lambda old, new: old)
def scope(t: T) -> Set[String]: ...


@dataclass
class EGraphData:
    classes: dict[str, list[ENode]]
    roots: dict[str, str]
    inputs: dict[str, frozenset[str]] = field(default_factory=dict)  # input name -> scope
    # e-class -> node of a seed term (the lowest one, so following these never cycles)
    seed: dict[str, ENode] = field(default_factory=dict)


@dataclass
class SaturationResult:
    graph: EGraphData
    iterations: int
    hit_limit: bool
    seconds: float
    num_nodes: int


RULE_SETS = ("full", "no_reverse", "minimal")


def _to_egglog(t: Term) -> T:
    if isinstance(t, Leaf):
        return T.factor(t.fid)
    if isinstance(t, Input):
        return T.input(t.name)
    if isinstance(t, Mul):
        return _to_egglog(t.a) * _to_egglog(t.b)
    return T.sum(t.var, _to_egglog(t.a))


def _rules(rules: str) -> list:
    """full: phase 1; no_reverse: without reverse distributivity;
    minimal: no_reverse without the associativity direction a*(b*c) -> (a*b)*c."""
    if rules not in RULE_SETS:
        raise ValueError(f"unknown rule set {rules!r}; expected one of {RULE_SETS}")
    a, b, c = vars_("a b c", T)
    v, w = vars_("v w", String)
    s1, s2 = vars_("s1 s2", Set[String])
    out = [
        # scope analysis
        rule(eq(a * b).to(a * b), scope(a) == s1, scope(b) == s2).then(set_(scope(a * b)).to(s1 | s2)),
        rule(eq(T.sum(v, a)).to(T.sum(v, a)), scope(a) == s1).then(set_(scope(T.sum(v, a))).to(s1.remove(v))),
        # commutative semiring algebra
        rewrite(a * b).to(b * a),
        rewrite((a * b) * c).to(a * (b * c)),
    ]
    if rules != "minimal":
        out.append(rewrite(a * (b * c)).to((a * b) * c))
    out += [
        rewrite(T.sum(v, T.sum(w, a))).to(T.sum(w, T.sum(v, a))),
        # distributivity: Σv (a*b) = a * Σv b  when v ∉ scope(a)
        rule(eq(T.sum(v, a * b)).to(T.sum(v, a * b)), scope(a) == s1, s1.not_contains(v)).then(
            union(T.sum(v, a * b)).with_(a * T.sum(v, b))
        ),
    ]
    if rules == "full":
        out.append(
            rule(eq(a * T.sum(v, b)).to(a * T.sum(v, b)), scope(a) == s1, s1.not_contains(v)).then(
                union(a * T.sum(v, b)).with_(T.sum(v, a * b))
            )
        )
    return out


def _size(eg: EGraph) -> int:
    return sum(n for fn, n in eg.all_function_sizes() if "scope" not in str(fn))


def _parse_prim(op: str) -> int | str:
    return json.loads(op) if op.startswith('"') else int(op)


def _export(
    eg: EGraph,
    queries: dict[str, Term],
    inputs: dict[str, frozenset[str]],
    seeds: dict[str, Term],
) -> EGraphData:
    data = json.loads(eg._serialize(split_primitive_outputs=False).to_json())
    nodes = data["nodes"]
    classes: dict[str, list[ENode]] = {}
    for node in nodes.values():
        op = node["op"]
        if op == "T.factor":
            en = ENode("leaf", _parse_prim(nodes[node["children"][0]]["op"]), ())
        elif op == "T.input":
            en = ENode("input", _parse_prim(nodes[node["children"][0]]["op"]), ())
        elif op == "T.sum":
            v_id, t_id = node["children"]
            en = ENode("sum", _parse_prim(nodes[v_id]["op"]), (nodes[t_id]["eclass"],))
        elif op == "· * ·":
            en = ENode("mul", None, tuple(nodes[ch]["eclass"] for ch in node["children"]))
        else:
            continue
        classes.setdefault(node["eclass"], []).append(en)
    index = {en: cid for cid, ens in classes.items() for en in ens}

    def key(t: Term) -> ENode:
        if isinstance(t, Leaf):
            return ENode("leaf", t.fid, ())
        if isinstance(t, Input):
            return ENode("input", t.name, ())
        if isinstance(t, Mul):
            return ENode("mul", None, (find(t.a), find(t.b)))
        return ENode("sum", t.var, (find(t.a),))

    def find(t: Term) -> str:
        return index[key(t)]

    # For each e-class holding seed subterms, keep the seed node of least height:
    # its children then hold seed nodes of smaller height, so the choice is acyclic.
    seed: dict[str, tuple[int, ENode]] = {}
    heights: dict[Term, int] = {}

    def visit(t: Term) -> int:
        if t not in heights:
            kids = [] if isinstance(t, (Leaf, Input)) else [t.a] if isinstance(t, Sum) else [t.a, t.b]
            h = 1 + max((visit(k) for k in kids), default=0)
            en = key(t)
            cid = index[en]
            if cid not in seed or h < seed[cid][0]:
                seed[cid] = (h, en)
            heights[t] = h
        return heights[t]

    for t in seeds.values():
        visit(t)
    return EGraphData(
        classes,
        {name: find(t) for name, t in queries.items()},
        dict(inputs),
        {cid: en for cid, (_, en) in seed.items()},
    )


def saturate(
    fg: FactorGraph,
    queries: dict[str, Term],
    max_iters: int = 30,
    node_limit: int = 50_000,
    rules: str = "full",
    inputs: dict[str, frozenset[str]] | None = None,
    seeds: dict[str, Term] | None = None,
) -> SaturationResult:
    """Saturate the queries. `inputs` gives the scope of each Input leaf; `seeds`
    maps query names to known equivalent terms, which are unioned in up front."""
    start = time.perf_counter()
    inputs = dict(inputs or {})
    seeds = dict(seeds or {})
    eg = EGraph()
    for f in fg.factors:
        s = Set[String].empty()
        for v in f.scope:
            s = s.insert(String(v))
        eg.register(set_(scope(T.factor(f.id))).to(s))
    for name, vs in inputs.items():
        s = Set[String].empty()
        for v in sorted(vs):
            s = s.insert(String(v))
        eg.register(set_(scope(T.input(name))).to(s))
    for name, t in queries.items():
        eg.register(_to_egglog(t))
        if name in seeds:
            eg.register(union(_to_egglog(t)).with_(_to_egglog(seeds[name])))
    eg.register(*_rules(rules))

    iterations, hit_limit = 0, False
    while iterations < max_iters:
        report = eg.run(1)
        iterations += 1
        if _size(eg) > node_limit:
            hit_limit = True
            break
        if not report.updated:
            break
    graph = _export(eg, queries, inputs, seeds)
    num_nodes = sum(len(v) for v in graph.classes.values())
    return SaturationResult(graph, iterations, hit_limit, time.perf_counter() - start, num_nodes)

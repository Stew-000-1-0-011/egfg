"""Code generation: turn an extracted Dag into a standalone numpy function.

The generated module defines `infer(tables) -> {variable: normalized marginal}`,
where `tables` maps factor ids to arrays whose axes follow each factor's scope.
Values follow the evaluator's convention (axes in sorted variable order). A sum
(or a chain of sums) over a binary product used nowhere else becomes a single
two-operand einsum; nothing larger is fused, so the contraction order is the
one the extraction chose. Low-rank factors (phase F) are rebuilt from the
original tables with an SVD inside `infer`.
"""

from __future__ import annotations


from .ir import Dag, dag_scopes
from .model import FactorGraph


def _local(*scopes) -> dict[str, int]:
    """einsum axis numbers (< 52) for the variables of one call."""
    names = sorted(set().union(*scopes))
    if len(names) > 52:
        raise ValueError("a single operation touches more than 52 variables")
    return {v: i for i, v in enumerate(names)}


def generate(dag: Dag, fg: FactorGraph, original: FactorGraph | None = None, structured=None) -> str:
    """Python source for `infer`. `fg` is the graph the Dag refers to (for low-rank results
    the extended graph); `original` the problem's graph (defaults to `fg`)."""
    original = original or fg
    scopes = dag_scopes(dag, fg)
    order = _topo(dag)
    refs: dict[str, int] = {}
    for nid in order:
        for c in dag.nodes[nid].children:
            refs[c] = refs.get(c, 0) + 1
    for nid in dag.roots.values():
        refs[nid] = refs.get(nid, 0) + 1
    name = {nid: f"v{k}" for k, nid in enumerate(order)}
    lines = ["import numpy as np", "", "", "def infer(tables):"]
    split_of = {}
    if structured is not None:
        for sp in structured.splits:
            split_of[sp.u] = (sp, "u")
            split_of[sp.v] = (sp, "v")
    emitted_splits = set()
    skip: set[str] = set()  # nodes folded into a fused einsum

    def leaf_expr(fid: int) -> str:
        f = fg.factor(fid)
        perm = [f.scope.index(v) for v in sorted(f.scope)]
        src = f"tables[{fid}]" if fid not in split_of else f"_lr{fid}"
        return src if perm == list(range(len(perm))) else f"np.transpose({src}, {tuple(perm)})"

    # fused einsums: a sum chain over a binary product used only there
    fused: dict[str, tuple[str, str, frozenset]] = {}
    for nid in order:
        node = dag.nodes[nid]
        if node.op != "sum":
            continue
        cur = nid
        chain = [cur]
        while dag.nodes[cur].op == "sum":
            ch = dag.nodes[cur].children[0]
            if refs.get(ch, 0) != 1:
                break
            cur = ch
            chain.append(cur)
        if dag.nodes[cur].op == "mul" and cur != nid:
            # every link from nid down to the product is used once: fold them all into nid
            for x in chain[1:]:
                skip.add(x)
            a, b = dag.nodes[cur].children
            fused[nid] = (a, b, scopes[nid])
    # nodes skipped must not be needed by anything else (refcount 1 guarantees that)
    for nid in order:
        if nid in skip:
            continue
        node = dag.nodes[nid]
        if node.op == "leaf" and node.arg in split_of and node.arg not in emitted_splits:
            sp, _ = split_of[node.arg]
            lines += _split_code(sp, original)
            emitted_splits |= {sp.u, sp.v}
        n = name[nid]
        if nid in fused:
            a, b, out = fused[nid]
            m = _local(scopes[a], scopes[b])
            ia = [m[v] for v in sorted(scopes[a])]
            ib = [m[v] for v in sorted(scopes[b])]
            io = [m[v] for v in sorted(out)]
            lines.append(f"    {n} = np.einsum({name[a]}, {ia}, {name[b]}, {ib}, {io})")
        elif node.op == "leaf":
            lines.append(f"    {n} = {leaf_expr(node.arg)}")
        elif node.op == "mul":
            a, b = node.children
            m = _local(scopes[a], scopes[b])
            ia = [m[v] for v in sorted(scopes[a])]
            ib = [m[v] for v in sorted(scopes[b])]
            io = [m[v] for v in sorted(scopes[nid])]
            lines.append(f"    {n} = np.einsum({name[a]}, {ia}, {name[b]}, {ib}, {io})")
        elif node.op == "sum":
            (c,) = node.children
            axis = sorted(scopes[c]).index(node.arg)
            lines.append(f"    {n} = {name[c]}.sum(axis={axis})")
        else:
            raise ValueError(f"cannot generate code for op {node.op!r}")
    lines.append("    out = {}")
    for v, nid in sorted(dag.roots.items()):
        lines.append(f"    out[{v!r}] = {name[nid]} / {name[nid]}.sum()")
    lines.append("    return out")
    return "\n".join(lines) + "\n"


def _topo(dag: Dag) -> list[str]:
    """Reachable nodes, children before parents."""
    seen: set[str] = set()
    out: list[str] = []
    for root in dag.roots.values():
        stack = [(root, False)]
        while stack:
            nid, done = stack.pop()
            if done:
                out.append(nid)
                continue
            if nid in seen:
                continue
            seen.add(nid)
            stack.append((nid, True))
            stack.extend((c, False) for c in dag.nodes[nid].children if c not in seen)
    return out


def _split_code(sp, original: FactorGraph) -> list[str]:
    f = original.factor(sp.fid)
    perm = [f.scope.index(v) for v in sp.left + sp.right]
    rows = 1
    for v in sp.left:
        rows *= original.cards[v]
    lshape = [original.cards[v] for v in sp.left]
    rshape = [original.cards[v] for v in sp.right]
    r = sp.rank
    return [
        f"    _m = np.transpose(tables[{sp.fid}], {tuple(perm)}).reshape({rows}, -1)",
        "    _u, _s, _vt = np.linalg.svd(_m, full_matrices=False)",
        f"    _lr{sp.u} = (_u[:, :{r}] * _s[:{r}]).reshape({lshape + [r]})",
        f"    _lr{sp.v} = _vt[:{r}, :].reshape({[r] + rshape})",
    ]


def compile_program(source: str):
    """The `infer` function of generated source."""
    ns: dict = {}
    exec(compile(source, "<egfg-generated>", "exec"), ns)
    return ns["infer"]


def generate_for(fg: FactorGraph, res) -> str:
    """Source for an OptimizeResult of `pipeline.optimize`."""
    eval_fg = res.eval_fg if res.eval_fg is not None else fg
    return generate(res.extraction.dag, eval_fg, fg, res.structured)

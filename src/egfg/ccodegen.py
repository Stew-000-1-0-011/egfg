"""C code generation: turn an extracted Dag into a standalone C function.

The generated file defines

    void infer(const double *const *tables, double *out);

`tables[i]` is factor i's table (doubles, row-major, axes in the factor's scope order);
`out` receives the marginals concatenated in sorted variable-name order, each normalized.

Every non-leaf node becomes one loop nest writing a static array whose axes follow the
node's variables in sorted order. Leaves are read in place through strides (no copies).
A sum (or chain of sums) over a binary product used nowhere else becomes one loop nest
that accumulates the product directly. All marginals share the normalizing constant.
"""

from __future__ import annotations

from .ir import Dag, dag_scopes
from .model import FactorGraph


def _strides(vars_: list[str], cards: dict[str, int]) -> dict[str, int]:
    out, s = {}, 1
    for v in reversed(vars_):
        out[v] = s
        s *= cards[v]
    return out


def _topo(dag: Dag) -> list[str]:
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


def generate_c(dag: Dag, fg: FactorGraph) -> str:
    cards = fg.cards
    scopes = dag_scopes(dag, fg)
    order = _topo(dag)
    refs: dict[str, int] = {}
    for nid in order:
        for c in dag.nodes[nid].children:
            refs[c] = refs.get(c, 0) + 1
    for nid in dag.roots.values():
        refs[nid] = refs.get(nid, 0) + 1

    # how to read each value: (C expression of the base pointer, stride per variable)
    access: dict[str, tuple[str, dict[str, int]]] = {}
    decls: list[str] = []
    body: list[str] = []
    loopvar = {v: f"i{k}" for k, v in enumerate(sorted(cards))}

    def size(vars_) -> int:
        n = 1
        for v in vars_:
            n *= cards[v]
        return n

    def index(nid: str) -> str:
        base, st = access[nid]
        terms = [f"{loopvar[v]}*{st[v]}" if st[v] != 1 else loopvar[v] for v in sorted(st)]
        return f"{base}[{' + '.join(terms) if terms else '0'}]"

    def loops(vars_: list[str], inner: list[str], indent: str) -> list[str]:
        out = []
        for d, v in enumerate(vars_):
            out.append(f"{indent}{'    ' * d}for (int {loopvar[v]} = 0; {loopvar[v]} < {cards[v]}; {loopvar[v]}++) {{")
        out += [f"{indent}{'    ' * len(vars_)}{line}" for line in inner]
        out += [f"{indent}{'    ' * d}}}" for d in reversed(range(len(vars_)))]
        return out

    fused: dict[str, tuple[str, str]] = {}
    skip: set[str] = set()
    for nid in order:
        if dag.nodes[nid].op != "sum":
            continue
        cur, chain = nid, [nid]
        while dag.nodes[cur].op == "sum":
            ch = dag.nodes[cur].children[0]
            if refs.get(ch, 0) != 1:
                break
            cur = ch
            chain.append(cur)
        if dag.nodes[cur].op == "mul" and cur != nid:
            skip.update(chain[1:])
            fused[nid] = tuple(dag.nodes[cur].children)
        elif cur != nid and len(chain) > 1:
            # a chain of sums over something shared: fold the chain into one reduction
            skip.update(chain[1:])
            fused[nid] = (chain[-1],)

    k = 0
    for nid in order:
        if nid in skip:
            continue
        node = dag.nodes[nid]
        if node.op == "leaf":
            f = fg.factor(node.arg)
            access[nid] = (f"tables[{node.arg}]", _strides(list(f.scope), cards))
            continue
        if node.op == "input":
            raise ValueError("input leaves cannot be compiled to C")
        out_vars = sorted(scopes[nid])
        name = f"t{k}"
        k += 1
        decls.append(f"static double {name}[{max(1, size(out_vars))}];")
        access[nid] = (name, _strides(out_vars, cards))
        if nid in fused or node.op == "sum":
            operands = fused.get(nid, node.children)
            inner_vars = sorted(set().union(*(scopes[c] for c in operands)) - set(out_vars))
            prod = " * ".join(index(c) for c in operands)
            body += loops(out_vars, ["double acc = 0.0;"] + loops(inner_vars, [f"acc += {prod};"], "")
                          + [f"{index(nid)} = acc;"], "    ")
        else:  # mul
            a, b = node.children
            body += loops(out_vars, [f"{index(nid)} = {index(a)} * {index(b)};"], "    ")
    # outputs: marginals in sorted variable order, sharing the normalizer
    roots = sorted(dag.roots.items())
    first = roots[0][1]
    lines = ["#include <stddef.h>", "", *decls, "", "void infer(const double *const *tables, double *out) {", *body]
    v0 = scopes[first]
    n0 = size(sorted(v0))
    base0, _ = access[first]
    lines.append(f"    double z = 0.0;")
    lines.append(f"    for (int j = 0; j < {n0}; j++) z += {base0}[j];")
    lines.append("    double iz = 1.0 / z;")
    off = 0
    for v, nid in roots:
        base, _ = access[nid]
        n = cards[v]
        lines.append(f"    for (int j = 0; j < {n}; j++) out[{off} + j] = {base}[j] * iz;")
        off += n
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_c_for(fg: FactorGraph, res) -> str:
    if res.eval_fg is not None and res.sum_product_only:
        raise ValueError("C generation does not support low-rank results yet")
    return generate_c(res.extraction.dag, fg)


CFLAGS = ["-O3", "-march=native", "-std=c11", "-shared", "-fPIC"]


def compile_c_program(source: str, variables: list[str], cards: dict[str, int], workdir=None):
    """Compile C source defining `infer` and return a Python wrapper
    `run(tables: {factor id: array}) -> {variable: marginal}` (and the raw ctypes handle)."""
    import ctypes
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    d = Path(workdir or tempfile.mkdtemp(prefix="egfg_c_"))
    d.mkdir(parents=True, exist_ok=True)
    src = d / "prog.c"
    lib = d / "prog.so"
    src.write_text(source)
    subprocess.run(["gcc", *CFLAGS, "-o", str(lib), str(src), "-lm"], check=True, capture_output=True)
    so = ctypes.CDLL(str(lib))
    so.infer.restype = None
    so.infer.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.c_double)]
    names = sorted(variables)
    total = sum(cards[v] for v in names)

    def prepare(tables: dict[int, "np.ndarray"]):
        n = max(tables) + 1
        bufs = [np.ascontiguousarray(tables[i], dtype=np.float64) if i in tables else np.zeros(1) for i in range(n)]
        ptrs = (ctypes.POINTER(ctypes.c_double) * n)(*[b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for b in bufs])
        out = np.zeros(total)
        return bufs, ptrs, out

    def run(tables):
        bufs, ptrs, out = prepare(tables)
        so.infer(ptrs, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        res, off = {}, 0
        for v in names:
            res[v] = out[off : off + cards[v]].copy()
            off += cards[v]
        return res

    run.prepare = prepare
    run.lib = so
    return run

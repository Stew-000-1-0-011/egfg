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


_SOLVE = """
/* Solve W X = B in place (W: r x r, B: r x n, row-major) by Gauss-Jordan with partial pivoting. */
static void egfg_solve(int r, double *W, double *B, int n) {
    for (int c = 0; c < r; c++) {
        int p = c;
        for (int i = c + 1; i < r; i++)
            if (fabs(W[i * r + c]) > fabs(W[p * r + c])) p = i;
        if (p != c) {
            for (int j = 0; j < r; j++) { double t = W[c * r + j]; W[c * r + j] = W[p * r + j]; W[p * r + j] = t; }
            for (int j = 0; j < n; j++) { double t = B[c * n + j]; B[c * n + j] = B[p * n + j]; B[p * n + j] = t; }
        }
        double inv = 1.0 / W[c * r + c];
        for (int j = 0; j < r; j++) W[c * r + j] *= inv;
        for (int j = 0; j < n; j++) B[c * n + j] *= inv;
        for (int i = 0; i < r; i++) {
            if (i == c) continue;
            double f = W[i * r + c];
            if (f == 0.0) continue;
            for (int j = 0; j < r; j++) W[i * r + j] -= f * W[c * r + j];
            for (int j = 0; j < n; j++) B[i * n + j] -= f * B[c * n + j];
        }
    }
}
"""


def _split_c(sp, original: FactorGraph) -> tuple[list[str], list[str]]:
    """Declarations and code that build U (= chosen columns) and V (= W^-1 R) of a low-rank split.

    The table is viewed as a matrix M (left variables x right variables, row-major); the first
    r rows and columns are used (W = M[:r, :r]): exact when M has rank r and W is invertible.
    """
    f = original.factor(sp.fid)
    perm = [f.scope.index(v) for v in sp.left + sp.right]
    rows = 1
    for v in sp.left:
        rows *= original.cards[v]
    cols = 1
    for v in sp.right:
        cols *= original.cards[v]
    r = sp.rank
    # element M[i][j] of the permuted matrix, read from the table in its own layout
    st = _strides(list(f.scope), original.cards)
    lvars, rvars = list(sp.left), list(sp.right)

    def offset(vars_, idx_name):
        # decompose a flat index over vars_ (row-major) into the table offset
        parts, div = [], 1
        for v in reversed(vars_):
            parts.append(f"(({idx_name} / {div}) % {original.cards[v]}) * {st[v]}")
            div *= original.cards[v]
        return " + ".join(parts)

    u, v = f"lrU{sp.u}", f"lrV{sp.v}"
    decls = [f"static double {u}[{rows * r}];", f"static double {v}[{r * cols}];", f"static double lrW{sp.u}[{r * r}];"]
    t = f"tables[{sp.fid}]"
    code = [
        f"    for (int i = 0; i < {rows}; i++) for (int a = 0; a < {r}; a++) "
        f"{u}[i * {r} + a] = {t}[{offset(lvars, 'i')} + {offset(rvars, 'a')}];",
        f"    for (int a = 0; a < {r}; a++) for (int j = 0; j < {cols}; j++) "
        f"{v}[a * {cols} + j] = {t}[{offset(lvars, 'a')} + {offset(rvars, 'j')}];",
        f"    for (int a = 0; a < {r}; a++) for (int b = 0; b < {r}; b++) "
        f"lrW{sp.u}[a * {r} + b] = {t}[{offset(lvars, 'a')} + {offset(rvars, 'b')}];",
        f"    egfg_solve({r}, lrW{sp.u}, {v}, {cols});",
    ]
    return decls, code


def _fusion(dag: Dag, order: list[str], refs: dict[str, int]) -> tuple[dict[str, tuple], set[str]]:
    """Chains of sums fused into one loop nest: {sum node: operands}, and the nodes they absorb."""
    fused: dict[str, tuple] = {}
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
        if cur == nid:
            continue
        if dag.nodes[cur].op == "mul":
            # sums down to a product used only here: one loop nest over the product's operands
            skip.update(chain[1:])
            fused[nid] = tuple(dag.nodes[cur].children)
        elif dag.nodes[cur].op == "sum":
            # stopped above a shared child: fold the whole chain into one reduction of that child
            skip.update(chain[1:])
            fused[nid] = (dag.nodes[cur].children[0],)
        else:
            # sums down to a leaf: reduce the leaf directly (the leaf itself is read, not skipped)
            skip.update(chain[1:-1])
            fused[nid] = (cur,)
    return fused, skip


def _refs(dag: Dag, order: list[str]) -> dict[str, int]:
    refs: dict[str, int] = {}
    for nid in order:
        for c in dag.nodes[nid].children:
            refs[c] = refs.get(c, 0) + 1
    for nid in dag.roots.values():
        refs[nid] = refs.get(nid, 0) + 1
    return refs


def generate_c(dag: Dag, fg: FactorGraph, original: FactorGraph | None = None, structured=None) -> str:
    """C source. `fg` is the graph the Dag refers to (the extended graph for low-rank results),
    `original` the problem's graph, `structured` the low-rank splits used (if any)."""
    original = original or fg
    cards = fg.cards
    scopes = dag_scopes(dag, fg)
    order = _topo(dag)
    refs = _refs(dag, order)

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

    fused, skip = _fusion(dag, order, refs)

    split_of = {}
    if structured is not None:
        for sp in structured.splits:
            split_of[sp.u] = sp
            split_of[sp.v] = sp
    built = set()
    k = 0
    for nid in order:
        if nid in skip:
            continue
        node = dag.nodes[nid]
        if node.op == "leaf":
            f = fg.factor(node.arg)
            if node.arg in split_of:
                sp = split_of[node.arg]
                if sp.u not in built:
                    d, c = _split_c(sp, original)
                    decls += d
                    body += c
                    built.add(sp.u)
                arr = f"lrU{sp.u}" if node.arg == sp.u else f"lrV{sp.v}"
                access[nid] = (arr, _strides(list(f.scope), cards))
            else:
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
    head = ["#include <stddef.h>", "#include <math.h>", ""]
    if built:
        head.append(_SOLVE)
    lines = [*head, *decls, "", "void infer(const double *const *tables, double *out) {", *body]
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


def generate_c_compact(dag: Dag, fg: FactorGraph, original: FactorGraph | None = None, structured=None) -> str:
    """The same computation as `generate_c`, shorter (phase J): every loop nest of the same shape
    (numbers of states, strides; not names) is one kernel function, the calls are ordered so that
    the same kernel runs back to back, intermediate arrays are pooled by size, and runs of calls of
    one kernel become loops (over a fixed step, or over constant tables of array numbers)."""
    original = original or fg
    cards = fg.cards
    scopes = dag_scopes(dag, fg)
    order = _topo(dag)
    refs = _refs(dag, order)
    fused, skip = _fusion(dag, order, refs)
    split_of = {}
    if structured is not None:
        for sp in structured.splits:
            split_of[sp.u] = sp
            split_of[sp.v] = sp
    decls: list[str] = []
    pre: list[str] = []
    built: set[int] = set()
    where: dict[str, tuple] = {}  # node -> ("tables", fid) | ("lr", name) | ("pool", size, index)
    strides: dict[str, dict[str, int]] = {}
    ops = []
    for nid in order:
        if nid in skip:
            continue
        node = dag.nodes[nid]
        if node.op == "leaf":
            f = fg.factor(node.arg)
            strides[nid] = _strides(list(f.scope), cards)
            if node.arg in split_of:
                sp = split_of[node.arg]
                if sp.u not in built:
                    d, c = _split_c(sp, original)
                    decls += d
                    pre += c
                    built.add(sp.u)
                where[nid] = ("lr", f"lrU{sp.u}" if node.arg == sp.u else f"lrV{sp.v}")
            else:
                where[nid] = ("tables", node.arg)
            continue
        if node.op == "input":
            raise ValueError("input leaves cannot be compiled to C")
        out_vars = sorted(scopes[nid])
        strides[nid] = _strides(out_vars, cards)
        reduce = nid in fused or node.op == "sum"
        operands = tuple(fused.get(nid, node.children)) if reduce else tuple(node.children)
        inner = sorted(set().union(*(scopes[c] for c in operands)) - set(out_vars)) if reduce else []
        ops.append((nid, reduce, out_vars, inner, operands))

    # shapes -> kernels
    def shape(op):
        nid, reduce, out_vars, inner, operands = op
        canon = out_vars + inner
        return (reduce, tuple(cards[v] for v in out_vars), tuple(cards[v] for v in inner),
                tuple(tuple(strides[c].get(v, 0) for v in canon) for c in operands))

    kernel: dict[tuple, str] = {}
    op_shape = {}
    for op in ops:
        sh = shape(op)
        kernel.setdefault(sh, f"k{len(kernel)}")
        op_shape[op[0]] = sh

    # order the calls: dependencies first, then the same kernel as the previous call
    pos = {op[0]: i for i, op in enumerate(ops)}
    is_op = set(pos)
    waiting = {op[0]: {c for c in op[4] if c in is_op} for op in ops}
    users: dict[str, list[str]] = {}
    for op in ops:
        for c in waiting[op[0]]:
            users.setdefault(c, []).append(op[0])
    ready = sorted((n for n, w in waiting.items() if not w), key=pos.get)
    sched, last = [], None
    by_id = {op[0]: op for op in ops}
    while ready:
        same = [n for n in ready if kernel[op_shape[n]] == last]
        pick = same[0] if same else ready[0]
        ready.remove(pick)
        sched.append(by_id[pick])
        last = kernel[op_shape[pick]]
        for u in users.get(pick, []):
            waiting[u].discard(pick)
            if not waiting[u]:
                ready.append(u)
        ready.sort(key=pos.get)

    # pooled intermediate arrays, numbered in call order
    pools: dict[int, int] = {}
    for op in sched:
        n = 1
        for v in op[2]:
            n *= cards[v]
        where[op[0]] = ("pool", n, pools.get(n, 0))
        pools[n] = pools.get(n, 0) + 1

    def expr(w, step=None) -> str:
        if w[0] == "tables":
            return f"tables[{w[1] if step is None else step}]"
        if w[0] == "lr":
            return w[1]
        return f"p{w[1]}[{w[2] if step is None else step}]"

    def slots(op):
        return [where[op[0]]] + [where[c] for c in op[4]]

    def affine(seq):
        """For a run's values of one argument: the loop expression, or None when not regular."""
        first = seq[0]
        if any(w[0] != first[0] for w in seq) or (first[0] == "pool" and any(w[1] != first[1] for w in seq)):
            return None
        if first[0] == "lr":
            return expr(first) if all(w == first for w in seq) else None
        num = [w[-1] for w in seq]
        d = num[1] - num[0]
        if any(num[i] != num[0] + i * d for i in range(len(num))):
            return None
        return expr(first, f"{num[0]} + {d} * r" if d else f"{num[0]}")

    def kind(w):
        return w[:2] if w[0] == "pool" else w if w[0] == "lr" else w[:1]

    body: list[str] = []
    tables_decl: list[str] = []
    i = 0
    while i < len(sched):
        k = kernel[op_shape[sched[i][0]]]
        kinds = [kind(w) for w in slots(sched[i])]
        j = i + 1
        while j < len(sched) and kernel[op_shape[sched[j][0]]] == k and [kind(w) for w in slots(sched[j])] == kinds:
            j += 1
        run = sched[i:j]
        if len(run) >= 2:
            args = []
            for a in range(len(kinds)):
                seq = [slots(op)[a] for op in run]
                e = affine(seq)
                if e is None:  # irregular: the numbers go in a constant table
                    name = f"ix{len(tables_decl)}"
                    tables_decl.append(f"static const int {name}[{len(seq)}] = {{{', '.join(str(w[-1]) for w in seq)}}};")
                    e = expr(seq[0], f"{name}[r]")
                args.append(e)
            body.append(f"    for (int r = 0; r < {len(run)}; r++) {k}({', '.join(args)});")
        else:
            body.append(f"    {k}({', '.join(expr(w) for w in slots(run[0]))});")
        i = j

    kernels = []
    for sh, name in kernel.items():
        reduce, oc, ic, op_strides = sh
        nv = len(oc) + len(ic)
        ext = list(oc) + list(ic)
        params = ", ".join(["double *restrict o"] + [f"const double *restrict a{t}" for t in range(len(op_strides))])
        out_st, acc = [], 1
        for c in reversed(oc):
            out_st.insert(0, acc)
            acc *= c
        ix = lambda st: " + ".join(f"v{t}*{x}" if x != 1 else f"v{t}" for t, x in enumerate(st) if x) or "0"  # noqa: E731
        prod = " * ".join(f"a{t}[{ix(st)}]" for t, st in enumerate(op_strides))
        oi = ix(out_st + [0] * len(ic))
        lines = [f"static void {name}({params}) {{"]
        ind = "    "
        for t in range(len(oc)):
            lines.append(f"{ind}for (int v{t} = 0; v{t} < {ext[t]}; v{t}++)")
            ind += "    "
        if reduce:
            lines.append(f"{ind}{{ double acc = 0.0;")
            inner_ind = ind + "  "
            for t in range(len(oc), nv):
                lines.append(f"{inner_ind}for (int v{t} = 0; v{t} < {ext[t]}; v{t}++)")
                inner_ind += "    "
            lines.append(f"{inner_ind}acc += {prod};")
            lines.append(f"{ind}  o[{oi}] = acc; }}")
        else:
            lines.append(f"{ind}o[{oi}] = {prod};")
        lines.append("}")
        kernels += lines
    pool_decls = [f"static double p{n}[{cnt}][{n}];" for n, cnt in sorted(pools.items())]

    roots = sorted(dag.roots.items())
    first = roots[0][1]
    n0 = 1
    for v in scopes[first]:
        n0 *= cards[v]
    head = ["#include <stddef.h>", "#include <math.h>", ""]
    if built:
        head.append(_SOLVE)
    # outputs: marginals in sorted variable order, sharing the normalizer; runs of marginals of the
    # same size kept in the same pool are copied by one loop over a table of their numbers
    outs, off = [], 0
    for v, nid in roots:
        outs.append((off, cards[v], where[nid]))
        off += cards[v]
    out_lines, i = [], 0
    while i < len(outs):
        o0, c, w = outs[i]
        j = i + 1
        while j < len(outs) and outs[j][1] == c and w[0] == "pool" and outs[j][2][:2] == w[:2]:
            j += 1
        if j - i >= 2:
            name = f"ix{len(tables_decl)}"
            tables_decl.append(f"static const int {name}[{j - i}] = {{{', '.join(str(x[2][2]) for x in outs[i:j])}}};")
            out_lines.append(f"    for (int r = 0; r < {j - i}; r++) for (int j = 0; j < {c}; j++) "
                             f"out[{o0} + {c} * r + j] = p{w[1]}[{name}[r]][j] * iz;")
        else:
            out_lines.append(f"    for (int j = 0; j < {c}; j++) out[{o0} + j] = {expr(w)}[j] * iz;")
        i = j
    lines = [*head, *decls, *pool_decls, *tables_decl, "", *kernels, "",
             "void infer(const double *const *tables, double *out) {", *pre, *body]
    lines.append(f"    const double *zb = {expr(where[first])};")
    lines.append(f"    double z = 0.0;")
    lines.append(f"    for (int j = 0; j < {n0}; j++) z += zb[j];")
    lines.append("    double iz = 1.0 / z;")
    lines += out_lines
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_c_for(fg: FactorGraph, res, compact: bool = False) -> str:
    gen = generate_c_compact if compact else generate_c
    if res.structured is not None:
        return gen(res.extraction.dag, res.eval_fg, fg, res.structured)
    return gen(res.extraction.dag, fg)


CFLAGS = ["-O3", "-march=native", "-std=c11", "-ffp-contract=fast", "-shared", "-fPIC"]

# A separate translation unit that calls `infer` n times natively, so that timing is not
# dominated by the cost of one ctypes call (about 0.3-0.5 us, more than many programs take).
_BENCH = """
void infer(const double *const *tables, double *out);
void egfg_bench(const double *const *tables, double *out, long n) {
    for (long k = 0; k < n; k++) {
        infer(tables, out);
        __asm__ __volatile__("" ::: "memory");
    }
}
"""


_GCC: str | None = None


def _gcc_version() -> str:
    global _GCC
    if _GCC is None:
        import subprocess

        _GCC = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return _GCC


def compile_c_program(source: str, variables: list[str], cards: dict[str, int], workdir=None):
    """Compile C source defining `infer` and return a Python wrapper
    `run(tables: {factor id: array}) -> {variable: marginal}` (and the raw ctypes handle).
    With `EGFG_CACHE` set, the shared library is kept by the hash of the source (see cache.py)."""
    import ctypes
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    from . import cache

    k = cache.key("so", source, _BENCH, CFLAGS, _gcc_version())
    lib = cache.path_for("so", k, ".so")
    if lib is not None and lib.exists():
        cache._touch(lib)
    else:
        d = Path(workdir or tempfile.mkdtemp(prefix="egfg_c_"))
        d.mkdir(parents=True, exist_ok=True)
        src = d / "prog.c"
        bench = d / "bench.c"
        built = d / "prog.so"
        src.write_text(source)
        bench.write_text(_BENCH)
        subprocess.run(["gcc", *CFLAGS, "-o", str(built), str(src), str(bench), "-lm"], check=True, capture_output=True)
        lib = cache.store_file("so", k, ".so", built) if lib is not None else built
    so = ctypes.CDLL(str(lib))
    so.infer.restype = None
    so.infer.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.c_double)]
    so.egfg_bench.restype = None
    so.egfg_bench.argtypes = so.infer.argtypes + [ctypes.c_long]
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

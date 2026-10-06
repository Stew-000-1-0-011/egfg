"""C code generation for linear-Gaussian filter programs (phase D's templates).

The head and step templates become straight-line C over fixed-size dense arrays:
every chosen (e-class, representation) state is one static buffer, every physical
operation (prediction, Kalman update, information product, Schur complement,
conversions) a few loops. Index maps are worked out here, at generation time.
Normalizing constants (and so the log-likelihood) are not computed: the program
returns the mean and covariance of every state variable.

Contract of the generated code (dimensions fixed, factors numbered initial +
transition + observation):

  void egfg_setup(const double *const *params);  // params[f]: A_1.., b, Q (row-major)
  void egfg_head(const double *const *ys, double *out);  // the first step
  void egfg_step(const double *const *ys, double *out);  // every later step
  // ys[k]: the observed vector of observation factor k
  // out: the means, then the covariances, of the variables in name order

The parameter-only parts of information forms (a factor's J, and HᵀR⁻¹ of an
observation) are computed once in `egfg_setup`, as with `amortize_constants`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dynamic import MSG, at, split
from .gaussian import GFactor
from .gdynamic import COND, INFO, MOMENT, GaussianDynamicModel, GFilterProgram, GTemplate

_LIB = r"""
#include <math.h>
#include <string.h>

/* lower Cholesky factor of an SPD n x n matrix (row-major) */
static inline void egfg_chol(int n, const double *A, double *L) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            double s = A[i * n + j];
            for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = i == j ? sqrt(s) : s / L[j * n + j];
        }
}

/* X = A^-1 B for SPD A (n x n) and B (n x m), through the Cholesky factor */
static inline void egfg_spd_solve(int n, int m, const double *A, const double *B, double *X) {
    double L[n * n];
    egfg_chol(n, A, L);
    for (int c = 0; c < m; c++) {
        for (int i = 0; i < n; i++) {
            double s = B[i * m + c];
            for (int k = 0; k < i; k++) s -= L[i * n + k] * X[k * m + c];
            X[i * m + c] = s / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
            double s = X[i * m + c];
            for (int k = i + 1; k < n; k++) s -= L[k * n + i] * X[k * m + c];
            X[i * m + c] = s / L[i * n + i];
        }
    }
}

/* inverse of an SPD matrix */
static inline void egfg_spd_inv(int n, const double *A, double *X) {
    double I[n * n];
    for (int i = 0; i < n * n; i++) I[i] = 0;
    for (int i = 0; i < n; i++) I[i * n + i] = 1;
    egfg_spd_solve(n, n, A, I, X);
}
"""


@dataclass
class _Val:
    rep: str
    vars: tuple[str, ...]
    a: str  # mu (moment) or h (info)
    b: str  # S (moment) or J (info)
    fid: int | None = None  # for COND: the template's factor index


class _Gen:
    def __init__(self, model: GaussianDynamicModel):
        self.model = model
        self.dims = model.local_dims()
        self.facs = model.initial + model.transition + model.observation
        self.decl: list[str] = []
        self.setup: list[str] = []
        self.n = 0
        self.consts: dict[tuple, str] = {}
        self.fac_ids = {id(f): i for i, f in enumerate(self.facs)}
        for i, f in enumerate(self.facs):
            self._params(i, f)

    # -- helpers -------------------------------------------------------------
    def D(self, vars_) -> int:
        return sum(self.dims[v] for v in vars_)

    def idx(self, vars_, sub) -> list[int]:
        off, k = {}, 0
        for v in vars_:
            off[v] = k
            k += self.dims[v]
        return [off[v] + j for v in sub for j in range(self.dims[v])]

    def buf(self, size: int, tag: str) -> str:
        self.n += 1
        name = f"{tag}{self.n}"
        self.decl.append(f"static double {name}[{max(size, 1)}];")
        return name

    def ints(self, values: list[int]) -> str:
        key = tuple(values)
        if key not in self.consts:
            name = f"I{len(self.consts)}"
            self.consts[key] = name
            body = ", ".join(map(str, values)) if values else "0"
            self.decl.append(f"static const int {name}[{max(len(values), 1)}] = {{{body}}};")
        return self.consts[key]

    def _params(self, i: int, f: GFactor) -> None:
        """Static copies of a factor's parameters, and its information form (J, and h or W)."""
        rows = len(f.b)
        names = []
        for k, (A, p) in enumerate(zip(f.A, f.parents)):
            names.append((self.buf(A.size, f"A{i}_{k}_"), A.shape))
        b, Q = self.buf(rows, f"b{i}_"), self.buf(rows * rows, f"Q{i}_")
        f_bufs = {"A": names, "b": b, "Q": Q}
        s = self.setup
        s.append(f"    {{ const double *p = params[{i}];")
        for name, shape in names:
            s.append(f"      memcpy({name}, p, sizeof(double) * {shape[0] * shape[1]}); p += {shape[0] * shape[1]};")
        s.append(f"      memcpy({b}, p, sizeof(double) * {rows}); p += {rows};")
        s.append(f"      memcpy({Q}, p, sizeof(double) * {rows * rows}); }}")
        # information form: residual r = M x - t with M over sorted(scope), J = MᵀQ⁻¹M
        order = tuple(sorted(f.scope))
        Dv = self.D(order)
        M = self.buf(rows * Dv, f"M{i}_")
        s.append(f"    memset({M}, 0, sizeof {M});")
        for v in order:
            cols = self.idx(order, (v,))
            if f.kind != "obs" and v == f.child:
                for r in range(rows):
                    s.append(f"    {M}[{r * Dv + cols[r]}] = 1;")
            else:
                (name, shape), = [names[k] for k, p in enumerate(f.parents) if p == v]
                sign = "" if f.kind == "obs" else "-"
                ci = self.ints(cols)
                s.append(f"    for (int r = 0; r < {rows}; r++) for (int c = 0; c < {shape[1]}; c++) "
                         f"{M}[r * {Dv} + {ci}[c]] = {sign}{name}[r * {shape[1]} + c];")
        X = self.buf(rows * Dv, f"QiM{i}_")  # Q⁻¹ M
        J = self.buf(Dv * Dv, f"Jf{i}_")
        s.append(f"    egfg_spd_solve({rows}, {Dv}, {Q}, {M}, {X});")
        s.append(f"    for (int r = 0; r < {Dv}; r++) for (int c = 0; c < {Dv}; c++) {{ double a = 0; "
                 f"for (int k = 0; k < {rows}; k++) a += {M}[k * {Dv} + r] * {X}[k * {Dv} + c]; {J}[r * {Dv} + c] = a; }}")
        f_bufs.update(order=order, J=J, X=X)
        if f.kind != "obs":
            h = self.buf(Dv, f"hf{i}_")  # Mᵀ Q⁻¹ b
            s.append(f"    for (int r = 0; r < {Dv}; r++) {{ double a = 0; "
                     f"for (int k = 0; k < {rows}; k++) a += {X}[k * {Dv} + r] * {b}[k]; {h}[r] = a; }}")
            f_bufs["h"] = h
        setattr(self, f"f{i}", f_bufs)

    def fbufs(self, f: GFactor) -> dict:
        return getattr(self, f"f{self.fac_ids[id(f)]}")

    # -- one template ----------------------------------------------------------
    def template(self, tmpl: GTemplate, name: str, fwd: _Val | None) -> tuple[list[str], _Val]:
        facs = tmpl.facs
        first = len(facs) - len(self.model.observation)
        ch = tmpl.choice.choice
        vals: dict = {}
        code: list[str] = []

        def visit(s):
            stack = [s]
            while stack:
                cur = stack[-1]
                if cur in vals:
                    stack.pop()
                    continue
                todo = [k for k in ch[cur].kids if k not in vals]
                if todo:
                    stack.extend(todo)
                    continue
                vals[cur] = self.apply(ch[cur], [vals[k] for k in ch[cur].kids], facs, first, fwd, code)
                stack.pop()

        for s in tmpl.choice.roots.values():
            visit(s)
        names = sorted(self.model.dims)
        off = 0
        for n in names:
            v = vals[tmpl.choice.roots[n]]
            assert v.rep == MOMENT and v.vars == (at(n, 0),), v.vars
            d = self.dims[at(n, 0)]
            code.append(f"    memcpy(out + {off}, {v.a}, sizeof(double) * {d});")
            off += d
        for n in names:
            v = vals[tmpl.choice.roots[n]]
            d = self.dims[at(n, 0)]
            code.append(f"    memcpy(out + {off}, {v.b}, sizeof(double) * {d * d});")
            off += d * d
        return code, vals[tmpl.choice.roots[MSG]]

    def apply(self, im, kids: list[_Val], facs, first, fwd, code) -> _Val:
        op, dims = im.op, self.dims
        if op == "leaf":
            return _Val(COND, facs[im.node.arg].scope, "", "", im.node.arg)
        if op == "prior":
            fb = self.fbufs(facs[im.node.arg])
            return _Val(MOMENT, (facs[im.node.arg].child,), fb["b"], fb["Q"])
        if op == "input":
            return fwd
        if op == "cond>info":
            fid = kids[0].fid
            f = facs[fid]
            fb = self.fbufs(f)
            if f.kind != "obs":
                return _Val(INFO, fb["order"], fb["h"], fb["J"])
            Dv, rows = self.D(fb["order"]), len(f.b)
            h = self.buf(Dv, "h")  # Mᵀ Q⁻¹ (y - b)
            code.append(f"    for (int r = 0; r < {Dv}; r++) {{ double a = 0; for (int k = 0; k < {rows}; k++) "
                        f"a += {fb['X']}[k * {Dv} + r] * (ys[{fid - first}][k] - {fb['b']}[k]); {h}[r] = a; }}")
            return _Val(INFO, fb["order"], h, fb["J"])
        if op in ("moment>info", "info>moment"):
            (k,) = kids
            D = self.D(k.vars)
            a, b = self.buf(D, "v"), self.buf(D * D, "M")
            code.append(f"    egfg_spd_inv({D}, {k.b}, {b});")
            code.append(f"    for (int r = 0; r < {D}; r++) {{ double s = 0; for (int c = 0; c < {D}; c++) "
                        f"s += {b}[r * {D} + c] * {k.a}[c]; {a}[r] = s; }}")
            return _Val(INFO if op == "moment>info" else MOMENT, k.vars, a, b)
        if op == "moment_sum":
            (k,) = kids
            rest = tuple(v for v in k.vars if v != im.node.arg)
            ir, D, R = self.idx(k.vars, rest), self.D(k.vars), self.D(rest)
            I = self.ints(ir)
            a, b = self.buf(R, "v"), self.buf(R * R, "M")
            code.append(f"    for (int r = 0; r < {R}; r++) {{ {a}[r] = {k.a}[{I}[r]]; for (int c = 0; c < {R}; c++) "
                        f"{b}[r * {R} + c] = {k.b}[{I}[r] * {D} + {I}[c]]; }}")
            return _Val(MOMENT, rest, a, b)
        if op == "info_sum":
            (k,) = kids
            x = im.node.arg
            rest = tuple(v for v in k.vars if v != x)
            D, R, dx = self.D(k.vars), self.D(rest), dims[x]
            Ix, Ir = self.ints(self.idx(k.vars, (x,))), self.ints(self.idx(k.vars, rest))
            Jxx, B, X = self.buf(dx * dx, "T"), self.buf(dx * (R + 1), "T"), self.buf(dx * (R + 1), "T")
            a, b = self.buf(R, "v"), self.buf(R * R, "M")
            code.append(f"    for (int i = 0; i < {dx}; i++) {{ for (int j = 0; j < {dx}; j++) "
                        f"{Jxx}[i * {dx} + j] = {k.b}[{Ix}[i] * {D} + {Ix}[j]]; for (int c = 0; c < {R}; c++) "
                        f"{B}[i * {R + 1} + c] = {k.b}[{Ix}[i] * {D} + {Ir}[c]]; {B}[i * {R + 1} + {R}] = {k.a}[{Ix}[i]]; }}")
            code.append(f"    egfg_spd_solve({dx}, {R + 1}, {Jxx}, {B}, {X});")
            code.append(f"    for (int r = 0; r < {R}; r++) {{ for (int c = 0; c < {R + 1}; c++) {{ double s = 0; "
                        f"for (int i = 0; i < {dx}; i++) s += {B}[i * {R + 1} + r] * {X}[i * {R + 1} + c]; "
                        f"if (c < {R}) {b}[r * {R} + c] = {k.b}[{Ir}[r] * {D} + {Ir}[c]] - s; "
                        f"else {a}[r] = {k.a}[{Ir}[r]] - s; }} }}")
            return _Val(INFO, rest, a, b)
        if op == "info_mul":
            ka, kb = kids
            out = tuple(sorted(set(ka.vars) | set(kb.vars)))
            D = self.D(out)
            a, b = self.buf(D, "v"), self.buf(D * D, "M")
            code.append(f"    memset({a}, 0, sizeof {a}); memset({b}, 0, sizeof {b});")
            for k in (ka, kb):
                Dk, I = self.D(k.vars), self.ints(self.idx(out, k.vars))
                code.append(f"    for (int r = 0; r < {Dk}; r++) {{ {a}[{I}[r]] += {k.a}[r]; for (int c = 0; c < {Dk}; c++) "
                            f"{b}[{I}[r] * {D} + {I}[c]] += {k.b}[r * {Dk} + c]; }}")
            return _Val(INFO, out, a, b)
        if op == "moment_mul":
            ka, kb = kids
            out = tuple(sorted(ka.vars + kb.vars))
            D = self.D(out)
            a, b = self.buf(D, "v"), self.buf(D * D, "M")
            code.append(f"    memset({b}, 0, sizeof {b});")
            for k in (ka, kb):
                Dk, I = self.D(k.vars), self.ints(self.idx(out, k.vars))
                code.append(f"    for (int r = 0; r < {Dk}; r++) {{ {a}[{I}[r]] = {k.a}[r]; for (int c = 0; c < {Dk}; c++) "
                            f"{b}[{I}[r] * {D} + {I}[c]] = {k.b}[r * {Dk} + c]; }}")
            return _Val(MOMENT, out, a, b)
        m = next(k for k in kids if k.rep == MOMENT)
        f = facs[next(k for k in kids if k.rep == COND).fid]
        fb = self.fbufs(f)
        n = self.D(m.vars)
        rows = len(f.b)
        if op == "predict":
            return self._predict(m, f, fb, n, rows, code)
        if op == "update":
            return self._update(m, f, fb, n, rows, first, facs, code)
        raise ValueError(f"unknown operation {op!r}")

    def _hs(self, m: _Val, f: GFactor, fb, n: int, rows: int, code) -> tuple[str, str]:
        """mean Σ A_i mu[p_i] + b and the cross term C = Σ A_i S[p_i, :] (rows x n)."""
        mc, C = self.buf(rows, "T"), self.buf(rows * n, "T")
        code.append(f"    for (int r = 0; r < {rows}; r++) {mc}[r] = {fb['b']}[r];")
        code.append(f"    memset({C}, 0, sizeof {C});")
        for (A, shape), p in zip(fb["A"], f.parents):
            I = self.ints(self.idx(m.vars, (p,)))
            code.append(f"    for (int r = 0; r < {rows}; r++) for (int k = 0; k < {shape[1]}; k++) {{ double w = {A}[r * {shape[1]} + k]; "
                        f"{mc}[r] += w * {m.a}[{I}[k]]; for (int c = 0; c < {n}; c++) {C}[r * {n} + c] += w * {m.b}[{I}[k] * {n} + c]; }}")
        return mc, C

    def _cc(self, C: str, f: GFactor, fb, m: _Val, n: int, rows: int, code) -> str:
        """Σ_i C[:, p_i] A_iᵀ + Q (rows x rows)."""
        Scc = self.buf(rows * rows, "T")
        code.append(f"    memcpy({Scc}, {fb['Q']}, sizeof {Scc});")
        for (A, shape), p in zip(fb["A"], f.parents):
            I = self.ints(self.idx(m.vars, (p,)))
            code.append(f"    for (int r = 0; r < {rows}; r++) for (int c = 0; c < {rows}; c++) {{ double s = 0; "
                        f"for (int k = 0; k < {shape[1]}; k++) s += {C}[r * {n} + {I}[k]] * {A}[c * {shape[1]} + k]; {Scc}[r * {rows} + c] += s; }}")
        return Scc

    def _predict(self, m, f, fb, n, rows, code) -> _Val:
        mc, C = self._hs(m, f, fb, n, rows, code)
        Scc = self._cc(C, f, fb, m, n, rows, code)
        src = m.vars + (f.child,)
        out = tuple(sorted(src))
        N = n + rows
        P = self.ints(self.idx(src, out))  # position in out -> position in src
        a, b = self.buf(N, "v"), self.buf(N * N, "M")
        # element (i, j) of the joint over src: S, Cᵀ / C, S_cc
        code.append(f"    for (int r = 0; r < {N}; r++) {{ int i = {P}[r]; {a}[r] = i < {n} ? {m.a}[i] : {mc}[i - {n}]; "
                    f"for (int c = 0; c < {N}; c++) {{ int j = {P}[c]; {b}[r * {N} + c] = i < {n} ? (j < {n} ? {m.b}[i * {n} + j] : {C}[(j - {n}) * {n} + i]) "
                    f": (j < {n} ? {C}[(i - {n}) * {n} + j] : {Scc}[(i - {n}) * {rows} + j - {n}]); }} }}")
        return _Val(MOMENT, out, a, b)

    def _update(self, m, f, fb, n, rows, first, facs, code) -> _Val:
        fid = next(i for i, g in enumerate(facs) if g is f)
        pred, HS = self._hs(m, f, fb, n, rows, code)
        Sy = self._cc(HS, f, fb, m, n, rows, code)
        B, X = self.buf(rows * (n + 1), "T"), self.buf(rows * (n + 1), "T")
        code.append(f"    for (int r = 0; r < {rows}; r++) {{ for (int c = 0; c < {n}; c++) {B}[r * {n + 1} + c] = {HS}[r * {n} + c]; "
                    f"{B}[r * {n + 1} + {n}] = ys[{fid - first}][r] - {pred}[r]; }}")
        code.append(f"    egfg_spd_solve({rows}, {n + 1}, {Sy}, {B}, {X});")
        a, b = self.buf(n, "v"), self.buf(n * n, "M")
        # mu + HSᵀ Sy⁻¹ r and S - HSᵀ Sy⁻¹ HS
        code.append(f"    for (int r = 0; r < {n}; r++) for (int c = 0; c <= {n}; c++) {{ double s = 0; "
                    f"for (int k = 0; k < {rows}; k++) s += {HS}[k * {n} + r] * {X}[k * {n + 1} + c]; "
                    f"if (c < {n}) {b}[r * {n} + c] = {m.b}[r * {n} + c] - s; else {a}[r] = {m.a}[r] + s; }}")
        return _Val(MOMENT, m.vars, a, b)


def _shift(vars_):
    return tuple(at(split(v)[0], -1) for v in vars_)


def generate_gaussian_c(program: GFilterProgram) -> str:
    gen = _Gen(program.model)
    head, msg_h = gen.template(program.head, "head", None)
    rep = msg_h.rep
    fvars = _shift(msg_h.vars)
    D = gen.D(msg_h.vars)
    st_a, st_b = gen.buf(D, "state_a"), gen.buf(D * D, "state_b")
    fwd = _Val(rep, fvars, st_a, st_b)
    step, msg_s = gen.template(program.step, "step", fwd)
    if msg_s.rep != rep or _shift(msg_s.vars) != fvars:
        raise ValueError("the head and step messages differ in representation or layout")
    save = lambda v: [f"    memcpy({st_a}, {v.a}, sizeof {st_a});",  # noqa: E731
                      f"    memcpy({st_b}, {v.b}, sizeof {st_b});"]
    lines = [_LIB, *gen.decl, "", "void egfg_setup(const double *const *params) {", *gen.setup, "}", "",
             "void egfg_head(const double *const *ys, double *out) {", *head, *save(msg_h), "}", "",
             "void egfg_step(const double *const *ys, double *out) {", *step, *save(msg_s), "}", ""]
    return "\n".join(lines)


def output_size(model: GaussianDynamicModel) -> int:
    return sum(d + d * d for d in model.dims.values())

"""Filtering in linear-Gaussian dynamic models with representation-aware templates.

The structure of one step is saturated exactly as in the discrete case (phase B's
`local_step` on a structure-only model); the extraction then chooses, for every
chosen value, a representation (moment, information, or a factor's own
conditional form) and the physical operation that computes it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .dynamic import FWD, MSG, DynamicModel, LocalStep, at, local_step, split
from .egraph import SaturationResult, saturate
from .extract import class_leaves, class_scopes
from .gaussian import (
    GFactor,
    Info,
    Moment,
    cost_convert,
    cost_factor_to_info,
    cost_info_mul,
    cost_info_sum,
    cost_predict,
    cost_update,
    dim,
    info_mul,
    info_of_factor,
    info_of_moment,
    info_sum,
    moment_of_info,
    moment_mul_disjoint,
    moment_of_prior,
    moment_predict,
    moment_sum,
    moment_update,
)
from .repextract import Impl, RepChoice, extract_rep

MOMENT, INFO, COND = "moment", "info", "cond"
REPS = {"both": (MOMENT, INFO), "moment": (MOMENT,), "info": (INFO,)}


@dataclass
class GaussianDynamicModel:
    """Vector state variables with dimensions; factors use relative time (@-1, @0).

    `initial`: factors over @0 (priors, or conds among @0 variables); `transition`:
    conds whose child is @0; `observation`: obs factors over @0 variables, whose
    observed values are given per time step.
    """

    dims: dict[str, int]
    initial: list[GFactor]
    transition: list[GFactor]
    observation: list[GFactor]

    def structure(self) -> DynamicModel:
        one = lambda f: (f.scope, np.ones((1,) * len(f.scope)))  # noqa: E731
        return DynamicModel(
            {n: 1 for n in self.dims},
            [one(f) for f in self.initial],
            [one(f) for f in self.transition],
            [f.scope for f in self.observation],
        )

    def local_dims(self) -> dict[str, int]:
        return {at(n, t): d for n, d in self.dims.items() for t in (-1, 0)}

    def factors(self, kind: str) -> list[GFactor]:
        return (self.initial if kind == "head" else self.transition) + self.observation


# ---------------------------------------------------------------------------
# implementations of every (e-class, representation)
# ---------------------------------------------------------------------------


def _contents(g) -> dict[str, frozenset[str]]:
    """Input names below each e-class (like class_leaves, for Input leaves)."""
    out: dict[str, frozenset[str]] = {}
    changed = True
    while changed:
        changed = False
        for cid, nodes in g.classes.items():
            if cid in out:
                continue
            for n in nodes:
                if all(c in out for c in n.children):
                    own = frozenset({n.arg}) if n.op == "input" else frozenset()
                    out[cid] = own.union(*(out[c] for c in n.children))
                    changed = True
                    break
    return out


def normalizable(leaves: frozenset[int], inputs: frozenset[str], facs: list[GFactor], fwd_scope) -> bool:
    """Every variable mentioned below has a generating factor (prior, cond child, or fwd)."""
    mentioned, generated = set(), set()
    for i in leaves:
        f = facs[i]
        mentioned |= set(f.scope)
        if f.kind in ("prior", "cond"):
            generated.add(f.child)
    if FWD in inputs:
        mentioned |= set(fwd_scope)
        generated |= set(fwd_scope)
    return mentioned <= generated


def implementations(
    g, loc: LocalStep, facs: list[GFactor], dims, fwd_rep: str, reps=REPS["both"], amortize: bool = False
) -> dict:
    scopes = class_scopes(g, loc.fg)
    leaves, inputs = class_leaves(g), _contents(g)
    fwd_scope = loc.inputs.get(FWD, frozenset())
    norm = {cid: normalizable(leaves[cid], inputs[cid], facs, fwd_scope) for cid in g.classes}
    allow_moment = MOMENT in reps
    allow_info = INFO in reps
    impls: dict = {}

    def add(im: Impl) -> None:
        if im.out == MOMENT and not norm[im.cid]:
            return
        impls.setdefault(im.state, []).append(im)

    leaf_of = {cid: n for cid, nodes in g.classes.items() for n in nodes if n.op == "leaf"}
    for cid, nodes in g.classes.items():
        D = dim(scopes[cid], dims)
        for n in nodes:
            if n.op == "leaf":
                f = facs[n.arg]
                add(Impl(cid, COND, n, (), 0, "leaf"))
                if f.kind == "prior" and allow_moment:
                    add(Impl(cid, MOMENT, n, (), 0, "prior"))
            elif n.op == "input":
                add(Impl(cid, fwd_rep, n, (), 0, "input"))
            elif n.op == "sum":
                (c,) = n.children
                if allow_moment:
                    add(Impl(cid, MOMENT, n, ((c, MOMENT),), 0, "moment_sum"))
                if allow_info:
                    add(Impl(cid, INFO, n, ((c, INFO),), cost_info_sum(dim(scopes[c], dims), dims[n.arg]), "info_sum"))
            else:
                ca, cb = n.children
                if allow_info:
                    add(Impl(cid, INFO, n, ((ca, INFO), (cb, INFO)), cost_info_mul(D), "info_mul"))
                if allow_moment and not (scopes[ca] & scopes[cb]):
                    add(Impl(cid, MOMENT, n, ((ca, MOMENT), (cb, MOMENT)), 0, "moment_mul"))
                if allow_moment:
                    for m_side, f_side in ((ca, cb), (cb, ca)):
                        if f_side not in leaf_of:
                            continue
                        f = facs[leaf_of[f_side].arg]
                        S = scopes[m_side]
                        kids = tuple((c, MOMENT if c == m_side else COND) for c in n.children)
                        if m_side == f_side:
                            continue
                        if f.kind == "cond" and set(f.parents) <= S and f.child not in S:
                            cost = cost_predict(dim(f.parents, dims), dims[f.child], dim(S, dims))
                            add(Impl(cid, MOMENT, n, kids, cost, "predict"))
                        elif f.kind == "obs" and set(f.parents) <= S:
                            add(Impl(cid, MOMENT, n, kids, cost_update(dim(S, dims), len(f.b)), "update"))
        # conversions within the e-class
        if cid in leaf_of and allow_info:
            add(Impl(cid, INFO, None, ((cid, COND),), cost_factor_to_info(facs[leaf_of[cid].arg], dims, amortize), "cond>info"))
        if allow_info and allow_moment:
            add(Impl(cid, INFO, None, ((cid, MOMENT),), cost_convert(D), "moment>info"))
        if allow_info:  # info -> moment is needed to read answers even when only info is "allowed"
            add(Impl(cid, MOMENT, None, ((cid, INFO),), cost_convert(D), "info>moment"))
    return impls


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


@dataclass
class GTemplate:
    local: LocalStep
    facs: list[GFactor]
    choice: RepChoice
    fwd_rep: str
    saturation: SaturationResult
    prep_cost: int = 0
    latency_cost: int = 0
    counts: dict = field(default_factory=dict)  # physical operation -> number used

    @property
    def cost(self) -> int:
        return self.choice.cost


@dataclass
class GFilterProgram:
    model: GaussianDynamicModel
    head: GTemplate
    step: GTemplate
    search_s: float

    @property
    def fwd_rep(self) -> str:
        return self.step.fwd_rep


def _template(sat, loc, facs, dims, fwd_rep, reps, method, weights_mode, time_limit_s, amortize):
    g = sat.graph
    impls = implementations(g, loc, facs, dims, fwd_rep, reps, amortize)
    roots = {q: (cid, fwd_rep if q == MSG else MOMENT) for q, cid in g.roots.items()}
    leaves = class_leaves(g)
    weights = None
    if weights_mode == "latency":
        base = extract_rep(g, impls, roots, "tree")
        M = 10 * base.cost + 1
        weights = {cid: M if leaves[cid] & loc.obs_ids else 1 for cid in g.classes}
    ch = extract_rep(g, impls, roots, method, weights, time_limit_s)
    if weights is not None:
        # local search may get stuck from the weighted start: the total-cost
        # solution is a valid candidate too; keep whichever is lower when weighted
        plain = extract_rep(g, impls, roots, method, None, time_limit_s)
        plain_w = sum(im.cost * weights.get(s[0], 1) for s, im in plain.choice.items())
        if plain_w < ch.weighted_cost:
            ch = RepChoice(plain.choice, plain.roots, plain.cost, plain_w, plain.optimal, plain.seconds)
    prep = sum(im.cost for s, im in ch.choice.items() if not (leaves[s[0]] & loc.obs_ids))
    counts: dict[str, int] = {}
    for im in ch.choice.values():
        counts[im.op] = counts.get(im.op, 0) + 1
    return GTemplate(loc, facs, ch, fwd_rep, sat, prep, ch.cost - prep, counts)


def compile_gaussian_filter(
    model: GaussianDynamicModel,
    reps: str = "both",
    rules: str = "full",
    seed: bool = True,
    extractor: str = "ilp",
    objective: str = "total",
    max_iters: int = 30,
    node_limit: int = 50_000,
    time_limit_s: float = 60,
    amortize_constants: bool = False,
) -> GFilterProgram:
    """Search the head and step templates; choose the message representation too.

    `amortize_constants`: count the parameter-only parts of putting factors in
    information form once for the whole run instead of at every step.
    """
    start = time.perf_counter()
    method = {"ilp": "ilp", "greedy": "greedy", "tree": "tree"}[extractor]
    dims = model.local_dims()
    struct = model.structure()
    sats = {}
    for kind in ("head", "step"):
        loc = local_step(struct, kind)
        sat = saturate(loc.fg, loc.queries, max_iters=max_iters, node_limit=node_limit, rules=rules,
                       inputs=loc.inputs, seeds=loc.seeds if seed else None)
        sats[kind] = (loc, sat)
    best = None
    for r in REPS[reps]:
        t = {k: _template(sats[k][1], sats[k][0], model.factors(k), dims, r, REPS[reps], method,
                          objective, time_limit_s, amortize_constants) for k in ("head", "step")}
        key = (t["step"].choice.weighted_cost, t["head"].choice.weighted_cost)
        if best is None or key < best[0]:
            best = (key, t)
    t = best[1]
    return GFilterProgram(model, t["head"], t["step"], time.perf_counter() - start)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _evaluate(tmpl: GTemplate, dims, ys: list[np.ndarray], fwd) -> dict[str, Moment | Info]:
    facs = tmpl.facs
    nobs = len(ys)
    first = len(facs) - nobs
    y_of = {first + k: y for k, y in enumerate(ys)}
    vals: dict = {}

    def value(s):
        stack = [s]
        while stack:
            cur = stack[-1]
            if cur in vals:
                stack.pop()
                continue
            im = tmpl.choice.choice[cur]
            todo = [k for k in im.kids if k not in vals]
            if todo:
                stack.extend(todo)
                continue
            vals[cur] = _apply(im, [vals[k] for k in im.kids], facs, y_of, dims, fwd)
            stack.pop()
        return vals[s]

    return {q: value(s) for q, s in tmpl.choice.roots.items()}


def _apply(im: Impl, kids, facs, y_of, dims, fwd):
    op = im.op
    if op == "leaf":
        return ("factor", im.node.arg)
    if op == "prior":
        return moment_of_prior(facs[im.node.arg])
    if op == "input":
        return fwd
    if op == "cond>info":
        fid = kids[0][1]
        return info_of_factor(facs[fid], dims, y_of.get(fid))
    if op == "moment>info":
        return info_of_moment(kids[0])
    if op == "info>moment":
        return moment_of_info(kids[0])
    if op == "moment_sum":
        return moment_sum(kids[0], im.node.arg, dims)
    if op == "info_sum":
        return info_sum(kids[0], im.node.arg, dims)
    if op == "info_mul":
        return info_mul(kids[0], kids[1], dims)
    if op == "moment_mul":
        return moment_mul_disjoint(kids[0], kids[1], dims)
    m = next(k for k in kids if isinstance(k, Moment))
    fid = next(k for k in kids if isinstance(k, tuple))[1]
    if op == "predict":
        return moment_predict(m, facs[fid], dims)
    if op == "update":
        return moment_update(m, facs[fid], y_of[fid], dims)
    raise ValueError(f"unknown operation {op!r}")


def _shift_back(v):
    names = tuple(at(split(x)[0], -1) for x in v.vars)
    if isinstance(v, Moment):
        return Moment(names, v.mu, v.S, v.logz)
    return Info(names, v.J, v.h, v.g)


@dataclass
class GaussianFilterResult:
    mean: dict[str, np.ndarray]
    cov: dict[str, np.ndarray]
    loglik: float


class GaussianFilter:
    """Online filtering: feed each step's observation vectors (one per obs factor)."""

    def __init__(self, program: GFilterProgram):
        self.program = program
        self.dims = program.model.local_dims()
        self.t = 0
        self._msg = None

    def step(self, ys: list[np.ndarray]) -> GaussianFilterResult:
        tmpl = self.program.head if self.t == 0 else self.program.step
        out = _evaluate(tmpl, self.dims, ys, self._msg)
        self._msg = _shift_back(out[MSG])
        self.t += 1
        names = sorted(self.program.model.dims)
        mean = {n: out[n].mu for n in names}
        cov = {n: out[n].S for n in names}
        return GaussianFilterResult(mean, cov, out[names[0]].logz)


def gaussian_filter(program: GFilterProgram, obs: list[list[np.ndarray]]) -> list[GaussianFilterResult]:
    f = GaussianFilter(program)
    return [f.step(ys) for ys in obs]


# ---------------------------------------------------------------------------
# stacking, baselines and the reference Kalman filter
# ---------------------------------------------------------------------------


def stack(model: GaussianDynamicModel) -> GaussianDynamicModel:
    """The same model with the whole state as one vector variable `s` (blocks in name order).

    Needs priors as initial factors and transitions whose parents are all at @-1.
    """
    names = sorted(model.dims)
    off = {}
    k = 0
    for n in names:
        off[n] = k
        k += model.dims[n]
    N = k
    sl = lambda n: slice(off[n], off[n] + model.dims[n])  # noqa: E731

    mu0, P0 = np.zeros(N), np.zeros((N, N))
    for f in model.initial:
        if f.kind != "prior":
            raise ValueError("stack() needs priors as initial factors")
        n = split(f.child)[0]
        mu0[sl(n)], P0[sl(n), sl(n)] = f.b, f.Q
    A, b, Q = np.zeros((N, N)), np.zeros(N), np.zeros((N, N))
    for f in model.transition:
        c = split(f.child)[0]
        for Ai, p in zip(f.A, f.parents):
            pn, pt = split(p)
            if pt != -1:
                raise ValueError("stack() needs transitions from @-1 only")
            A[sl(c), sl(pn)] += Ai
        b[sl(c)], Q[sl(c), sl(c)] = f.b, f.Q
    ms = [len(f.b) for f in model.observation]
    M = sum(ms)
    H, c, R = np.zeros((M, N)), np.zeros(M), np.zeros((M, M))
    r = 0
    for f, m in zip(model.observation, ms):
        for Ai, p in zip(f.A, f.parents):
            H[r : r + m, sl(split(p)[0])] += Ai
        c[r : r + m], R[r : r + m, r : r + m] = f.b, f.Q
        r += m
    return GaussianDynamicModel(
        {"s": N},
        [GFactor("prior", "s@0", (), (), mu0, P0)],
        [GFactor("cond", "s@0", ("s@-1",), (A,), b, Q)],
        [GFactor("obs", None, ("s@0",), (H,), c, R)],
    )


def stack_obs(obs: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    return [[np.concatenate(ys)] for ys in obs]


def kalman_reference(model: GaussianDynamicModel, obs) -> list[GaussianFilterResult]:
    """Textbook Kalman filter on the stacked model, written directly with numpy."""
    st = stack(model)
    (p,), (tr,), (ob,) = st.initial, st.transition, st.observation
    names = sorted(model.dims)
    off = np.cumsum([0] + [model.dims[n] for n in names])
    mu, P = p.b.copy(), p.Q.copy()
    A, (H,) = tr.A[0], ob.A
    out = []
    loglik = 0.0
    for t, ys in enumerate(obs):
        y = np.concatenate(ys)
        if t > 0:
            mu, P = A @ mu + tr.b, A @ P @ A.T + tr.Q
        S = H @ P @ H.T + ob.Q
        r = y - (H @ mu + ob.b)
        K = P @ H.T @ np.linalg.inv(S)
        mu, P = mu + K @ r, P - K @ H @ P
        loglik += -0.5 * (r @ np.linalg.solve(S, r) + len(r) * np.log(2 * np.pi) + np.linalg.slogdet(S)[1])
        out.append(GaussianFilterResult(
            {n: mu[off[i] : off[i + 1]] for i, n in enumerate(names)},
            {n: P[off[i] : off[i + 1], off[i] : off[i + 1]] for i, n in enumerate(names)},
            float(loglik),
        ))
    return out


def kalman_program(model: GaussianDynamicModel, **kw) -> GFilterProgram:
    """The standard Kalman filter: the stacked model in moment form only.

    The stacked model leaves nothing to choose but the order of the update, so
    tree extraction is the default.
    """
    kw.setdefault("extractor", "tree")
    return compile_gaussian_filter(stack(model), reps="moment", **kw)


def information_program(model: GaussianDynamicModel, **kw) -> GFilterProgram:
    """The information filter: the stacked model in information form only (tree extraction by default)."""
    kw.setdefault("extractor", "tree")
    return compile_gaussian_filter(stack(model), reps="info", **kw)

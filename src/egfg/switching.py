"""Switching linear-Gaussian models: discrete modes and continuous states (phase H).

Values are conditional-Gaussian (CG): for every assignment of the discrete
variables in scope, a Gaussian over the continuous ones (phase D's moment or
information form, with its normalizing constant). Products, integrals over
continuous variables, and sums over discrete variables with no continuous
variable left are exact. A sum over a discrete variable with continuous ones
left is a mixture, which this representation cannot hold: the only
implementation is `collapse`, the moment-matching projection (an approximation).

The e-graph stays exact (structure only, as in phase D); `collapse` is one of
the implementations the representation-aware extraction may choose, with a
penalty per merged component. At run time every value carries an upper bound on
its total-variation distance to the exact value (0 when exact).
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .dynamic import DynamicModel, _local_fg, _substitute, at, split
from .extract import class_leaves, class_scopes
from .gaussian import (
    LOG2PI,
    GFactor,
    Info,
    Moment,
    cost_convert,
    cost_factor_to_info,
    cost_info_mul,
    cost_info_sum,
    cost_predict,
    cost_update,
    info_mul,
    info_of_factor,
    info_of_moment,
    info_sum,
    moment_mul_disjoint,
    moment_of_info,
    moment_predict,
    moment_sum,
    moment_update,
)
from .gdynamic import COND, INFO, MOMENT, _contents
from .ir import Input, Leaf, Term
from .jtree import Piece, TreeTerms, junction_tree, product, sum_out
from .model import Factor, FactorGraph
from .repextract import Impl, RepChoice, extract_rep

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DTable:
    """A table over discrete variables; `child` set when it is a conditional (normalized over it)."""

    scope: tuple[str, ...]
    table: np.ndarray
    child: str | None = None


@dataclass(frozen=True)
class CLG:
    """A conditional linear-Gaussian factor with discrete parents: one GFactor per assignment
    of `dparents` (keys in `dparents` order). Kinds as GFactor: prior, cond, obs."""

    kind: str
    child: str | None
    parents: tuple[str, ...]
    dparents: tuple[str, ...]
    params: dict

    @property
    def scope(self) -> tuple[str, ...]:
        return self.dparents + next(iter(self.params.values())).scope


@dataclass
class SwitchingModel:
    """Discrete states (`cards`) and continuous ones (`dims`); factors use relative time."""

    cards: dict[str, int]
    dims: dict[str, int]
    initial: list
    transition: list
    observation: list  # CLG obs factors; their observed vectors are given per step

    def structure(self) -> DynamicModel:
        one = lambda f: (f.scope, np.ones((1,) * len(f.scope)))  # noqa: E731
        names = {n: 1 for n in list(self.cards) + list(self.dims)}
        return DynamicModel(names, [one(f) for f in self.initial], [one(f) for f in self.transition],
                            [f.scope for f in self.observation])

    def factors(self, kind: str) -> list:
        return (self.initial if kind == "head" else self.transition) + self.observation

    def local_cards(self) -> dict[str, int]:
        return {at(n, t): k for n, k in self.cards.items() for t in (-1, 0)}

    def local_dims(self) -> dict[str, int]:
        return {at(n, t): d for n, d in self.dims.items() for t in (-1, 0)}

    def components(self) -> list[list[str]]:
        """State names linked by some factor (independent groups), in name order."""
        names = sorted(list(self.cards) + list(self.dims))
        parent = {n: n for n in names}

        def find(n):
            while parent[n] != n:
                n = parent[n]
            return n

        for f in self.initial + self.transition + self.observation:
            ns = [split(v)[0] for v in f.scope]
            for a in ns[1:]:
                parent[find(a)] = find(ns[0])
        groups: dict[str, list[str]] = {}
        for n in names:
            groups.setdefault(find(n), []).append(n)
        return sorted(groups.values())


# ---------------------------------------------------------------------------
# CG values
# ---------------------------------------------------------------------------


@dataclass
class CG:
    dvars: tuple[str, ...]  # sorted
    comps: dict  # assignment (in dvars order) -> Moment | Info over the same continuous variables
    tau: float = 0.0  # upper bound on the TV distance to the exact value
    kernel_child: str | None = None  # set when the value is an exact conditional over this new child
    kl: float = 0.0  # for a collapse: the bound on KL(exact ‖ result) it adds (this step's local error)
    kls: dict = field(default_factory=dict)  # (step, state) -> bound, for every collapse this value depends on

    @property
    def klsum(self) -> float:
        """The sum of the KL bounds of the collapses this value depends on (each counted once)."""
        return float(sum(self.kls.values()))

    @property
    def form(self) -> str:
        return MOMENT if isinstance(next(iter(self.comps.values())), Moment) else INFO

    @property
    def cvars(self) -> tuple[str, ...]:
        return next(iter(self.comps.values())).vars

    @property
    def vars(self) -> frozenset[str]:
        return frozenset(self.dvars) | frozenset(self.cvars)


def _configs(dvars, cards):
    return itertools.product(*[range(cards[v]) for v in dvars])


def _sub(cfg, dvars, want) -> tuple[int, ...]:
    pos = {v: i for i, v in enumerate(dvars)}
    return tuple(cfg[pos[v]] for v in want)


def _logsumexp(xs) -> float:
    xs = np.asarray(list(xs), float)
    m = np.max(xs)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(xs - m))))


def _weight(v) -> float:
    return v.logz if isinstance(v, Moment) else v.g


def _with_weight(v, w: float):
    if isinstance(v, Moment):
        return Moment(v.vars, v.mu, v.S, w)
    return Info(v.vars, v.J, v.h, w)


def scalar(w: float, form: str):
    return Moment((), np.zeros(0), np.zeros((0, 0)), w) if form == MOMENT else Info((), np.zeros((0, 0)), np.zeros(0), w)


def cg_pointwise(a: CG, b: CG, cards, op) -> CG:
    out = tuple(sorted(set(a.dvars) | set(b.dvars)))
    comps = {c: op(a.comps[_sub(c, out, a.dvars)], b.comps[_sub(c, out, b.dvars)]) for c in _configs(out, cards)}
    return CG(out, comps)


def table_value(f: DTable, cards, form: str) -> CG:
    dv = tuple(sorted(f.scope))
    perm = [f.scope.index(v) for v in dv]
    t = np.transpose(np.asarray(f.table, float), perm)
    with np.errstate(divide="ignore"):
        comps = {c: scalar(float(np.log(t[c])), form) for c in _configs(dv, cards)}
    return CG(dv, comps, kernel_child=f.child)


def _factor_at(f: CLG, cfg, dvars) -> GFactor:
    return f.params[_sub(cfg, dvars, f.dparents)]


def clg_prior(f: CLG, cards) -> CG:
    dv = tuple(sorted(f.dparents))
    return CG(dv, {c: (lambda g: Moment((g.child,), np.asarray(g.b, float), np.asarray(g.Q, float), 0.0))(_factor_at(f, c, dv))
                   for c in _configs(dv, cards)}, kernel_child=f.child)


def clg_info(f: CLG, cards, dims, y=None) -> CG:
    dv = tuple(sorted(f.dparents))
    comps = {c: info_of_factor(_factor_at(f, c, dv), dims, y) for c in _configs(dv, cards)}
    return CG(dv, comps, kernel_child=f.child if f.kind != "obs" else None)


def cg_predict(m: CG, f: CLG, cards, dims) -> CG:
    out = tuple(sorted(set(m.dvars) | set(f.dparents)))
    comps = {c: moment_predict(m.comps[_sub(c, out, m.dvars)], f.params[_sub(c, out, f.dparents)], dims)
             for c in _configs(out, cards)}
    return CG(out, comps)


def cg_update(m: CG, f: CLG, y, cards, dims) -> CG:
    out = tuple(sorted(set(m.dvars) | set(f.dparents)))
    comps = {c: moment_update(m.comps[_sub(c, out, m.dvars)], f.params[_sub(c, out, f.dparents)], y, dims)
             for c in _configs(out, cards)}
    return CG(out, comps)


def cg_dsum(v: CG, x: str, cards) -> CG:
    """Exact sum over a discrete variable when no continuous variable is left."""
    if v.cvars:
        raise ValueError("an exact discrete sum needs a value without continuous variables")
    rest = tuple(d for d in v.dvars if d != x)
    k = v.dvars.index(x)
    comps = {}
    for c in _configs(rest, cards):
        ws = [_weight(v.comps[c[:k] + (i,) + c[k:]]) for i in range(cards[x])]
        comps[c] = scalar(_logsumexp(ws), v.form)
    return CG(rest, comps, v.tau, kls=v.kls)


def kl_gauss(a: Moment, b: Moment) -> float:
    """KL(N(a) ‖ N(b)) of the normalized Gaussians."""
    d = len(a.mu)
    if d == 0:
        return 0.0
    Sb_inv = np.linalg.inv(b.S)
    diff = b.mu - a.mu
    return 0.5 * float(np.trace(Sb_inv @ a.S) + diff @ Sb_inv @ diff - d
                       + np.linalg.slogdet(b.S)[1] - np.linalg.slogdet(a.S)[1])


def collapse_one(parts: list[Moment]) -> tuple[Moment, float]:
    """Moment matching of a weighted mixture; returns the Gaussian and Σ_i w_i KL(N_i ‖ q)."""
    logw = np.array([p.logz for p in parts])
    total = _logsumexp(logw)
    if not np.isfinite(total):
        return Moment(parts[0].vars, parts[0].mu, parts[0].S, total), 0.0
    w = np.exp(logw - total)
    mu = sum(wi * p.mu for wi, p in zip(w, parts))
    S = sum(wi * (p.S + np.outer(p.mu - mu, p.mu - mu)) for wi, p in zip(w, parts))
    q = Moment(parts[0].vars, mu, S, total)
    kl = float(sum(wi * kl_gauss(p, q) for wi, p in zip(w, parts) if wi > 0))
    return q, kl


def cg_collapse(v: CG, x: str, cards) -> CG:
    """The moment-matching projection of Σ_x v (a mixture over x for each remaining assignment)."""
    rest = tuple(d for d in v.dvars if d != x)
    k = v.dvars.index(x)
    comps, kls = {}, {}
    for c in _configs(rest, cards):
        q, kl = collapse_one([v.comps[c[:k] + (i,) + c[k:]] for i in range(cards[x])])
        comps[c], kls[c] = q, kl
    # KL(p ‖ q) of the normalized joints: the discrete marginal is kept exactly
    W = _logsumexp([q.logz for q in comps.values()])
    kl = sum(math.exp(comps[c].logz - W) * kls[c] for c in comps) if np.isfinite(W) else 0.0
    return CG(rest, comps, min(1.0, v.tau + math.sqrt(max(kl, 0.0) / 2)), kl=max(kl, 0.0), kls=v.kls)


def cg_convert(v: CG, to: str) -> CG:
    f = info_of_moment if to == INFO else moment_of_info
    return CG(v.dvars, {c: f(p) for c, p in v.comps.items()}, v.tau, v.kernel_child, kls=v.kls)


def log_mass(v: CG) -> float | None:
    """log of the total mass, None when some component is not normalizable."""
    out = []
    for p in v.comps.values():
        if isinstance(p, Moment):
            out.append(p.logz)
            continue
        if len(p.h) == 0:
            out.append(p.g)
            continue
        if np.min(np.linalg.eigvalsh(p.J)) <= 1e-12:
            return None
        out.append(moment_of_info(p).logz)
    return _logsumexp(out)


def log_sup(v: CG) -> float | None:
    out = []
    for p in v.comps.values():
        if isinstance(p, Moment):
            out.append(p.logz - 0.5 * (len(p.mu) * LOG2PI + np.linalg.slogdet(p.S)[1]) if len(p.mu) else p.logz)
        elif len(p.h) == 0:
            out.append(p.g)
        else:
            x = np.linalg.lstsq(p.J, p.h, rcond=None)[0]
            if np.linalg.norm(p.J @ x - p.h) > 1e-9 * (1 + np.linalg.norm(p.h)):
                return None  # unbounded
            out.append(p.g + 0.5 * float(p.h @ x))
    return max(out)


def product_tau(approx: CG, exact: CG | None, out: CG, sup_exact: float | None = None) -> float:
    """TV bound after multiplying an approximate value by an exact one (see the module doc)."""
    if approx.tau == 0:
        return 0.0
    if exact is not None and exact.kernel_child is not None and exact.kernel_child not in approx.vars:
        return approx.tau
    ls = sup_exact if sup_exact is not None else (log_sup(exact) if exact is not None else None)
    la, lo = log_mass(approx), log_mass(out)
    if ls is None or la is None or lo is None or not np.isfinite(lo):
        return 1.0
    return min(1.0, 2 * approx.tau * math.exp(ls + la - lo))


def normalize(v: CG) -> CG:
    lm = log_mass(v)
    if lm is None or not np.isfinite(lm):
        return v
    return CG(v.dvars, {c: _with_weight(p, _weight(p) - lm) for c, p in v.comps.items()}, v.tau, v.kernel_child, kls=v.kls)


def shift_back(v: CG) -> CG:
    ren = lambda xs: tuple(at(split(x)[0], -1) for x in xs)  # noqa: E731
    comps = {}
    for c, p in v.comps.items():
        comps[c] = Moment(ren(p.vars), p.mu, p.S, p.logz) if isinstance(p, Moment) else Info(ren(p.vars), p.J, p.h, p.g)
    return CG(ren(v.dvars), comps, v.tau, kls=v.kls)


# ---------------------------------------------------------------------------
# local problems with the message split into groups
# ---------------------------------------------------------------------------


@dataclass
class SLocalStep:
    kind: str
    fg: FactorGraph
    inputs: dict[str, frozenset[str]]
    queries: dict[str, Term]
    seeds: dict[str, Term]
    obs_ids: frozenset[int]
    groups: list[tuple[str, ...]]  # state names of each message part: msg{k}, fwd{k}


def local_step(model: SwitchingModel, kind: str, groups: list[tuple[str, ...]]) -> SLocalStep:
    struct = model.structure()
    fg = _local_fg(struct, kind)
    boundary = set(struct.boundary())
    groups = [g for g in groups if set(g) & boundary]
    inputs = {f"fwd{k}": frozenset(at(n, -1) for n in g if n in boundary) for k, g in enumerate(groups)} if kind == "step" else {}
    pieces: list[Piece] = [(Leaf(f.id), frozenset(f.scope), frozenset({f.id})) for f in fg.factors]
    pieces += [(Input(name), s, frozenset()) for name, s in sorted(inputs.items())]
    whole = product(pieces)
    keep = {f"msg{k}": frozenset(at(n, 0) for n in g if n in boundary) for k, g in enumerate(groups)}
    queries: dict[str, Term] = {q: sum_out(whole, s)[0] for q, s in keep.items()}
    names = sorted(list(model.cards) + list(model.dims))
    for n in names:
        queries[n] = sum_out(whole, {at(n, 0)})[0]
    # seeds: the junction tree of the step, with stand-in factors for the inputs and helpers for the parts
    extra = list(fg.factors)
    stand = {}
    for name, s in sorted(inputs.items()):
        stand[name] = len(extra)
        extra.append(Factor(len(extra), tuple(sorted(s)), np.ones([1] * len(s))))
    helpers = set()
    for s in keep.values():
        if len(s) > 1:
            helpers.add(len(extra))
            extra.append(Factor(len(extra), tuple(sorted(s)), np.ones([1] * len(s))))
    aug = FactorGraph(fg.cards, extra)
    jt = junction_tree(aug)
    for i in jt.assigned:
        jt.assigned[i] = [f for f in jt.assigned[i] if f not in helpers]
    calc = TreeTerms(aug, jt)
    seeds = {q: calc.joint(set(s))[0] for q, s in keep.items()}
    for n in names:
        seeds[n] = calc.marginal(at(n, 0))[0]
    for name, fid in stand.items():
        seeds = {q: _substitute(t, fid, Input(name)) for q, t in seeds.items()}
    nfirst = len(struct.initial) if kind == "head" else len(struct.transition)
    return SLocalStep(kind, fg, inputs, queries, seeds, frozenset(range(nfirst, len(fg.factors))), groups)


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------


def _ncfg(dvars, cards) -> int:
    return int(np.prod([cards[v] for v in dvars])) if dvars else 1


def implementations(g, loc: SLocalStep, model: SwitchingModel, fwd_rep: str, lam: float,
                    price: float | None = None, bound=None) -> tuple[dict, dict, dict]:
    """(impls by state, flops by impl, expected KL bound by collapse impl when `bound` is given).
    The extraction cost of `collapse` adds λ per merged component, times (1 + the number of this step's observations not yet in the merged value).
    With `price` (μ), it adds μ × the expected KL bound of that collapse instead (`bound`, an
    `ExpectedBound`; phase I)."""
    cards, dims = model.local_cards(), model.local_dims()
    facs = model.factors(loc.kind)
    scopes = class_scopes(g, loc.fg)
    leaves, contents = class_leaves(g), _contents(g)
    dpart = {cid: tuple(sorted(v for v in s if v in cards)) for cid, s in scopes.items()}
    cpart = {cid: tuple(sorted(v for v in s if v in dims)) for cid, s in scopes.items()}
    D = {cid: sum(dims[v] for v in cpart[cid]) for cid in scopes}
    n = {cid: _ncfg(dpart[cid], cards) for cid in scopes}

    def generated(cid) -> bool:
        mentioned, gen = set(cpart[cid]), set()
        for i in leaves[cid]:
            f = facs[i]
            mentioned |= {v for v in f.scope if v in dims}
            if isinstance(f, CLG) and f.kind in ("prior", "cond"):
                gen.add(f.child)
        for name in contents[cid]:
            s = loc.inputs.get(name, frozenset())
            mentioned |= {v for v in s if v in dims}
            gen |= {v for v in s if v in dims}
        return mentioned <= gen

    norm = {cid: generated(cid) for cid in scopes}
    # collapses that only give an answer (a state's marginal over all modes) and not a message:
    # every program needs them, and they do not change what is passed on, so they are not penalized
    msg_roots = {cid for q, cid in g.roots.items() if q.startswith("msg")}
    output_only = {cid for q, cid in g.roots.items() if not q.startswith("msg")} - msg_roots
    impls: dict = {}
    flops: dict = {}
    errs: dict = {}

    def add(cid, out, node, kids, fl, op, penalty=0.0):
        if out == MOMENT and not norm[cid]:
            return None
        im = Impl(cid, out, node, tuple(kids), fl + penalty if penalty else int(fl), op)
        impls.setdefault(im.state, []).append(im)
        flops[im] = int(fl)
        return im

    leaf_of = {cid: nd for cid, nodes in g.classes.items() for nd in nodes if nd.op == "leaf"}
    for cid, nodes in g.classes.items():
        for nd in nodes:
            if nd.op == "leaf":
                f = facs[nd.arg]
                if isinstance(f, DTable):
                    add(cid, MOMENT, nd, (), 0, "table")
                    add(cid, INFO, nd, (), 0, "table")
                else:
                    add(cid, COND, nd, (), 0, "leaf")
                    if f.kind == "prior":
                        add(cid, MOMENT, nd, (), 0, "prior")
            elif nd.op == "input":
                add(cid, fwd_rep, nd, (), 0, "input")
            elif nd.op == "sum":
                (c,) = nd.children
                x = nd.arg
                if x in dims:
                    add(cid, MOMENT, nd, ((c, MOMENT),), 0, "moment_sum")
                    add(cid, INFO, nd, ((c, INFO),), n[c] * cost_info_sum(D[c], dims[x]), "info_sum")
                elif not cpart[c]:
                    add(cid, MOMENT, nd, ((c, MOMENT),), n[c], "dsum")
                    add(cid, INFO, nd, ((c, INFO),), n[c], "dsum")
                else:
                    # an error made before an observation is multiplied in gets amplified by it
                    # (see product_tau): count the observations not yet in the collapsed value
                    eps = None
                    if bound is not None:
                        first = len(facs) - len(model.observation)
                        obs = [i - first for i in loc.obs_ids & leaves[c]]
                        eps = bound(bound.site(dpart[c], x, cpart[c], obs, loc.kind == "head"))
                    if cid in output_only:
                        penalty = 0.0
                    elif price is not None:
                        penalty = price * eps
                    else:
                        missing = len(loc.obs_ids - leaves[c])
                        penalty = lam * n[c] * (1 + missing)
                    im = add(cid, MOMENT, nd, ((c, MOMENT),), n[c] * (D[c] + D[c] ** 2), "collapse", penalty)
                    if im is not None and eps is not None and cid not in output_only:
                        errs[im] = eps
            else:
                ca, cb = nd.children
                add(cid, INFO, nd, ((ca, INFO), (cb, INFO)), n[cid] * cost_info_mul(D[cid]), "info_mul")
                if not (set(cpart[ca]) & set(cpart[cb])):
                    add(cid, MOMENT, nd, ((ca, MOMENT), (cb, MOMENT)), 0, "moment_mul")
                for m_side, f_side in ((ca, cb), (cb, ca)):
                    if f_side not in leaf_of or m_side == f_side:
                        continue
                    f = facs[leaf_of[f_side].arg]
                    if not isinstance(f, CLG):
                        continue
                    S = set(cpart[m_side])
                    kids = tuple((c, MOMENT if c == m_side else COND) for c in nd.children)
                    g0 = next(iter(f.params.values()))
                    pdim = sum(dims[p] for p in f.parents)
                    if f.kind == "cond" and set(f.parents) <= S and f.child not in S:
                        add(cid, MOMENT, nd, kids, n[cid] * cost_predict(pdim, dims[f.child], D[m_side]), "predict")
                    elif f.kind == "obs" and set(f.parents) <= S:
                        add(cid, MOMENT, nd, kids, n[cid] * cost_update(D[m_side], len(g0.b)), "update")
        if cid in leaf_of and isinstance(facs[leaf_of[cid].arg], CLG):
            f = facs[leaf_of[cid].arg]
            g0 = next(iter(f.params.values()))
            add(cid, INFO, None, ((cid, COND),), n[cid] * cost_factor_to_info(g0, dims), "cond>info")
        add(cid, INFO, None, ((cid, MOMENT),), n[cid] * cost_convert(D[cid]), "moment>info")
        add(cid, MOMENT, None, ((cid, INFO),), n[cid] * cost_convert(D[cid]), "info>moment")
    return impls, flops, errs


# ---------------------------------------------------------------------------
# templates and compilation
# ---------------------------------------------------------------------------


@dataclass
class STemplate:
    local: SLocalStep
    facs: list
    choice: RepChoice
    fwd_rep: str
    flops: int
    counts: dict = field(default_factory=dict)
    eps: float | None = None  # with an ExpectedBound: the summed expected KL bounds of its collapses (messages only)


@dataclass
class SFilterProgram:
    model: SwitchingModel
    head: STemplate
    step: STemplate
    groups: list[tuple[str, ...]]
    lam: float
    search_s: float
    price: float | None = None

    @property
    def collapses(self) -> int:
        return self.step.counts.get("collapse", 0)


def _template(sat, loc, model, fwd_rep, lam, method, time_limit_s, price=None, bound=None) -> STemplate:
    g = sat.graph
    impls, flops, errs = implementations(g, loc, model, fwd_rep, lam, price, bound)
    roots = {q: (cid, fwd_rep if q.startswith("msg") else MOMENT) for q, cid in g.roots.items()}
    ch = extract_rep(g, impls, roots, method, None, time_limit_s)
    counts: dict[str, int] = {}
    for im in ch.choice.values():
        counts[im.op] = counts.get(im.op, 0) + 1
    eps = sum(errs.get(im, 0.0) for im in ch.choice.values()) if bound is not None else None
    return STemplate(loc, model.factors(loc.kind), ch, fwd_rep, sum(flops[im] for im in ch.choice.values()), counts, eps)


def message_groups(model: SwitchingModel) -> list[list[tuple[str, ...]]]:
    """Every way to pass the message: per independent component, either whole (a Gaussian for every
    mode) or split into its discrete and its continuous part (GPB1's message)."""
    options = []
    for comp in model.components():
        d = tuple(n for n in comp if n in model.cards)
        c = tuple(n for n in comp if n in model.dims)
        opts = [[tuple(comp)]]
        if d and c:
            opts.append([d, c])
        options.append(opts)
    return [sum(choice, []) for choice in itertools.product(*options)]


def compile_switching_programs(model: SwitchingModel, lams=(0.0, 1e6), uniform: bool = False,
                               prices=None, **kw) -> list[SFilterProgram]:
    """The best program for every message form and λ (duplicates removed): candidates that trade
    operations for approximation differently (the message form sets how many Gaussians are kept;
    λ, where the mixtures are merged). `uniform`: only the forms with the same choice for every
    component."""
    out, seen = [], set()
    forms = message_groups(model)
    if uniform:
        forms = [forms[0], forms[-1]] if len(forms) > 1 else forms
    if prices is not None and "bound" not in kw:
        from .switching_bound import ExpectedBound

        kw["bound"] = ExpectedBound(model)
    settings = [dict(price=pr) for pr in prices] if prices is not None else [dict(lam=lam) for lam in lams]
    for grp in forms:
        for st in settings:
            p = compile_switching_filter(model, groups=grp, **st, **kw)
            key = (tuple(p.groups), p.step.fwd_rep, tuple(sorted(p.step.counts.items())), p.step.flops)
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def compile_switching_filter(
    model: SwitchingModel,
    lam: float = 1.0,
    groups: list[tuple[str, ...]] | None = None,
    rules: str = "minimal",
    extractor: str = "greedy",
    strategy: str = "bfs",
    max_iters: int = 30,
    node_limit: int = 50_000,
    time_limit_s: float = 60,
    reps: tuple[str, ...] = (MOMENT, INFO),
    price: float | None = None,
    bound=None,
    cache: dict | None = None,
) -> SFilterProgram:
    """The cheapest program (flops + λ × merged components per step) over the message forms
    (`groups`, default: every form from `message_groups`) and message representations.
    With `price`, the penalty is μ × the expected KL bound of each collapse (phase I). `cache`
    keeps the saturations between calls on the same model."""
    from .search import run_strategy

    if price is not None and bound is None:
        from .switching_bound import ExpectedBound

        bound = ExpectedBound(model)
    start = time.perf_counter()
    best = None
    for grp in [groups] if groups is not None else message_groups(model):
        sats = {}
        for kind in ("head", "step"):
            key = (tuple(grp), kind, strategy, rules, max_iters, node_limit)
            if cache is not None and key in cache:
                sats[kind] = cache[key]
                continue
            loc = local_step(model, kind, grp)
            sat, _ = run_strategy(loc.fg, loc.queries, strategy, max_iters=max_iters, node_limit=node_limit,
                                  rules=rules, inputs=loc.inputs, seeds=loc.seeds)
            sats[kind] = (loc, sat)
            if cache is not None:
                cache[key] = sats[kind]
        for r in reps:
            try:
                t = {k: _template(sats[k][1], sats[k][0], model, r, lam, extractor, time_limit_s, price, bound)
                     for k in ("head", "step")}
            except ValueError:  # this representation cannot reach every root
                continue
            key = (t["step"].choice.cost, t["head"].choice.cost)
            if best is None or key < best[0]:
                best = (key, t, grp)
    _, t, grp = best
    return SFilterProgram(model, t["head"], t["step"], list(grp), lam, time.perf_counter() - start, price)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _apply(im: Impl, kids: list, facs, y_of, cards, dims, inputs, step: int = 0):
    op = im.op
    if op == "leaf":
        return ("factor", im.node.arg)
    if op == "table":
        return table_value(facs[im.node.arg], cards, im.out)
    if op == "prior":
        return clg_prior(facs[im.node.arg], cards)
    if op == "input":
        return inputs[im.node.arg]
    if op == "cond>info":
        fid = kids[0][1]
        return clg_info(facs[fid], cards, dims, y_of.get(fid))
    if op == "moment>info":
        return cg_convert(kids[0], INFO)
    if op == "info>moment":
        return cg_convert(kids[0], MOMENT)
    if op in ("moment_sum", "info_sum"):
        (k,) = kids
        f = moment_sum if op == "moment_sum" else info_sum
        return CG(k.dvars, {c: f(p, im.node.arg, dims) for c, p in k.comps.items()}, k.tau, kls=k.kls)
    if op == "dsum":
        return cg_dsum(kids[0], im.node.arg, cards)
    if op == "collapse":
        out = cg_collapse(kids[0], im.node.arg, cards)
        out.kls = {**kids[0].kls, (step, im.state): out.kl}
        return out
    if op in ("info_mul", "moment_mul"):
        a, b = kids
        f = (lambda p, q: info_mul(p, q, dims)) if op == "info_mul" else (lambda p, q: moment_mul_disjoint(p, q, dims))
        out = cg_pointwise(a, b, cards, f)
        out.kls = {**a.kls, **b.kls}
        if a.tau and b.tau:
            # independent parts (such as a message split into its marginals): the bounds add
            disjoint = not (a.vars & b.vars) and log_mass(a) is not None and log_mass(b) is not None
            out.tau = min(1.0, a.tau + b.tau) if disjoint else 1.0
        elif a.tau or b.tau:
            ap, ex = (a, b) if a.tau else (b, a)
            out.tau = product_tau(ap, ex, out)
        return out
    m = next(k for k in kids if isinstance(k, CG))
    fid = next(k for k in kids if isinstance(k, tuple))[1]
    f = facs[fid]
    if op == "predict":
        out = cg_predict(m, f, cards, dims)
        out.tau, out.kls = m.tau, m.kls  # a conditional over a new child: TV is kept
        return out
    if op == "update":
        out = cg_update(m, f, y_of[fid], cards, dims)
        out.kls = m.kls
        if m.tau:
            sup = max(-0.5 * (len(gf.b) * LOG2PI + np.linalg.slogdet(gf.Q)[1]) for gf in f.params.values())
            out.tau = product_tau(m, None, out, sup)
        return out
    raise ValueError(f"unknown operation {op!r}")


def evaluate_template(tmpl: STemplate, model: SwitchingModel, ys, inputs, kls: list | None = None, step: int = 0) -> dict[str, CG]:
    facs = tmpl.facs
    first = len(facs) - len(model.observation)
    y_of = {first + k: y for k, y in enumerate(ys)}
    cards, dims = model.local_cards(), model.local_dims()
    vals: dict = {}
    ch = tmpl.choice.choice
    # the states the messages depend on: only their collapses are recorded in `kls`
    on_msg, stack = set(), [s for q, s in tmpl.choice.roots.items() if q.startswith("msg")]
    while stack:
        cur = stack.pop()
        if cur not in on_msg:
            on_msg.add(cur)
            stack.extend(ch[cur].kids)
    for s in tmpl.choice.roots.values():
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
            vals[cur] = _apply(ch[cur], [vals[k] for k in ch[cur].kids], facs, y_of, cards, dims, inputs, step)
            if kls is not None and ch[cur].op == "collapse" and cur in on_msg:
                kls.append(vals[cur].kl)
            stack.pop()
    return {q: vals[s] for q, s in tmpl.choice.roots.items()}


@dataclass
class SResult:
    probs: dict[str, np.ndarray]  # discrete state -> probabilities
    mean: dict[str, np.ndarray]  # continuous state -> mean (over all modes)
    cov: dict[str, np.ndarray]
    tau: dict[str, float]  # per output, the TV bound
    messages: list[CG]  # the messages passed on (names @0)
    collapse_kl: list[float]  # this step's collapses on the way to a message: each one's bound on the KL it adds


class SwitchingFilter:
    def __init__(self, program: SFilterProgram):
        self.p = program
        self.t = 0
        self.inputs: dict[str, CG] = {}

    def step(self, ys) -> SResult:
        tmpl = self.p.head if self.t == 0 else self.p.step
        kls: list[float] = []
        out = evaluate_template(tmpl, self.p.model, ys, self.inputs, kls, self.t)
        # normalized (a message is defined up to a constant; unnormalized ones pick up the other
        # parts' constants at every step and their log weights grow without bound)
        msgs = [normalize(out[f"msg{k}"]) for k in range(len(tmpl.local.groups))]
        self.inputs = {f"fwd{k}": shift_back(cg_convert(v, self.p.step.fwd_rep) if v.form != self.p.step.fwd_rep else v)
                       for k, v in enumerate(msgs)}
        self.t += 1
        probs, mean, cov, tau = {}, {}, {}, {}
        for n in self.p.model.cards:
            v = out[n]
            w = np.array([v.comps[(i,)].logz for i in range(self.p.model.cards[n])])
            probs[n] = np.exp(w - _logsumexp(w))
            tau[n] = v.tau
        for n in self.p.model.dims:
            (p,) = out[n].comps.values()
            mean[n], cov[n], tau[n] = p.mu, p.S, out[n].tau
        return SResult(probs, mean, cov, tau, msgs, kls)


def run_filter(program: SFilterProgram, obs) -> list[SResult]:
    f = SwitchingFilter(program)
    return [f.step(ys) for ys in obs]

"""Reference filters for switching models (written directly with numpy) and model generators.

All references work on the stacked form: one joint mode (every discrete state at
once) and one stacked continuous state. GPB1, GPB2 and IMM follow the textbook
definitions; `exact_filter` enumerates every mode history (for short runs).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .dynamic import split
from .gaussian import LOG2PI, GFactor
from .switching import CLG, DTable, SwitchingModel, collapse_one
from .gaussian import Moment


@dataclass
class Stacked:
    modes: list[tuple[int, ...]]  # joint modes: assignments of `dnames`
    dnames: list[str]
    cnames: list[str]
    off: dict[str, slice]
    p0: np.ndarray
    T: np.ndarray  # T[i, j] = P(joint mode j at t | i at t-1)
    mu0: list[np.ndarray]
    P0: list[np.ndarray]
    A: list[np.ndarray]
    b: list[np.ndarray]
    Q: list[np.ndarray]
    H: list[np.ndarray]
    c: list[np.ndarray]
    R: list[np.ndarray]


def stack(model: SwitchingModel) -> Stacked:
    dn = sorted(model.cards)
    cn = sorted(model.dims)
    off, k = {}, 0
    for x in cn:
        off[x] = slice(k, k + model.dims[x])
        k += model.dims[x]
    N = k
    modes = list(itertools.product(*[range(model.cards[d]) for d in dn]))
    val = lambda J, t: {f"{d}@{t}": J[i] for i, d in enumerate(dn)}  # noqa: E731

    def table_at(f: DTable, a: dict) -> float:
        return float(f.table[tuple(a[v] for v in f.scope)])

    def params(f: CLG, a: dict) -> GFactor:
        return f.params[tuple(a[v] for v in f.dparents)]

    p0 = np.array([np.prod([table_at(f, val(J, 0)) for f in model.initial if isinstance(f, DTable)]) for J in modes])
    T = np.array([[np.prod([table_at(f, {**val(I, -1), **val(J, 0)}) for f in model.transition if isinstance(f, DTable)])
                   for J in modes] for I in modes])
    # a discrete state with no transition table of its own at @0 keeps no memory: none in our models
    mu0, P0, A, b, Q, H, c, R = [], [], [], [], [], [], [], []
    for J in modes:
        a0 = val(J, 0)
        m, P = np.zeros(N), np.zeros((N, N))
        for f in model.initial:
            if isinstance(f, CLG):
                g = params(f, a0)
                s = off[split(f.child)[0]]
                m[s], P[s, s] = g.b, g.Q
        mu0.append(m)
        P0.append(P)
        a = {**{f"{d}@-1": 0 for d in dn}, **a0}
        AJ, bJ, QJ = np.zeros((N, N)), np.zeros(N), np.zeros((N, N))
        for f in model.transition:
            if isinstance(f, CLG):
                g = params(f, a)
                s = off[split(f.child)[0]]
                for Ai, p in zip(g.A, g.parents):
                    AJ[s, off[split(p)[0]]] += Ai
                bJ[s], QJ[s, s] = g.b, g.Q
        A.append(AJ)
        b.append(bJ)
        Q.append(QJ)
        rows = []
        for f in model.observation:
            g = params(f, a0)
            Hf = np.zeros((len(g.b), N))
            for Ai, p in zip(g.A, g.parents):
                Hf[:, off[split(p)[0]]] += Ai
            rows.append((Hf, g.b, g.Q))
        H.append(np.vstack([r[0] for r in rows]))
        c.append(np.concatenate([r[1] for r in rows]))
        Rs = [r[2] for r in rows]
        RJ = np.zeros((sum(len(r) for r in Rs),) * 2)
        o = 0
        for r in Rs:
            RJ[o:o + len(r), o:o + len(r)] = r
            o += len(r)
        R.append(RJ)
    return Stacked(modes, dn, cn, off, p0, T, mu0, P0, A, b, Q, H, c, R)


def _update(mu, S, H, c, R, y):
    Sy = H @ S @ H.T + R
    r = y - (H @ mu + c)
    K = np.linalg.solve(Sy, H @ S).T
    ll = -0.5 * (r @ np.linalg.solve(Sy, r) + len(r) * LOG2PI + np.linalg.slogdet(Sy)[1])
    return mu + K @ r, S - K @ H @ S, float(ll)


def _predict(mu, S, A, b, Q):
    return A @ mu + b, A @ S @ A.T + Q


@dataclass
class RefResult:
    mode: np.ndarray  # probabilities of the joint modes
    mean: np.ndarray  # stacked
    cov: np.ndarray
    comps: list[tuple[float, np.ndarray, np.ndarray]]  # the message: per joint mode (log weight, mean, cov)


def _outputs(st: Stacked, comps) -> RefResult:
    """comps: per joint mode a list of (log weight, mean, cov) (a mixture for each mode)."""
    lw = np.array([np.logaddexp.reduce([w for w, _, _ in cs]) for cs in comps])
    mode = np.exp(lw - np.logaddexp.reduce(lw))
    allc = [Moment(("x",), m, S, w) for cs in comps for w, m, S in cs]
    q, _ = collapse_one(allc)
    return RefResult(mode, q.mu, q.S, [])


def _head(st: Stacked, y):
    out = []
    for J in range(len(st.modes)):
        mu, S, ll = _update(st.mu0[J], st.P0[J], st.H[J], st.c[J], st.R[J], y)
        out.append((float(np.log(st.p0[J])) + ll, mu, S))
    return out


def _collapse(parts):
    q, _ = collapse_one([Moment(("x",), m, S, w) for w, m, S in parts])
    return q.logz, q.mu, q.S


def gpb2(st: Stacked, obs) -> list[RefResult]:
    res = []
    msg = None
    for t, ys in enumerate(obs):
        y = np.concatenate(ys)
        if t == 0:
            msg = _head(st, y)
        else:
            new = []
            for j in range(len(st.modes)):
                parts = []
                for i, (w, mu, S) in enumerate(msg):
                    m2, S2 = _predict(mu, S, st.A[j], st.b[j], st.Q[j])
                    m3, S3, ll = _update(m2, S2, st.H[j], st.c[j], st.R[j], y)
                    parts.append((w + np.log(st.T[i, j]) + ll, m3, S3))
                new.append(_collapse(parts))
            msg = new
        r = _outputs(st, [[c] for c in msg])
        r.comps = list(msg)
        res.append(r)
    return res


def imm(st: Stacked, obs) -> list[RefResult]:
    res = []
    msg = None
    for t, ys in enumerate(obs):
        y = np.concatenate(ys)
        if t == 0:
            msg = _head(st, y)
        else:
            new = []
            for j in range(len(st.modes)):
                w, mu, S = _collapse([(wi + np.log(st.T[i, j]), m, P) for i, (wi, m, P) in enumerate(msg)])
                m2, S2 = _predict(mu, S, st.A[j], st.b[j], st.Q[j])
                m3, S3, ll = _update(m2, S2, st.H[j], st.c[j], st.R[j], y)
                new.append((w + ll, m3, S3))
            msg = new
        r = _outputs(st, [[c] for c in msg])
        r.comps = list(msg)
        res.append(r)
    return res


def gpb1(st: Stacked, obs) -> list[RefResult]:
    res = []
    lp, g = None, None  # log mode probabilities (unnormalized, with the evidence) and one Gaussian
    for t, ys in enumerate(obs):
        y = np.concatenate(ys)
        if t == 0:
            per = _head(st, y)
        else:
            per = []
            for j in range(len(st.modes)):
                m2, S2 = _predict(g[0], g[1], st.A[j], st.b[j], st.Q[j])
                m3, S3, ll = _update(m2, S2, st.H[j], st.c[j], st.R[j], y)
                prior = np.logaddexp.reduce([lp[i] + np.log(st.T[i, j]) for i in range(len(st.modes))])
                per.append((prior + ll, m3, S3))
        lp = np.array([w for w, _, _ in per])
        w, mu, S = _collapse(per)
        g = (mu, S)
        r = _outputs(st, [[c] for c in per])
        r.comps = list(per)
        res.append(r)
    return res


def exact_filter(st: Stacked, obs) -> list[RefResult]:
    """Every mode history (len(modes) ** T components): the exact posterior."""
    res = []
    hyps = []  # (log weight, current joint mode, mean, cov)
    for t, ys in enumerate(obs):
        y = np.concatenate(ys)
        if t == 0:
            hyps = [(w, J, mu, S) for J, (w, mu, S) in enumerate(_head(st, y))]
        else:
            new = []
            for w, i, mu, S in hyps:
                for j in range(len(st.modes)):
                    m2, S2 = _predict(mu, S, st.A[j], st.b[j], st.Q[j])
                    m3, S3, ll = _update(m2, S2, st.H[j], st.c[j], st.R[j], y)
                    new.append((w + np.log(st.T[i, j]) + ll, j, m3, S3))
            hyps = new
        comps = [[(w, mu, S) for w, J, mu, S in hyps if J == j] for j in range(len(st.modes))]
        r = _outputs(st, comps)
        r.comps = comps
        res.append(r)
    return res


def marginals(st: Stacked, r: RefResult, model: SwitchingModel):
    """Per state name: probabilities (discrete) or (mean, cov) (continuous)."""
    probs = {}
    for k, d in enumerate(st.dnames):
        p = np.zeros(model.cards[d])
        for J, pj in zip(st.modes, r.mode):
            p[J[k]] += pj
        probs[d] = p
    cont = {x: (r.mean[st.off[x]], r.cov[st.off[x], st.off[x]]) for x in st.cnames}
    return probs, cont


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def _g(kind, child, parents, A, b, Q):
    return GFactor(kind, child, parents, tuple(np.atleast_2d(a) for a in A), np.atleast_1d(np.asarray(b, float)),
                   np.atleast_2d(np.asarray(Q, float)))


def _sticky(M: int, stay: float) -> np.ndarray:
    T = np.full((M, M), (1 - stay) / (M - 1))
    np.fill_diagonal(T, stay)
    return T


def _component(tag: str, M: int, d: int, rng, obs_mode: bool):
    """One target: mode m{tag} (Markov) and state x{tag} (scalar if d = 1, else position-velocity in d/2 axes)."""
    m, x = f"m{tag}", f"x{tag}"
    if d == 1:
        As = [np.array([[0.99]]), np.array([[0.5]]), np.array([[-0.5]])][:M]
        Qs = [np.array([[0.01]]), np.array([[1.0]]), np.array([[0.3]])][:M]
        H = np.array([[1.0]])
    else:
        k = d // 2
        F = np.block([[np.eye(k), np.eye(k)], [np.zeros((k, k)), np.eye(k)]])
        damp = np.block([[np.eye(k), np.eye(k)], [np.zeros((k, k)), 0.2 * np.eye(k)]])
        As = [F, F, damp][:M]
        Qs = [0.01 * np.eye(d), np.diag([0.1] * k + [1.0] * k), 0.05 * np.eye(d)][:M]
        H = np.hstack([np.eye(k), np.zeros((k, k))])
    R = 0.5 * np.eye(H.shape[0])
    mu0 = rng.normal(size=d)
    init = [DTable((f"{m}@0",), np.full(M, 1.0 / M), f"{m}@0"),
            CLG("prior", f"{x}@0", (), (), {(): _g("prior", f"{x}@0", (), (), mu0, np.eye(d))})]
    trans = [DTable((f"{m}@-1", f"{m}@0"), _sticky(M, 0.9), f"{m}@0"),
             CLG("cond", f"{x}@0", (f"{x}@-1",), (f"{m}@0",),
                 {(j,): _g("cond", f"{x}@0", (f"{x}@-1",), (As[j],), np.zeros(d), Qs[j]) for j in range(M)})]
    if obs_mode:  # the observation noise also depends on the mode
        obs = CLG("obs", None, (f"{x}@0",), (f"{m}@0",),
                  {(j,): _g("obs", None, (f"{x}@0",), (H,), np.zeros(H.shape[0]), R * (1 + 4 * j)) for j in range(M)})
    else:
        obs = CLG("obs", None, (f"{x}@0",), (), {(): _g("obs", None, (f"{x}@0",), (H,), np.zeros(H.shape[0]), R)})
    return {m: M}, {x: d}, init, trans, [obs]


def maneuver(M: int = 2, d: int = 1, seed: int = 0, obs_mode: bool = False) -> SwitchingModel:
    rng = np.random.default_rng(seed + 120_000)
    cards, dims, init, trans, obs = _component("", M, d, rng, obs_mode)
    return SwitchingModel(cards, dims, init, trans, obs)


def targets(k: int, M: int = 2, d: int = 1, seed: int = 0) -> SwitchingModel:
    rng = np.random.default_rng(seed + 130_000)
    cards, dims, init, trans, obs = {}, {}, [], [], []
    for i in range(k):
        c, dm, a, b, o = _component(str(i), M, d, rng, False)
        cards |= c
        dims |= dm
        init += a
        trans += b
        obs += o
    return SwitchingModel(cards, dims, init, trans, obs)


def outlier(d: int = 1, p_out: float = 0.1, seed: int = 0) -> SwitchingModel:
    """A linear-Gaussian state whose each observation is an outlier (100x the noise) with probability p_out."""
    rng = np.random.default_rng(seed + 140_000)
    A = 0.95 * np.eye(d)
    H = np.eye(d)
    R = 0.5 * np.eye(d)
    tab = np.array([1 - p_out, p_out])
    init = [DTable(("o@0",), tab, "o@0"),
            CLG("prior", "x@0", (), (), {(): _g("prior", "x@0", (), (), rng.normal(size=d), np.eye(d))})]
    trans = [DTable(("o@0",), tab, "o@0"),
             CLG("cond", "x@0", ("x@-1",), (), {(): _g("cond", "x@0", ("x@-1",), (A,), np.zeros(d), 0.1 * np.eye(d))})]
    obs = [CLG("obs", None, ("x@0",), ("o@0",),
               {(j,): _g("obs", None, ("x@0",), (H,), np.zeros(d), R * (1 if j == 0 else 100)) for j in range(2)})]
    return SwitchingModel({"o": 2}, {"x": d}, init, trans, obs)


def simulate(model: SwitchingModel, T: int, seed: int = 0) -> list[list[np.ndarray]]:
    """Observations drawn from the model (via the stacked form)."""
    rng = np.random.default_rng(seed + 150_000)
    st = stack(model)
    J = rng.choice(len(st.modes), p=st.p0)
    x = rng.multivariate_normal(st.mu0[J], st.P0[J])
    out = []
    for t in range(T):
        if t > 0:
            J = rng.choice(len(st.modes), p=st.T[J] / st.T[J].sum())
            x = st.A[J] @ x + st.b[J] + rng.multivariate_normal(np.zeros(len(x)), st.Q[J])
        y = st.H[J] @ x + st.c[J] + rng.multivariate_normal(np.zeros(len(st.c[J])), st.R[J])
        ys, o = [], 0
        for f in model.observation:
            m = len(next(iter(f.params.values())).b)
            ys.append(y[o:o + m])
            o += m
        out.append(ys)
    return out


def stacked_model(model: SwitchingModel) -> SwitchingModel:
    """The same model with one joint mode `m` and one stacked state `x` (what a state-wide IMM works on)."""
    st = stack(model)
    K, N = len(st.modes), len(st.mu0[0])
    M = len(st.H[0])
    init = [DTable(("m@0",), st.p0, "m@0"),
            CLG("prior", "x@0", (), ("m@0",), {(j,): _g("prior", "x@0", (), (), st.mu0[j], st.P0[j]) for j in range(K)})]
    trans = [DTable(("m@-1", "m@0"), st.T, "m@0"),
             CLG("cond", "x@0", ("x@-1",), ("m@0",),
                 {(j,): _g("cond", "x@0", ("x@-1",), (st.A[j],), st.b[j], st.Q[j]) for j in range(K)})]
    obs = [CLG("obs", None, ("x@0",), ("m@0",), {(j,): _g("obs", None, ("x@0",), (st.H[j],), st.c[j], st.R[j]) for j in range(K)})]
    assert all(len(h) == M for h in st.H)
    return SwitchingModel({"m": K}, {"x": N}, init, trans, obs)


def stacked_obs(obs):
    return [[np.concatenate(ys)] for ys in obs]


def message_joint(model: SwitchingModel, messages) -> tuple[list[tuple[int, ...]], dict]:
    """The approximate joint over (joint mode, stacked state) of a single-component model's messages:
    per joint mode a list of (log weight, mean, cov)."""
    st = stack(model)
    comps = {}
    if len(messages) == 1:
        (v,) = messages
        for c, p in v.comps.items():
            comps[c] = [(p.logz, p.mu, p.S)]
    else:
        d = next(m for m in messages if not m.cvars)
        x = next(m for m in messages if m.cvars)
        (q,) = x.comps.values()
        for c, p in d.comps.items():
            comps[c] = [(p.logz + q.logz, q.mu, q.S)]
    return st.modes, comps


def tv_1d(exact: list[list[tuple[float, np.ndarray, np.ndarray]]], approx: dict, modes, grid: int = 20001) -> float:
    """TV between two mixtures over (mode, scalar state), both normalized, by numerical integration."""
    def dens(parts, xs):
        out = np.zeros_like(xs)
        for w, m, S in parts:
            s2 = float(S[0, 0])
            out += np.exp(w - 0.5 * np.log(2 * np.pi * s2) - 0.5 * (xs - float(m[0])) ** 2 / s2)
        return out

    if all(k == () for k in approx):  # the message has no discrete part: compare the marginals over x
        exact, modes = [[c for parts in exact for c in parts]], [()]
    allw = [w for parts in exact for w, _, _ in parts]
    Ze = np.logaddexp.reduce(allw)
    Za = np.logaddexp.reduce([w for parts in approx.values() for w, _, _ in parts])
    mus = [float(m[0]) for parts in exact for _, m, _ in parts] + [float(m[0]) for parts in approx.values() for _, m, _ in parts]
    sds = [float(np.sqrt(S[0, 0])) for parts in exact for _, _, S in parts] + \
          [float(np.sqrt(S[0, 0])) for parts in approx.values() for _, _, S in parts]
    xs = np.linspace(min(mus) - 12 * max(sds), max(mus) + 12 * max(sds), grid)
    dx = xs[1] - xs[0]
    tv = 0.0
    for j, J in enumerate(modes):
        e = dens([(w - Ze, m, S) for w, m, S in exact[j]], xs)
        a = dens([(w - Za, m, S) for w, m, S in approx.get(J, [])], xs)
        tv += 0.5 * float(np.sum(np.abs(e - a)) * dx)
    return tv

"""Linear-Gaussian factors, their two value representations, and the operations between them.

A value is an unnormalized Gaussian potential over some vector variables (kept in
sorted name order). It is held either in information form, exp(-½ xᵀJx + hᵀx + g),
or in moment form, exp(logz) · N(x; μ, Σ) (only for normalizable potentials).
The original factors are also kept in their conditional form (A, b, Q) so that
moment-form prediction and Kalman updates can use them directly.

Costs follow the spec (dense matrices, unit constants): see the `cost_*` functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG2PI = float(np.log(2 * np.pi))


@dataclass(frozen=True)
class GFactor:
    """A linear-Gaussian factor.

    prior: N(child; b, Q).  cond: N(child; Σ_i A_i parent_i + b, Q).
    obs: the likelihood N(y; Σ_i A_i parent_i + b, Q) of an observed value y (given
    at evaluation time) as a function of the parents; `child` is None.
    """

    kind: str
    child: str | None
    parents: tuple[str, ...]
    A: tuple[np.ndarray, ...]
    b: np.ndarray
    Q: np.ndarray

    @property
    def scope(self) -> tuple[str, ...]:
        if self.kind == "prior":
            return (self.child,)
        if self.kind == "cond":
            return self.parents + (self.child,)
        return self.parents


@dataclass
class Info:
    vars: tuple[str, ...]
    J: np.ndarray
    h: np.ndarray
    g: float


@dataclass
class Moment:
    vars: tuple[str, ...]
    mu: np.ndarray
    S: np.ndarray
    logz: float = 0.0


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------


def dim(vars_, dims: dict[str, int]) -> int:
    return sum(dims[v] for v in vars_)


def _slices(vars_: tuple[str, ...], dims: dict[str, int]) -> dict[str, slice]:
    out, k = {}, 0
    for v in vars_:
        out[v] = slice(k, k + dims[v])
        k += dims[v]
    return out


def _index(vars_: tuple[str, ...], dims: dict[str, int], sub) -> np.ndarray:
    sl = _slices(vars_, dims)
    return np.concatenate([np.arange(sl[v].start, sl[v].stop) for v in sub]) if sub else np.zeros(0, int)


def _embed(J: np.ndarray, h: np.ndarray, src: tuple[str, ...], dst: tuple[str, ...], dims) -> tuple[np.ndarray, np.ndarray]:
    idx = _index(dst, dims, src)
    D = dim(dst, dims)
    J2, h2 = np.zeros((D, D)), np.zeros(D)
    J2[np.ix_(idx, idx)] = J
    h2[idx] = h
    return J2, h2


def _logdet(M: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0:
        raise np.linalg.LinAlgError("matrix is not positive definite")
    return float(ld)


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


def info_of_factor(f: GFactor, dims: dict[str, int], y: np.ndarray | None = None) -> Info:
    """Information form of a factor (for obs, of its likelihood given y)."""
    order = tuple(sorted(f.scope))
    Qi = np.linalg.inv(f.Q)
    if f.kind == "obs":
        # residual r = Σ A_i x_i - (y - b)
        M = np.hstack([f.A[f.parents.index(v)] for v in order])
        t = np.asarray(y, float) - f.b
    else:
        # residual r = child - Σ A_i p_i - b
        blocks = []
        for v in order:
            if v == f.child:
                blocks.append(np.eye(dims[v]))
            else:
                blocks.append(-f.A[f.parents.index(v)])
        M = np.hstack(blocks)
        t = f.b
    J = M.T @ Qi @ M
    h = M.T @ Qi @ t
    g = -0.5 * float(t @ Qi @ t) - 0.5 * (len(t) * LOG2PI + _logdet(f.Q))
    return Info(order, J, h, g)


def moment_of_prior(f: GFactor) -> Moment:
    return Moment((f.child,), np.asarray(f.b, float), np.asarray(f.Q, float), 0.0)


def info_of_moment(m: Moment) -> Info:
    Si = np.linalg.inv(m.S)
    h = Si @ m.mu
    g = m.logz - 0.5 * float(m.mu @ h) - 0.5 * (len(m.mu) * LOG2PI + _logdet(m.S))
    return Info(m.vars, Si, h, g)


def moment_of_info(i: Info) -> Moment:
    S = np.linalg.inv(i.J)
    mu = S @ i.h
    logz = i.g + 0.5 * float(i.h @ mu) + 0.5 * (len(mu) * LOG2PI + _logdet(S))
    return Moment(i.vars, mu, S, logz)


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------


def info_mul(a: Info, b: Info, dims) -> Info:
    out = tuple(sorted(set(a.vars) | set(b.vars)))
    Ja, ha = _embed(a.J, a.h, a.vars, out, dims)
    Jb, hb = _embed(b.J, b.h, b.vars, out, dims)
    return Info(out, Ja + Jb, ha + hb, a.g + b.g)


def info_sum(a: Info, x: str, dims) -> Info:
    """∫ dx of an information-form potential (Schur complement)."""
    rest = tuple(v for v in a.vars if v != x)
    ix, ir = _index(a.vars, dims, (x,)), _index(a.vars, dims, rest)
    Jxx, Jrx = a.J[np.ix_(ix, ix)], a.J[np.ix_(ir, ix)]
    Jxx_inv = np.linalg.inv(Jxx)
    hx = a.h[ix]
    J = a.J[np.ix_(ir, ir)] - Jrx @ Jxx_inv @ Jrx.T
    h = a.h[ir] - Jrx @ Jxx_inv @ hx
    g = a.g + 0.5 * float(hx @ Jxx_inv @ hx) + 0.5 * (len(ix) * LOG2PI - _logdet(Jxx))
    return Info(rest, J, h, g)


def moment_sum(m: Moment, x: str, dims) -> Moment:
    rest = tuple(v for v in m.vars if v != x)
    ir = _index(m.vars, dims, rest)
    return Moment(rest, m.mu[ir], m.S[np.ix_(ir, ir)], m.logz)


def moment_mul_disjoint(a: Moment, b: Moment, dims) -> Moment:
    """Product of moment-form values over disjoint variables: the block-diagonal joint."""
    if set(a.vars) & set(b.vars):
        raise ValueError("moment_mul_disjoint needs disjoint scopes")
    src = a.vars + b.vars
    out = tuple(sorted(src))
    na, nb = len(a.mu), len(b.mu)
    S = np.zeros((na + nb, na + nb))
    S[:na, :na], S[na:, na:] = a.S, b.S
    mu = np.concatenate([a.mu, b.mu])
    perm = _index(src, dims, out)
    return Moment(out, mu[perm], S[np.ix_(perm, perm)], a.logz + b.logz)


def moment_predict(m: Moment, f: GFactor, dims) -> Moment:
    """m(S) · N(child; Σ A p + b, Q) with parents ⊆ S, child ∉ S: the joint over S ∪ {child}."""
    ip = [_index(m.vars, dims, (p,)) for p in f.parents]
    mu_c = sum(A @ m.mu[i] for A, i in zip(f.A, ip)) + f.b
    # cross-covariance Cov(child, S) = Σ_i A_i Σ_{p_i, S}
    C = sum(A @ m.S[i, :] for A, i in zip(f.A, ip))
    S_cc = sum(C[:, i] @ A.T for A, i in zip(f.A, ip)) + f.Q
    out = tuple(sorted(m.vars + (f.child,)))
    src = m.vars + (f.child,)
    D = dim(src, dims)
    mu = np.concatenate([m.mu, mu_c])
    S = np.zeros((D, D))
    n = len(m.mu)
    S[:n, :n], S[n:, :n], S[:n, n:], S[n:, n:] = m.S, C, C.T, S_cc
    perm = _index(src, dims, out)
    return Moment(out, mu[perm], S[np.ix_(perm, perm)], m.logz)


def moment_update(m: Moment, f: GFactor, y: np.ndarray, dims) -> Moment:
    """m(S) · N(y; Σ A x + b, R) with the x ⊆ S: the Kalman update (the scope stays S)."""
    ix = [_index(m.vars, dims, (x,)) for x in f.parents]
    pred = sum(A @ m.mu[i] for A, i in zip(f.A, ix)) + f.b
    HS = sum(A @ m.S[i, :] for A, i in zip(f.A, ix))  # H Σ_{x,S}
    Sy = sum(HS[:, i] @ A.T for A, i in zip(f.A, ix)) + f.Q  # H Σ_xx Hᵀ + R
    K = np.linalg.solve(Sy, HS).T  # Σ_{S,x} Hᵀ Sy⁻¹
    r = np.asarray(y, float) - pred
    mu = m.mu + K @ r
    S = m.S - K @ HS
    loglik = -0.5 * (float(r @ np.linalg.solve(Sy, r)) + len(r) * LOG2PI + _logdet(Sy))
    return Moment(m.vars, mu, S, m.logz + loglik)


# ---------------------------------------------------------------------------
# costs (dense, unit constants)
# ---------------------------------------------------------------------------


def cost_info_mul(D: int) -> int:
    return D * D


def cost_info_sum(D: int, dx: int) -> int:
    return (D - dx) ** 2 * dx + dx**3


def cost_predict(d_parents: int, d_child: int, d_moment: int) -> int:
    return d_child * d_parents * d_moment + d_child**2 * d_parents


def cost_update(D: int, m: int) -> int:
    return D * D * m + D * m * m + m**3


def cost_convert(D: int) -> int:
    return D**3


def cost_factor_to_info(f: GFactor, dims, amortize: bool = False) -> int:
    """Per-step cost of putting a factor in information form.

    With `amortize`, the parts that depend only on the model parameters (all of a
    prior or transition, and HᵀR⁻¹H, HᵀR⁻¹ of an observation) are computed once
    for the whole run; only the observation's h = HᵀR⁻¹(y − b) is counted per step.
    """
    if amortize:
        return len(f.b) * dim(f.parents, dims) if f.kind == "obs" else 0
    if f.kind == "obs":
        m, d = len(f.b), dim(f.parents, dims)
        return m**3 + d * m * m + d * d * m
    dc, dp = dims[f.child], dim(f.parents, dims)
    return dc**3 + dp * dc * dc + dp * dp * dc


# ---------------------------------------------------------------------------
# reference: dense log-potential, for tests
# ---------------------------------------------------------------------------


def log_potential(v: Info | Moment, x: np.ndarray) -> float:
    if isinstance(v, Info):
        return float(-0.5 * x @ v.J @ x + v.h @ x + v.g)
    d = x - v.mu
    return v.logz - 0.5 * (float(d @ np.linalg.solve(v.S, d)) + len(d) * LOG2PI + _logdet(v.S))


def log_factor(f: GFactor, dims, point: dict[str, np.ndarray], y=None) -> float:
    if f.kind == "prior":
        return log_potential(moment_of_prior(f), point[f.child])
    mean = sum(A @ point[p] for A, p in zip(f.A, f.parents)) + f.b
    target = np.asarray(y, float) if f.kind == "obs" else point[f.child]
    return log_potential(Moment(("_",), mean, f.Q, 0.0), target)

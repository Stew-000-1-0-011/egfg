"""Expected collapse error from the model alone (phase I).

Assuming the observations come from the model, the expected KL bound of one
collapse (with exact inputs) is at most

    Σ_a P(a) · ½ log det C_lin(a)  −  Σ_b P(b) · ½ E[log det K_σ | b]

where a runs over the discrete values left after the collapse, b over those
before it, C_lin(a) is the error covariance of the linear (LMMSE) estimate of the
collapsed continuous state from the observations of a window, given a, and K_σ is
the Kalman covariance of that state given the whole mode sequence σ of the
window and its observations, with the window's first state known (a lower bound).
Everything comes from first and second moments: no simulation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .dynamic import split
from .switching import SwitchingModel
from .switching_ref import Stacked, stack


def mode_moments(st: Stacked, horizon: int = 50, tol: float = 1e-12):
    """π_i, q_i = E[x_t 1{θ_t=i}], X_i = E[x_t x_tᵀ 1{θ_t=i}] of the Markov jump linear system at time
    `horizon` (the steady state if it is reached earlier). Unstable dynamics (a position integrating
    a velocity) have no steady state; the window then starts at that time (and ends L steps later)."""
    M = len(st.modes)
    pi = st.p0.copy()
    q = [pi[i] * st.mu0[i] for i in range(M)]
    X = [pi[i] * (st.P0[i] + np.outer(st.mu0[i], st.mu0[i])) for i in range(M)]
    for _ in range(horizon):
        npi = st.T.T @ pi
        nq = [sum(st.T[i, j] * (st.A[j] @ q[i] + st.b[j] * pi[i]) for i in range(M)) for j in range(M)]
        nX = [sum(st.T[i, j] * (st.A[j] @ X[i] @ st.A[j].T + np.outer(st.A[j] @ q[i], st.b[j])
                                + np.outer(st.b[j], st.A[j] @ q[i]) + pi[i] * (np.outer(st.b[j], st.b[j]) + st.Q[j]))
                  for i in range(M)) for j in range(M)]
        delta = max(float(np.max(np.abs(nX[j] - X[j]))) for j in range(M)) + float(np.max(np.abs(npi - pi)))
        pi, q, X = npi, nq, nX
        if delta < tol * (1 + max(float(np.max(np.abs(x))) for x in X)):
            break
    return pi, q, X


@dataclass(frozen=True)
class Site:
    """A collapse: the discrete variables of the merged value (stacked component, time), the summed
    one, the continuous ones (stacked name, time), the observation factors at t it includes,
    and whether it is in the first step (t = 0)."""

    dvars: tuple[tuple[int, int], ...]
    summed: tuple[int, int]
    cvars: tuple[tuple[str, int], ...]
    obs: tuple[int, ...]
    head: bool


class ExpectedBound:
    def __init__(self, model: SwitchingModel, max_seqs: int = 4096, max_window: int = 6, horizon: int = 50):
        self.model = model
        self.st = stack(model)
        M = len(self.st.modes)
        self.L = 0
        self.pi, self.q, self.X = mode_moments(self.st, horizon)
        self.L = max(1, min(max_window, int(math.log(max_seqs) / math.log(max(M, 2))) - 1))
        rows, o = [], 0
        for f in model.observation:
            m = len(next(iter(f.params.values())).b)
            rows.append(slice(o, o + m))
            o += m
        self.rows = rows
        self.cache: dict[Site, float] = {}
        self._seq_cache: dict = {}

    # -- local names -> stacked ----------------------------------------------
    def site(self, dvars, summed, cvars, obs_factors, head: bool) -> Site:
        dn = self.st.dnames
        d = lambda v: (dn.index(split(v)[0]), split(v)[1])  # noqa: E731
        return Site(tuple(sorted(d(v) for v in dvars)), d(summed),
                    tuple(sorted((split(v)[0], split(v)[1]) for v in cvars)), tuple(sorted(obs_factors)), head)

    def __call__(self, site: Site) -> float:
        if site not in self.cache:
            self.cache[site] = self._bound(site)
        return self.cache[site]

    # -- the bound -------------------------------------------------------------
    def _bound(self, s: Site) -> float:
        P, comp, g, C, Ck, layout = self._sequences(s.head)
        st = self.st
        L = comp.shape[1] - 1
        xi = np.concatenate([layout["x", tau][st.off[n]] for n, tau in s.cvars])
        cur = [layout["y_now"][self.rows[i]] for i in s.obs]
        yi = np.concatenate([layout["y_past"]] + cur) if (len(layout["y_past"]) or cur) else np.zeros(0, int)
        idx = np.concatenate([xi, yi]).astype(int)
        nx = len(xi)
        gs, Cs, Ks = g[:, idx], C[:, idx][:, :, idx], Ck[:, idx][:, :, idx]
        assign = lambda dv: [tuple(r) for r in np.stack([comp[:, L + tau, k] for k, tau in dv], 1)] if dv else [()] * len(P)  # noqa: E731
        out_d = tuple(v for v in s.dvars if v != s.summed)
        keys_out = assign(out_d)
        first = 0.0
        for a in set(keys_out):
            sel = np.array([k == a for k in keys_out])
            w = P[sel]
            Pa = w.sum()
            mbar = (w[:, None] * gs[sel]).sum(0) / Pa
            d = gs[sel] - mbar
            Cbar = ((w[:, None, None] * (Cs[sel] + d[:, :, None] * d[:, None, :])).sum(0)) / Pa
            Cxx, Cxy, Cyy = Cbar[:nx, :nx], Cbar[:nx, nx:], Cbar[nx:, nx:]
            lin = Cxx - Cxy @ np.linalg.solve(Cyy, Cxy.T) if Cyy.size else Cxx
            first += Pa * 0.5 * np.linalg.slogdet(lin)[1]
        Kxx, Kxy, Kyy = Ks[:, :nx, :nx], Ks[:, :nx, nx:], Ks[:, nx:, nx:]
        K = Kxx - Kxy @ np.linalg.solve(Kyy, np.swapaxes(Kxy, 1, 2)) if Kyy.shape[1] else Kxx
        second = float((P * 0.5 * np.linalg.slogdet(K)[1]).sum())
        return max(0.0, float(first - second))

    def _sequences(self, head: bool):
        """For every mode sequence of the window: its probability, the joint modes' components at
        each time, and the mean, covariance, and known-start covariance of the vector
        [x_{t-1}, x_t, y of the earlier times, y_t] (with its layout)."""
        if head in self._seq_cache:
            return self._seq_cache[head]
        st = self.st
        M = len(st.modes)
        L = 0 if head else self.L
        N = len(st.mu0[0])
        my = len(st.c[0])
        seqs = list(itertools.product(range(M), repeat=L + 1))
        P, comp, G_, C_, K_ = [], [], [], [], []
        nsrc = N + N * L + my * (L + 1)
        for seq in seqs:
            p = st.p0[seq[0]] if head else self.pi[seq[0]] * np.prod([st.T[seq[k], seq[k + 1]] for k in range(L)])
            if p <= 0:
                continue
            j0 = seq[0]
            if head:
                m0, S0 = st.mu0[j0], st.P0[j0]
            else:
                m0 = self.q[j0] / self.pi[j0]
                S0 = self.X[j0] / self.pi[j0] - np.outer(m0, m0)
            F = np.zeros((N, nsrc))
            F[:, :N] = np.eye(N)
            f = m0.copy()
            xs = [(F.copy(), f.copy())]
            for k in range(1, L + 1):
                j = seq[k]
                F = st.A[j] @ F
                F[:, N + (k - 1) * N:N + k * N] += np.eye(N)
                f = st.A[j] @ f + st.b[j]
                xs.append((F.copy(), f.copy()))
            vo = N + N * L
            ys = []
            for k in range(L + 1):
                Fk, fk = xs[k]
                Gy = st.H[seq[k]] @ Fk
                Gy[:, vo + k * my:vo + (k + 1) * my] += np.eye(my)
                ys.append((Gy, st.H[seq[k]] @ fk + st.c[seq[k]]))
            xprev = xs[L - 1] if L >= 1 else (np.zeros((N, nsrc)), np.zeros(N))
            rowsG = [xprev[0], xs[L][0]] + [Gy for Gy, _ in ys]
            rowsg = [xprev[1], xs[L][1]] + [gy for _, gy in ys]
            G = np.vstack(rowsG)
            Ssrc = np.zeros((nsrc, nsrc))
            Ssrc[:N, :N] = S0
            for k in range(1, L + 1):
                Ssrc[N + (k - 1) * N:N + k * N, N + (k - 1) * N:N + k * N] = st.Q[seq[k]]
            for k in range(L + 1):
                Ssrc[vo + k * my:vo + (k + 1) * my, vo + k * my:vo + (k + 1) * my] = st.R[seq[k]]
            Sk = Ssrc.copy()
            if not head:
                Sk[:N, :N] = 0  # the window's first state known: a lower bound on the covariance
            P.append(p)
            comp.append([st.modes[j] for j in seq])
            G_.append(np.concatenate(rowsg))
            C_.append(G @ Ssrc @ G.T)
            K_.append(G @ Sk @ G.T)
        layout = {("x", -1): np.arange(0, N), ("x", 0): np.arange(N, 2 * N),
                  "y_past": np.arange(2 * N, 2 * N + L * my), "y_now": np.arange(2 * N + L * my, 2 * N + (L + 1) * my)}
        out = (np.array(P), np.array(comp), np.array(G_), np.array(C_), np.array(K_), layout)
        self._seq_cache[head] = out
        return out

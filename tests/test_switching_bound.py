"""Phase I: the expected collapse error computed from the model alone."""

import numpy as np

from egfg.switching import compile_switching_filter, run_filter
from egfg.switching_bound import ExpectedBound, mode_moments
from egfg.switching_ref import gpb2, imm, maneuver, marginals, simulate, stack


def test_mode_moments_follow_the_recursion_and_match_the_window_mixture():
    model = maneuver(2, 1)
    eb = ExpectedBound(model)  # moments at time 50
    st = eb.st
    pi, q, X = eb.pi, eb.q, eb.X
    # one step of the second-moment recursion of the jump linear system gives the moments at time 51
    nq = [sum(st.T[i, j] * (st.A[j] @ q[i] + st.b[j] * pi[i]) for i in range(2)) for j in range(2)]
    nX = [sum(st.T[i, j] * (st.A[j] @ X[i] @ st.A[j].T + pi[i] * st.Q[j]) for i in range(2)) for j in range(2)]
    _, q51, X51 = mode_moments(st, horizon=51)
    assert all(np.allclose(nq[j], q51[j]) and np.allclose(nX[j], X51[j]) for j in range(2))
    # the window (starting from the moments at time 50) mixed over its mode sequences gives the
    # moments of x at its end, time 50 + L, for each current mode
    P, comp, g, C, _, lay = eb._sequences(False)
    L = comp.shape[1] - 1
    pi, q, X = mode_moments(st, horizon=50 + L)
    xi = lay["x", 0]
    for j in range(2):
        sel = comp[:, L, 0] == j
        w = P[sel]
        mean = (w[:, None] * g[sel][:, xi]).sum(0)
        second = (w[:, None, None] * (C[sel][:, xi][:, :, xi] + g[sel][:, xi, None] * g[sel][:, None, xi])).sum(0)
        assert np.isclose(w.sum(), pi[j]) and np.allclose(mean, q[j]) and np.allclose(second, X[j])


def test_bound_is_above_the_expected_collapse_bound_in_the_first_step():
    # first step, the observation noise depends on the mode: the posteriors differ and collapsing costs
    model = maneuver(2, 1, obs_mode=True)
    eb = ExpectedBound(model)
    eps = eb(eb.site(("m@0",), "m@0", ("x@0",), (0,), True))
    st = stack(model)
    ys = np.linspace(-40, 40, 40001)
    dens, val = np.zeros_like(ys), np.zeros_like(ys)
    comps = []
    for j in range(2):
        m, P, H, R = st.mu0[j][0], st.P0[j][0, 0], st.H[j][0, 0], st.R[j][0, 0]
        S = H * P * H + R
        lik = np.exp(-0.5 * (ys - H * m) ** 2 / S) / np.sqrt(2 * np.pi * S)
        K = P * H / S
        comps.append((st.p0[j] * lik, m + K * (ys - H * m), P - K * H * P))
    dens = sum(c[0] for c in comps)
    w = [c[0] / dens for c in comps]
    mu = sum(wi * c[1] for wi, c in zip(w, comps))
    var = sum(wi * (c[2] + (c[1] - mu) ** 2) for wi, c in zip(w, comps))
    val = 0.5 * (np.log(var) - sum(wi * np.log(c[2]) for wi, c in zip(w, comps)))
    expected = float(np.sum(dens * val) * (ys[1] - ys[0]))
    assert expected > 1e-3 and eps >= expected - 1e-9


def test_price_moves_from_imm_to_gpb2():
    model = maneuver(2, 1)
    obs = simulate(model, 6, seed=5)
    st = stack(model)
    eb = ExpectedBound(model)

    def diff(p, ref):
        err = 0.0
        for r, e in zip(run_filter(p, obs), ref):
            probs, cont = marginals(st, e, model)
            err = max(err, float(np.max(np.abs(r.mean["x"] - cont["x"][0]))), float(np.max(np.abs(r.probs["m"] - probs["m"]))))
        return err

    cheap = compile_switching_filter(model, price=0.0, groups=[("m", "x")], bound=eb)
    careful = compile_switching_filter(model, price=1e4, groups=[("m", "x")], bound=eb)
    assert diff(cheap, imm(st, obs)) < 1e-8 and diff(careful, gpb2(st, obs)) < 1e-8
    assert cheap.step.flops < careful.step.flops and careful.step.eps < cheap.step.eps

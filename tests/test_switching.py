"""Phase H: switching models, mixture collapse as a chosen implementation, TV bounds."""

import numpy as np
import pytest

from egfg.gaussian import Moment
from egfg.switching import collapse_one, compile_switching_filter, compile_switching_programs, run_filter
from egfg.switching_ref import (
    exact_filter,
    gpb1,
    gpb2,
    imm,
    maneuver,
    marginals,
    message_joint,
    simulate,
    stack,
    targets,
    tv_1d,
)


def _max_diff(model, res, ref):
    st = stack(model)
    err = 0.0
    for r, e in zip(res, ref):
        probs, cont = marginals(st, e, model)
        for d in model.cards:
            err = max(err, float(np.max(np.abs(r.probs[d] - probs[d]))))
        for x in model.dims:
            err = max(err, float(np.max(np.abs(r.mean[x] - cont[x][0]))), float(np.max(np.abs(r.cov[x] - cont[x][1]))))
    return err


def test_collapse_matches_mixture_moments():
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(3):
        M = rng.normal(size=(2, 2))
        parts.append(Moment(("x",), rng.normal(size=2), M @ M.T + np.eye(2), float(rng.normal())))
    q, kl = collapse_one(parts)
    w = np.exp([p.logz for p in parts])
    w /= w.sum()
    mu = sum(wi * p.mu for wi, p in zip(w, parts))
    second = sum(wi * (p.S + np.outer(p.mu, p.mu)) for wi, p in zip(w, parts))
    assert np.allclose(q.mu, mu) and np.allclose(q.S, second - np.outer(mu, mu))
    assert np.isclose(q.logz, np.logaddexp.reduce([p.logz for p in parts])) and kl > 0
    same, kl0 = collapse_one([parts[0], Moment(("x",), parts[0].mu, parts[0].S, 0.3)])
    assert np.allclose(same.mu, parts[0].mu) and kl0 < 1e-12


@pytest.mark.parametrize("model", [maneuver(2, 1), maneuver(3, 2, obs_mode=True)])
def test_programs_rediscover_imm_gpb2_gpb1(model):
    obs = simulate(model, 6, seed=2)
    st = stack(model)
    refs = {"imm": imm(st, obs), "gpb2": gpb2(st, obs), "gpb1": gpb1(st, obs)}
    found = {}
    for p in compile_switching_programs(model):
        res = run_filter(p, obs)
        for name, ref in refs.items():
            if _max_diff(model, res, ref) < 1e-8:
                found[name] = (tuple(p.groups), p.lam)
    whole, split_ = (tuple(model.components()[0]),), ((("m",), ("x",)))
    assert found["imm"] == (whole, 0.0)  # the cheapest: merge before predicting
    assert found["gpb2"][0] == whole and found["gpb2"][1] > 0  # with a high penalty: merge after the update
    assert found["gpb1"][0] == split_  # the split message


def test_no_approximation_in_the_first_step_matches_exact():
    model = maneuver(2, 1)
    obs = simulate(model, 1, seed=3)
    p = compile_switching_filter(model, groups=[("m", "x")])
    assert _max_diff(model, run_filter(p, obs), exact_filter(stack(model), obs)) < 1e-10


def test_independent_targets_per_target_imm_equals_state_wide_imm():
    model = targets(2, 2, 1)
    obs = simulate(model, 5, seed=4)
    p = compile_switching_filter(model, lam=0.0, groups=[("m0", "x0"), ("m1", "x1")])
    assert _max_diff(model, run_filter(p, obs), imm(stack(model), obs)) < 1e-8


def test_tv_bound_holds_in_one_dimension():
    model = maneuver(2, 1)
    obs = simulate(model, 6, seed=1)
    exact = exact_filter(stack(model), obs)
    for p in compile_switching_programs(model):
        for r, e in zip(run_filter(p, obs), exact):
            modes, approx = message_joint(model, r.messages)
            assert tv_1d(e.comps, approx, modes) <= min(1.0, sum(m.tau for m in r.messages)) + 1e-6

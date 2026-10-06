"""Phase D: linear-Gaussian values, representation-aware extraction, Gaussian filtering."""

import numpy as np
import pytest

from egfg.gaussian import (
    GFactor,
    Info,
    Moment,
    info_mul,
    info_of_factor,
    info_of_moment,
    info_sum,
    log_factor,
    log_potential,
    moment_of_info,
    moment_of_prior,
    moment_predict,
    moment_sum,
    moment_update,
)

DIMS = {"a": 2, "b": 1, "c": 3}


def _spd(rng, n):
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _point(rng, vars_):
    return {v: rng.normal(size=DIMS[v]) for v in vars_}


def _flat(p, vars_):
    return np.concatenate([p[v] for v in vars_])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _cond(rng, child, parents):
    return GFactor(
        "cond", child, parents, tuple(rng.normal(size=(DIMS[child], DIMS[p])) for p in parents),
        rng.normal(size=DIMS[child]), _spd(rng, DIMS[child]),
    )


def _prior(rng, v):
    return GFactor("prior", v, (), (), rng.normal(size=DIMS[v]), _spd(rng, DIMS[v]))


def _obs(rng, parents, m=2):
    return GFactor("obs", None, parents, tuple(rng.normal(size=(m, DIMS[p])) for p in parents),
                   rng.normal(size=m), _spd(rng, m))


# --- values and operations ----------------------------------------------------


def test_factor_info_form_matches_density(rng):
    for f in (_prior(rng, "a"), _cond(rng, "c", ("a", "b")), _obs(rng, ("a", "c"))):
        y = rng.normal(size=len(f.b))
        inf = info_of_factor(f, DIMS, y)
        for _ in range(3):
            p = _point(rng, f.scope)
            assert np.isclose(log_potential(inf, _flat(p, inf.vars)), log_factor(f, DIMS, p, y))


def test_conversions_round_trip(rng):
    m = Moment(("a", "c"), rng.normal(size=5), _spd(rng, 5), 0.7)
    i = info_of_moment(m)
    m2 = moment_of_info(i)
    assert np.allclose(m.mu, m2.mu) and np.allclose(m.S, m2.S) and np.isclose(m.logz, m2.logz)
    x = rng.normal(size=5)
    assert np.isclose(log_potential(m, x), log_potential(i, x))


def test_info_mul_and_sum(rng):
    a = info_of_factor(_cond(rng, "c", ("a",)), DIMS)
    b = info_of_moment(Moment(("a",), rng.normal(size=2), _spd(rng, 2), 0.1))
    ab = info_mul(a, b, DIMS)
    x = _point(rng, ("a", "c"))
    assert np.isclose(log_potential(ab, _flat(x, ab.vars)),
                      log_potential(a, _flat(x, a.vars)) + log_potential(b, _flat(x, b.vars)))
    # ∫ da of a proper joint: compare with moment-form marginalization
    marg = info_sum(ab, "a", DIMS)
    mm = moment_sum(moment_of_info(ab), "a", DIMS)
    z = rng.normal(size=3)
    assert np.isclose(log_potential(marg, z), log_potential(mm, z))


def test_predict_matches_info_route(rng):
    m = Moment(("a", "b"), rng.normal(size=3), _spd(rng, 3), 0.2)
    f = _cond(rng, "c", ("a",))
    pm = moment_predict(m, f, DIMS)
    pi = info_mul(info_of_moment(m), info_of_factor(f, DIMS), DIMS)
    x = _point(rng, ("a", "b", "c"))
    assert pm.vars == pi.vars
    assert np.isclose(log_potential(pm, _flat(x, pm.vars)), log_potential(pi, _flat(x, pi.vars)))


def test_update_matches_info_route(rng):
    m = Moment(("a", "c"), rng.normal(size=5), _spd(rng, 5), -0.3)
    f = _obs(rng, ("c",), m=2)
    y = rng.normal(size=2)
    um = moment_update(m, f, y, DIMS)
    ui = info_mul(info_of_moment(m), info_of_factor(f, DIMS, y), DIMS)
    x = _point(rng, ("a", "c"))
    assert np.isclose(log_potential(um, _flat(x, um.vars)), log_potential(ui, _flat(x, ui.vars)))


def test_prior_moment_form(rng):
    f = _prior(rng, "c")
    x = rng.normal(size=3)
    assert np.isclose(log_potential(moment_of_prior(f), x), log_potential(info_of_factor(f, DIMS), x))


# --- representation-aware extraction -----------------------------------------

from egfg.dynamic import local_step  # noqa: E402
from egfg.egraph import saturate  # noqa: E402
from egfg.extract import _costs, class_scopes, extract_dag_ilp, extract_tree  # noqa: E402
from egfg.gdynamic import (  # noqa: E402
    compile_gaussian_filter,
    gaussian_filter,
    information_program,
    kalman_program,
    kalman_reference,
    normalizable,
    stack_obs,
)
from egfg.generators import (  # noqa: E402
    cycle,
    gaussian_blocks,
    gaussian_coupled,
    gaussian_observations,
    gaussian_vec,
    random_tree,
)
from egfg.ir import all_marginal_queries  # noqa: E402
from egfg.repextract import Impl, extract_rep  # noqa: E402


@pytest.mark.parametrize("fg", [random_tree(5, 3, 1), cycle(4, 2)])
def test_single_representation_matches_plain_extraction(fg):
    g = saturate(fg, all_marginal_queries(fg)).graph
    costs = _costs(g, fg, class_scopes(g, fg))
    impls = {
        (cid, "x"): [Impl(cid, "x", n, tuple((c, "x") for c in n.children), costs[(cid, i)], n.op)
                     for i, n in enumerate(nodes)]
        for cid, nodes in g.classes.items()
    }
    roots = {q: (cid, "x") for q, cid in g.roots.items()}
    assert extract_rep(g, impls, roots, "tree").cost == extract_tree(g, fg).cost
    assert extract_rep(g, impls, roots, "ilp").cost == extract_dag_ilp(g, fg).cost
    assert extract_rep(g, impls, roots, "greedy").cost <= extract_rep(g, impls, roots, "tree").cost


def test_normalizable_by_structure():
    model = gaussian_coupled(2, 1)
    facs = model.factors("step")  # transitions 0, 1; observations 2, 3
    fwd = frozenset({"x0@-1", "x1@-1"})
    assert normalizable(frozenset({0}), frozenset({"fwd"}), facs, fwd)  # fwd · p(x0 | x0')
    assert not normalizable(frozenset({0}), frozenset(), facs, fwd)  # p(x0 | x0') alone
    assert not normalizable(frozenset({2}), frozenset(), facs, fwd)  # a likelihood alone
    assert normalizable(frozenset({0, 1, 2, 3}), frozenset({"fwd"}), facs, fwd)
    head = model.factors("head")  # priors 0, 1; observations 2, 3
    assert normalizable(frozenset({0, 2}), frozenset(), head, fwd)


# --- Gaussian filtering -------------------------------------------------------

GMODELS = {
    "vec": gaussian_vec(3, 2),
    "blocks": gaussian_blocks(2, 2),
    "coupled": gaussian_coupled(3, 1),
}


def _same(res, ref, model):
    for r, q in zip(res, ref):
        for n in model.dims:
            assert np.allclose(r.mean[n], q.mean[n]) and np.allclose(r.cov[n], q.cov[n])
        assert np.isclose(r.loglik, q.loglik)


@pytest.mark.parametrize("name", list(GMODELS))
@pytest.mark.parametrize("reps", ["both", "info", "moment"])
@pytest.mark.parametrize("T", [1, 2, 6])
def test_gaussian_filter_matches_kalman(name, reps, T):
    model = GMODELS[name]
    obs = gaussian_observations(model, T, seed=T)
    prog = compile_gaussian_filter(model, reps=reps, extractor="greedy")
    _same(gaussian_filter(prog, obs), kalman_reference(model, obs), model)


@pytest.mark.parametrize("extractor", ["ilp", "tree"])
def test_gaussian_filter_other_extractors(extractor):
    model = gaussian_coupled(2, 1)
    obs = gaussian_observations(model, 4)
    prog = compile_gaussian_filter(model, extractor=extractor, time_limit_s=5)
    _same(gaussian_filter(prog, obs), kalman_reference(model, obs), model)


def test_baselines_are_correct():
    model = GMODELS["blocks"]
    obs = gaussian_observations(model, 5)
    ref = kalman_reference(model, obs)
    for prog in (kalman_program(model), information_program(model)):
        res = gaussian_filter(prog, stack_obs(obs))
        for r, q in zip(res, ref):
            assert np.allclose(r.mean["s"], np.concatenate([q.mean[n] for n in sorted(model.dims)]))
            assert np.isclose(r.loglik, q.loglik)


# --- costs --------------------------------------------------------------------


@pytest.mark.parametrize("d,m", [(2, 1), (4, 2), (3, 6)])
def test_baseline_costs_by_hand(d, m):
    model = gaussian_vec(d, m)
    # Kalman: predict 2d³, integrate the old state (free), update d²m + dm² + m³
    assert kalman_program(model).step.cost == 2 * d**3 + d * d * m + d * m * m + m**3
    # information filter: transition to info form 3d³, add 4d², Schur 2d³,
    # observation to info form m³ + dm² + d²m, add d², read the answer d³
    assert information_program(model).step.cost == 6 * d**3 + 5 * d * d + m**3 + d * m * m + d * d * m


@pytest.mark.parametrize("name", list(GMODELS))
def test_both_representations_not_worse(name):
    model = GMODELS[name]
    both = compile_gaussian_filter(model, extractor="greedy").step.cost
    assert both <= compile_gaussian_filter(model, reps="info", extractor="greedy").step.cost
    assert both <= compile_gaussian_filter(model, reps="moment", extractor="greedy").step.cost


def test_vec_matches_cheaper_baseline():
    # with parameter-only work counted once, the information filter wins for large m
    for d, m in [(4, 16), (8, 2), (2, 2)]:
        model = gaussian_vec(d, m)
        kf = kalman_program(model, amortize_constants=True).step.cost
        inf = information_program(model, amortize_constants=True).step.cost
        eg = compile_gaussian_filter(model, amortize_constants=True).step.cost
        assert eg == min(kf, inf)
    assert information_program(gaussian_vec(4, 16), amortize_constants=True).step.cost < \
        kalman_program(gaussian_vec(4, 16), amortize_constants=True).step.cost


def test_factored_state_beats_kalman():
    for model in (gaussian_blocks(3, 2), gaussian_coupled(3, 2)):
        assert compile_gaussian_filter(model, extractor="greedy").step.cost < kalman_program(model).step.cost


def test_latency_objective_splits_prediction():
    model = gaussian_coupled(2, 2)
    tot = compile_gaussian_filter(model, extractor="greedy")
    lat = compile_gaussian_filter(model, extractor="greedy", objective="latency")
    assert lat.step.latency_cost <= tot.step.latency_cost
    assert lat.step.prep_cost + lat.step.latency_cost == lat.step.cost
    obs = gaussian_observations(model, 4)
    _same(gaussian_filter(lat, obs), kalman_reference(model, obs), model)


def test_latency_objective_never_worse_in_latency():
    for model in (gaussian_coupled(3, 2), gaussian_coupled(2, 2), gaussian_blocks(2, 2)):
        for am in (False, True):
            tot = compile_gaussian_filter(model, extractor="greedy", rules="minimal", amortize_constants=am)
            lat = compile_gaussian_filter(model, extractor="greedy", rules="minimal", amortize_constants=am,
                                          objective="latency")
            assert lat.step.latency_cost <= tot.step.latency_cost

"""Phase B: time-step templates for filtering."""

import numpy as np
import pytest

from egfg.baselines import brute_force_joint, brute_force_marginals
from egfg.cost import dag_cost
from egfg.dynamic import (
    FWD,
    DynamicModel,
    Filter,
    compile_filter,
    filter_marginals,
    filter_max_marginals,
    filter_moments,
    forward_program,
    instantiate,
    jt_program,
    local_step,
    reference_filter,
    unroll,
)
from egfg.evaluate import SUM_PRODUCT, Table, evaluate
from egfg.generators import coupled_hmm, factorial_hmm, hmm, random_observations
from egfg.ir import Input, to_dag

MODELS = {
    "hmm": hmm(3),
    "fhmm2": factorial_hmm(2, 2),
    "chmm2": coupled_hmm(2, 2),
    "fhmm3": factorial_hmm(3, 2),
}


@pytest.fixture(scope="module")
def programs():
    return {k: compile_filter(m, extractor="greedy") for k, m in MODELS.items()}


def _prefix_marginals(model, obs, t):
    """Brute-force filtering answer at time t: marginals of the graph unrolled up to t."""
    bm = brute_force_marginals(unroll(model, obs[: t + 1]))
    return {n: bm[f"{n}@{t}"] for n in model.states}


# --- small extensions ---------------------------------------------------------


def test_evaluate_binds_input_leaves():
    fg = unroll(hmm(2), random_observations(hmm(2), 1))
    dag = to_dag({"q": Input("m")}, inputs={"m": frozenset({"a@0"})})
    t = Table(("a@0",), np.array([1.0, 3.0]))
    tables, _ = evaluate(dag, fg, SUM_PRODUCT, inputs={"m": t})
    assert np.allclose(tables["q"].data, [1.0, 3.0])
    with pytest.raises(ValueError):
        evaluate(dag, fg, SUM_PRODUCT)


# --- model and unrolling ------------------------------------------------------


def test_unroll_shape():
    model = coupled_hmm(3, 2)
    fg = unroll(model, random_observations(model, 4))
    assert len(fg.cards) == 3 * 4
    assert len(fg.factors) == (3 + 3) + 3 * (3 + 3)
    assert {f.scope for f in fg.factors if f.id == 6 + 1} == {("x1@0", "x0@0", "x1@1")}


def test_unroll_single_step():
    model = factorial_hmm(2, 3)
    fg = unroll(model, random_observations(model, 1))
    assert sorted(fg.cards) == ["x0@0", "x1@0"] and len(fg.factors) == 3


def test_bad_scope_rejected():
    with pytest.raises(ValueError):
        DynamicModel({"a": 2}, [(("a@-1",), np.ones(2))], [], [])


# --- templates ----------------------------------------------------------------


def test_step_problem_does_not_depend_on_time():
    # the local problem is built from relative names only: building it twice gives the same terms
    model = coupled_hmm(2, 3)
    a, b = local_step(model, "step"), local_step(model, "step")
    assert a.queries == b.queries and a.seeds == b.seeds and a.inputs == {FWD: frozenset({"x0@-1", "x1@-1"})}
    assert local_step(model, "head").inputs == {}


# --- filtering ----------------------------------------------------------------


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("T", [1, 2, 3, 5])
def test_filtering_matches_brute_force(programs, name, T):
    model = MODELS[name]
    obs = random_observations(model, T, seed=T)
    for prog in (programs[name], forward_program(model)):
        fm = filter_marginals(prog, obs)
        for t in range(T):
            bf = _prefix_marginals(model, obs, t)
            assert all(np.allclose(fm[t][n], bf[n]) for n in model.states)


def test_online_steps_match_batch(programs):
    model = MODELS["chmm2"]
    obs = random_observations(model, 6, seed=1)
    batch = filter_marginals(programs["chmm2"], obs)
    f = Filter(programs["chmm2"])
    for t, tabs in enumerate(obs):
        out = f.step(tabs)
        assert all(np.allclose(out[n].data / out[n].data.sum(), batch[t][n]) for n in model.states)


def test_long_run_no_underflow(programs):
    model = MODELS["fhmm2"]
    obs = random_observations(model, 1024, seed=2)
    fm = filter_marginals(programs["fhmm2"], obs)
    ref = reference_filter(model, obs)
    assert all(np.all(np.isfinite(fm[t][n])) for t in range(1024) for n in model.states)
    assert all(np.allclose(fm[t][n], ref[t][n]) for t in (0, 511, 1023) for n in model.states)


def test_reference_filter_matches_brute_force():
    model = coupled_hmm(2, 3)
    obs = random_observations(model, 4, seed=3)
    ref = reference_filter(model, obs)
    for t in range(4):
        bf = _prefix_marginals(model, obs, t)
        assert all(np.allclose(ref[t][n], bf[n]) for n in model.states)


# --- cost ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MODELS))
def test_total_cost_matches_instantiated_dag(programs, name):
    model = MODELS[name]
    for prog in (programs[name], forward_program(model)):
        for T in (1, 2, 4):
            fg = unroll(model, random_observations(model, T))
            assert dag_cost(instantiate(prog, T), fg) == prog.total_cost(T)


def test_instantiated_dag_evaluates_to_filtering(programs):
    model = MODELS["fhmm2"]
    obs = random_observations(model, 3, seed=4)
    tables, _ = evaluate(instantiate(programs["fhmm2"], 3), unroll(model, obs), SUM_PRODUCT)
    fm = filter_marginals(programs["fhmm2"], obs)
    for t in range(3):
        for n in model.states:
            d = tables[f"{n}@{t}"].data
            assert np.allclose(d / d.sum(), fm[t][n])


@pytest.mark.parametrize("name", ["fhmm2", "chmm2", "fhmm3"])
def test_egraph_not_worse_than_forward(programs, name):
    assert programs[name].step.cost <= forward_program(MODELS[name]).step.cost


def test_seed_bounds_cost_under_tiny_node_limit():
    model = factorial_hmm(3, 3)
    prog = compile_filter(model, extractor="ilp", node_limit=20)
    assert prog.step.saturation.hit_limit
    jt = jt_program(model)
    assert prog.step.cost <= jt.step.cost and prog.head.cost <= jt.head.cost


def test_jt_program_is_correct():
    model = MODELS["chmm2"]
    obs = random_observations(model, 3, seed=7)
    fm = filter_marginals(jt_program(model), obs)
    for t in range(3):
        bf = _prefix_marginals(model, obs, t)
        assert all(np.allclose(fm[t][n], bf[n]) for n in model.states)


# --- semirings ----------------------------------------------------------------


def test_filter_max_marginals_match_brute_force(programs):
    model = MODELS["chmm2"]
    obs = random_observations(model, 3, seed=5)
    mm = filter_max_marginals(programs["chmm2"], obs)
    for t in range(3):
        fg = unroll(model, obs[: t + 1])
        J = brute_force_joint(fg)
        vs = fg.variables()
        for n in model.states:
            i = vs.index(f"{n}@{t}")
            ref = J.max(axis=tuple(j for j in range(len(vs)) if j != i))
            assert np.allclose(mm[t][n], ref / ref.sum())


def test_filter_moments_match_brute_force(programs):
    model = MODELS["fhmm2"]
    obs = random_observations(model, 4, seed=6)
    vals = np.array([1.0, 5.0])
    for t in (0, 2, 3):
        p = _prefix_marginals(model, obs, t)["x1"]
        E = float((p * vals).sum())
        V = float((p * vals**2).sum()) - E * E
        assert np.allclose(filter_moments(programs["fhmm2"], obs, "x1", t, vals), (E, V))

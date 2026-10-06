"""Phase C: preparation vs latency cost, latency-weighted extraction, prepare()/update()."""

import numpy as np
import pytest

from egfg.baselines import brute_force_marginals
from egfg.cost import dag_cost, split_cost
from egfg.dynamic import Filter, compile_filter, filter_marginals, forward_program, local_step, unroll
from egfg.egraph import saturate
from egfg.extract import class_leaves, extract_dag_greedy, extract_dag_ilp, extract_tree
from egfg.generators import chain, coupled_hmm, factorial_hmm, hmm, random_observations, random_tree
from egfg.ir import all_marginal_queries

MODELS = {"hmm": hmm(3), "fhmm2": factorial_hmm(2, 2), "chmm2": coupled_hmm(2, 3), "fhmm3": factorial_hmm(3, 2)}


def _prefix_marginals(model, obs, t):
    bm = brute_force_marginals(unroll(model, obs[: t + 1]))
    return {n: bm[f"{n}@{t}"] for n in model.states}


# --- dependence on factors ----------------------------------------------------


@pytest.mark.parametrize("model", [coupled_hmm(2, 2), factorial_hmm(2, 2)])
def test_every_node_of_an_eclass_has_the_same_factors(model):
    loc = local_step(model, "step")
    g = saturate(loc.fg, loc.queries, inputs=loc.inputs, seeds=loc.seeds).graph
    leaves = class_leaves(g)
    for cid, nodes in g.classes.items():
        for n in nodes:
            below = frozenset({n.arg}) if n.op == "leaf" else frozenset().union(*(leaves[c] for c in n.children))
            assert below == leaves[cid]


def test_no_weights_changes_nothing():
    fg = random_tree(6, 2, 4)
    g = saturate(fg, all_marginal_queries(fg)).graph
    for ex in (extract_tree, extract_dag_greedy, extract_dag_ilp):
        a, b = ex(g, fg), ex(g, fg, weights={})
        assert a.cost == b.cost == a.weighted_cost == b.weighted_cost


# --- cost split ---------------------------------------------------------------


def test_split_cost_hmm_by_hand():
    # forward step of an HMM with K=3: fwd·T (9) and Σ (9) need no observation; ·O (3) does
    prog = forward_program(hmm(3))
    assert (prog.step.prep_cost, prog.step.latency_cost) == (18, 3)
    assert (prog.head.prep_cost, prog.head.latency_cost) == (0, 3)


@pytest.mark.parametrize("name", list(MODELS))
def test_split_adds_up(name):
    for obj in ("total", "latency"):
        prog = compile_filter(MODELS[name], extractor="greedy", objective=obj)
        for t in (prog.head, prog.step):
            assert t.prep_cost + t.latency_cost == t.cost == dag_cost(t.dag, t.local.fg)
            assert split_cost(t.dag, t.local.fg, t.local.obs_ids) == (t.prep_cost, t.latency_cost)


# --- objectives ---------------------------------------------------------------


@pytest.mark.parametrize("name", list(MODELS))
def test_total_objective_is_phase_b(name):
    a = compile_filter(MODELS[name], extractor="greedy")
    b = compile_filter(MODELS[name], extractor="greedy", objective="total")
    assert (a.head.cost, a.step.cost) == (b.head.cost, b.step.cost)


@pytest.mark.parametrize("extractor", ["greedy", "ilp"])
@pytest.mark.parametrize("name", list(MODELS))
def test_latency_objective_beats_total_and_forward(name, extractor):
    model = MODELS[name]
    tot = compile_filter(model, extractor=extractor)
    lat = compile_filter(model, extractor=extractor, objective="latency")
    fwd = forward_program(model)
    for kind in ("head", "step"):
        t, l, f = getattr(tot, kind), getattr(lat, kind), getattr(fwd, kind)
        assert l.latency_cost <= t.latency_cost
        assert l.latency_cost <= f.latency_cost


def test_latency_objective_moves_work_before_the_observation():
    # coupled HMM: the total-cost optimum multiplies one observation early, so more waits for it
    model = coupled_hmm(2, 3)
    tot = compile_filter(model, extractor="ilp")
    lat = compile_filter(model, extractor="ilp", objective="latency")
    assert lat.step.latency_cost < tot.step.latency_cost
    assert lat.step.cost >= tot.step.cost


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        compile_filter(hmm(2), extractor="greedy", objective="fast")


# --- prepare / update ---------------------------------------------------------


@pytest.mark.parametrize("name", list(MODELS))
def test_prepare_update_matches_step_and_brute_force(name):
    model = MODELS[name]
    obs = random_observations(model, 4, seed=8)
    prog = compile_filter(model, extractor="greedy", objective="latency")
    batch = filter_marginals(prog, obs)
    f = Filter(prog)
    for t, tabs in enumerate(obs):
        f.prepare()  # no observation tables needed
        out = f.update(tabs)
        bf = _prefix_marginals(model, obs, t)
        for n in model.states:
            p = out[n].data / out[n].data.sum()
            assert np.allclose(p, batch[t][n]) and np.allclose(p, bf[n])


@pytest.mark.parametrize("obj", ["total", "latency", "weighted:2", "weighted:10"])
def test_every_objective_is_correct(obj):
    model = coupled_hmm(2, 2)
    obs = random_observations(model, 5, seed=9)
    fm = filter_marginals(compile_filter(model, extractor="ilp", objective=obj), obs)
    for t in range(5):
        bf = _prefix_marginals(model, obs, t)
        assert all(np.allclose(fm[t][n], bf[n]) for n in model.states)


def test_phase1_pipeline_unaffected():
    from egfg.pipeline import marginals, optimize

    fg = chain(4, 2, 3)
    res = optimize(fg)
    bm = brute_force_marginals(fg)
    assert all(np.allclose(marginals(fg, res)[v], bm[v]) for v in fg.variables())

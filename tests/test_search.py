"""Phase E: search strategies and the strong baselines."""

import numpy as np
import pytest

from egfg.baselines import best_junction_tree, brute_force_marginals, junction_tree_dag
from egfg.cost import dag_cost
from egfg.egraph import saturate
from egfg.evaluate import SUM_PRODUCT, evaluate
from egfg.extract import extract_dag_greedy, extract_dag_ilp
from egfg.generators import chain, cycle, grid, random_small, random_tree, star
from egfg.ir import all_marginal_queries
from egfg.jtree import ORDER_CRITERIA, elimination_order
from egfg.search import STRATEGIES, SearchConfig, extract_result, search


def _correct(dag, fg) -> bool:
    tables, _ = evaluate(dag, fg, SUM_PRODUCT)
    bm = brute_force_marginals(fg)
    return all(np.allclose(tables[v].data / tables[v].data.sum(), bm[v]) for v in fg.variables())


def test_bfs_is_saturate():
    fg = cycle(5, 2)
    q = all_marginal_queries(fg)
    a = saturate(fg, q, node_limit=5000)
    b = search(fg, q, SearchConfig(strategy="bfs", node_limit=5000))
    assert a.num_nodes == b.num_nodes and a.hit_limit == b.hit_limit
    assert extract_dag_greedy(a.graph, fg).cost == extract_dag_greedy(b.graph, fg).cost


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("seed", range(6))
def test_every_strategy_is_correct(strategy, seed):
    fg = random_small(seed)
    res = search(fg, all_marginal_queries(fg), SearchConfig(strategy=strategy, node_limit=3000))
    assert _correct(extract_result(res, fg).dag, fg)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies_on_larger_graphs_are_correct(strategy):
    for fg in (star(6, 2), grid(2, 3, 2)):
        res = search(fg, all_marginal_queries(fg), SearchConfig(strategy=strategy, node_limit=8000))
        assert _correct(extract_result(res, fg).dag, fg)


def test_staged_reaches_the_bfs_optimum_when_saturated():
    fg = chain(4, 2)
    q = all_marginal_queries(fg)
    a = search(fg, q, SearchConfig(strategy="bfs"))
    b = search(fg, q, SearchConfig(strategy="staged", max_steps=100))
    assert not a.hit_limit and not b.hit_limit
    assert extract_dag_ilp(a.graph, fg).cost == extract_dag_ilp(b.graph, fg).cost


@pytest.mark.parametrize("fg", [star(7, 3), cycle(7, 3), random_tree(9, 3, 1)])
def test_seeds_not_worse_than_best_jt(fg):
    res = search(fg, all_marginal_queries(fg), SearchConfig(strategy="seeds", node_limit=5000))
    assert extract_result(res, fg).cost <= best_junction_tree(fg)[1]


def test_restart_cost_never_increases():
    fg = grid(2, 4, 3)
    res = search(fg, all_marginal_queries(fg), SearchConfig(strategy="restart", node_limit=10000), trace=True)
    costs = [p.cost for p in res.trace]
    assert costs and all(b <= a for a, b in zip(costs, costs[1:]))


def test_trace_records_progress():
    fg = star(6, 3)
    res = search(fg, all_marginal_queries(fg), SearchConfig(strategy="staged", node_limit=10000), trace=True)
    assert res.trace[0].step == 0 and len(res.trace) >= 2
    assert all(p.nodes > 0 and p.cost > 0 for p in res.trace)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        search(chain(3, 2), all_marginal_queries(chain(3, 2)), SearchConfig(strategy="dfs"))


# --- strong baselines ---------------------------------------------------------


@pytest.mark.parametrize("criterion", ORDER_CRITERIA)
def test_elimination_orders_are_permutations(criterion):
    fg = grid(3, 3, 2)
    for rng in (None, np.random.default_rng(1)):
        order = elimination_order(fg, criterion, rng)
        assert sorted(order) == fg.variables()


@pytest.mark.parametrize("fg", [star(7, 3), grid(3, 3, 2), cycle(6, 3), random_tree(8, 3, 2)])
def test_best_jt_baselines(fg):
    jt = dag_cost(junction_tree_dag(fg)[0], fg)
    dag1, c1, _ = best_junction_tree(fg)
    dag2, c2, _ = best_junction_tree(fg, share_products=True)
    assert c2 <= c1 <= jt
    assert _correct(dag1, fg) and _correct(dag2, fg)


# --- the default strategy -----------------------------------------------------


def test_optimize_default_is_seeds_staged_and_bfs_is_phase1():
    from egfg.pipeline import marginals, optimize

    fg = random_tree(12, 3, 0)
    new = optimize(fg, extractor="greedy")
    old = optimize(fg, extractor="greedy", strategy="bfs")
    ref = saturate(fg, all_marginal_queries(fg))
    assert old.saturation.num_nodes == ref.num_nodes and old.saturation.iterations == ref.iterations
    assert new.extraction.cost <= best_junction_tree(fg)[1] < old.extraction.cost
    bm = brute_force_marginals(fg)
    assert all(np.allclose(marginals(fg, new)[v], bm[v]) for v in fg.variables())


def test_local_problems_drop_order_seeds():
    # clusters have Input leaves: "seeds+staged" falls back to staged with the cluster's own seeds
    from egfg.pipeline import marginals, optimize

    fg = grid(2, 5, 2, 1)
    res = optimize(fg, cluster_budget=3, seed=True, extractor="greedy")
    bm = brute_force_marginals(fg)
    assert all(np.allclose(marginals(fg, res)[v], bm[v]) for v in fg.variables())


def test_filter_default_strategy_is_correct():
    from egfg.dynamic import compile_filter, filter_marginals, reference_filter
    from egfg.generators import coupled_hmm, random_observations

    model = coupled_hmm(3, 2)
    obs = random_observations(model, 5)
    fm = filter_marginals(compile_filter(model, extractor="greedy"), obs)
    ref = reference_filter(model, obs)
    assert all(np.allclose(fm[t][n], ref[t][n]) for t in range(5) for n in model.states)

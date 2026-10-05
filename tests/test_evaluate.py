import numpy as np

from egfg.baselines import brute_force_marginals, brute_force_max_value, brute_force_moments
from egfg.cost import dag_cost
from egfg.evaluate import MAX_PRODUCT, SUM_PRODUCT, ExpectationSemiring, evaluate, value_feature
from egfg.generators import chain, random_small
from egfg.ir import all_marginal_queries, to_dag


def test_sum_product_matches_brute_force():
    fg = chain(4, 3, seed=2)
    dag = to_dag(all_marginal_queries(fg))
    tables, flops = evaluate(dag, fg, SUM_PRODUCT)
    for v, t in tables.items():
        assert t.vars == (v,)
        assert np.allclose(t.data / t.data.sum(), brute_force_marginals(fg)[v])
    assert flops == dag_cost(dag, fg)


def test_max_product_max_value():
    fg = chain(4, 3, seed=3)
    tables, _ = evaluate(to_dag(all_marginal_queries(fg)), fg, MAX_PRODUCT)
    assert np.isclose(tables["x0"].data.max(), brute_force_max_value(fg))


def test_expectation_semiring_moments():
    fg = chain(4, 3, seed=4)
    vals = np.array([0.0, 1.0, 5.0])
    sr = ExpectationSemiring(value_feature(fg, "x2", vals))
    tables, _ = evaluate(to_dag(all_marginal_queries(fg)), fg, sr)
    p, r, s = (tables["x0"].data[i].sum() for i in range(3))
    E, V = brute_force_moments(fg, "x2", vals)
    assert np.isclose(r / p, E) and np.isclose(s / p - (r / p) ** 2, V)


def test_random_small_sum_product():
    for seed in range(20):
        fg = random_small(seed)
        tables, _ = evaluate(to_dag(all_marginal_queries(fg)), fg, SUM_PRODUCT)
        bm = brute_force_marginals(fg)
        for v, t in tables.items():
            assert np.allclose(t.data / t.data.sum(), bm[v])

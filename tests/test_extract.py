import numpy as np

from egfg.baselines import brute_force_marginals
from egfg.cost import dag_cost
from egfg.egraph import saturate
from egfg.evaluate import SUM_PRODUCT, evaluate
from egfg.extract import extract_dag_ilp, extract_tree
from egfg.generators import chain, random_tree
from egfg.ir import all_marginal_queries
from helpers import chain3


def test_chain3_ilp_optimum():
    fg = chain3()
    g = saturate(fg, all_marginal_queries(fg)).graph
    ex = extract_dag_ilp(g, fg)
    # 手計算：Σc f1 (12) + Σa f0 (6) + p(b)=積 (3) + p(a)=Σb(f0·m) (6+6) + p(c)=Σb(f1·m) (12+12)
    assert ex.optimal and ex.cost == 57
    assert ex.cost == dag_cost(ex.dag, fg)


def test_ilp_not_worse_than_tree():
    fg = random_tree(6, 3, seed=5)
    g = saturate(fg, all_marginal_queries(fg)).graph
    assert extract_dag_ilp(g, fg).cost <= extract_tree(g, fg).cost


def test_extracted_dag_is_correct():
    fg = chain(5, 2, seed=6)
    g = saturate(fg, all_marginal_queries(fg)).graph
    for ex in (extract_dag_ilp(g, fg), extract_tree(g, fg)):
        tables, _ = evaluate(ex.dag, fg, SUM_PRODUCT)
        for v, t in tables.items():
            assert np.allclose(t.data / t.data.sum(), brute_force_marginals(fg)[v])


def test_ilp_time_out_falls_back_to_valid_result():
    fg = random_tree(6, 2, seed=0)
    g = saturate(fg, all_marginal_queries(fg)).graph
    ex = extract_dag_ilp(g, fg, time_limit_s=0.0)
    assert ex.optimal is False
    assert ex.cost <= extract_tree(g, fg).cost
    tables, _ = evaluate(ex.dag, fg, SUM_PRODUCT)
    for v, t in tables.items():
        assert np.allclose(t.data / t.data.sum(), brute_force_marginals(fg)[v])

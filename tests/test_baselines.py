import numpy as np
import pytest

from egfg.baselines import brute_force_marginals, junction_tree_dag, min_fill_order, opt_einsum_cost
from egfg.cost import dag_cost
from egfg.evaluate import SUM_PRODUCT, evaluate
from egfg.generators import chain, cycle, grid, random_tree
from helpers import chain3


def test_min_fill_chain3():
    assert min_fill_order(chain3()) == ["a", "b", "c"]


def test_jt_chain3_cost_and_messages():
    dag, msgs = junction_tree_dag(chain3())
    # δ(ab→bc)=Σa f0 (6)、δ(bc→ab)=Σc f1 (12)、積 f0·δ(bc→ab) (6、p(a) と p(b) で共有)、
    # p(a)=Σb(…) (6)、p(b)=Σa(…) (6)、p(c)=Σb(f1·δ(ab→bc)) (12+12)
    assert dag_cost(dag, chain3()) == 60
    assert set(msgs) == {(frozenset({0}), frozenset({"b"})), (frozenset({1}), frozenset({"b"}))}


@pytest.mark.parametrize(
    "fg", [random_tree(7, 2, 1), cycle(5, 2, 2), grid(2, 3, 2, 3), chain(1 + 1, 3, 4)]
)
def test_jt_correct(fg):
    dag, _ = junction_tree_dag(fg)
    tables, _ = evaluate(dag, fg, SUM_PRODUCT)
    bm = brute_force_marginals(fg)
    assert set(tables) == set(fg.variables())
    for v, t in tables.items():
        assert np.allclose(t.data / t.data.sum(), bm[v])


def test_opt_einsum_cost_positive():
    assert opt_einsum_cost(chain(4, 3)) > 0

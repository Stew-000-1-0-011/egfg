"""Phase F: low-rank factorization equalities."""

import numpy as np
import pytest

from egfg.baselines import best_junction_tree, brute_force_marginals
from egfg.evaluate import SUM_PRODUCT, evaluate
from egfg.generators import chain, cycle, grid, star
from egfg.ir import to_dag
from egfg.model import Factor, FactorGraph
from egfg.pipeline import marginals, mode, moments, optimize
from egfg.structure import low_rank, low_rank_tables


def test_split_equality_is_numerically_exact():
    fg = low_rank_tables(chain(3, 5), 2)
    st = low_rank(fg)
    assert st.splits and all(sp.rank == 2 for sp in st.splits)
    for leaf, term in st.equalities:
        dag = to_dag({"a": leaf, "b": term})
        t, _ = evaluate(dag, st.fg, SUM_PRODUCT)
        assert np.allclose(t["a"].data, t["b"].data)


def test_three_variable_factor_bipartitions():
    rng = np.random.default_rng(0)
    U, V = rng.uniform(0.1, 1, (3, 2)), rng.uniform(0.1, 1, (2, 4, 4))
    fg = FactorGraph({"a": 3, "b": 4, "c": 4}, [Factor(0, ("a", "b", "c"), np.einsum("ar,rbc->abc", U, V))])
    st = low_rank(fg)
    assert any(sp.left == ("a",) and sp.rank == 2 for sp in st.splits)


def test_full_rank_tables_are_not_split():
    st = low_rank(chain(4, 3))
    assert st.splits == [] and st.replaced is None


@pytest.mark.parametrize("make", [lambda: chain(5, 6), lambda: star(5, 6), lambda: cycle(5, 6), lambda: grid(2, 3, 5)])
@pytest.mark.parametrize("rank", [1, 2])
def test_low_rank_marginals_match_brute_force(make, rank):
    fg = low_rank_tables(make(), rank)
    res = optimize(fg, extractor="greedy", structure=("lowrank",))
    bm = brute_force_marginals(fg)
    assert all(np.allclose(marginals(fg, res)[v], bm[v]) for v in fg.variables())


@pytest.mark.parametrize("make", [lambda: chain(6, 8), lambda: cycle(6, 8)])
def test_low_rank_is_cheaper_and_beats_replaced_jt(make):
    fg = low_rank_tables(make(), 2)
    plain = optimize(fg, extractor="greedy").extraction.cost
    lr = optimize(fg, extractor="greedy", structure=("lowrank",))
    assert lr.extraction.cost < plain
    assert lr.extraction.cost <= best_junction_tree(low_rank(fg).replaced, share_products=True)[1]


def test_unprofitable_split_is_not_used():
    # rank 4 of 8 on a chain: the factorized junction tree is worse than the plain computation
    fg = low_rank_tables(chain(6, 8), 4)
    plain = optimize(fg, extractor="greedy").extraction.cost
    lr = optimize(fg, extractor="greedy", structure=("lowrank",)).extraction.cost
    assert lr <= plain < best_junction_tree(low_rank(fg).replaced, share_products=True)[1]


def test_sum_only_results_refuse_mode_and_moments():
    fg = low_rank_tables(chain(4, 4), 1)
    res = optimize(fg, extractor="greedy", structure=("lowrank",))
    assert res.sum_product_only
    with pytest.raises(ValueError):
        mode(fg, res)
    with pytest.raises(ValueError):
        moments(fg, res, "x0", np.arange(4.0))


def test_structure_with_clusters_rejected():
    with pytest.raises(ValueError):
        optimize(chain(4, 2), cluster_budget=3, structure=("lowrank",))
    with pytest.raises(ValueError):
        optimize(chain(4, 2), structure=("fft",))


def test_negative_tables_rejected_by_default():
    with pytest.raises(ValueError):
        FactorGraph({"a": 2}, [Factor(0, ("a",), np.array([1.0, -1.0]))])
    FactorGraph({"a": 2}, [Factor(0, ("a",), np.array([1.0, -1.0]))], allow_negative=True)

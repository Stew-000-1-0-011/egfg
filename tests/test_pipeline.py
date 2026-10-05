import numpy as np
import pytest

from egfg.baselines import brute_force_marginals, brute_force_max_value, brute_force_moments, joint_value
from egfg.generators import chain, random_small
from egfg.model import Factor, FactorGraph
from egfg.pipeline import marginals, mode, moments, optimize


@pytest.mark.parametrize("seed", range(30))
def test_random_small_matches_brute_force(seed):
    fg = random_small(seed)
    res = optimize(fg)
    bm = brute_force_marginals(fg)
    m = marginals(fg, res)
    assert all(np.allclose(m[v], bm[v]) for v in fg.variables())
    assert np.isclose(joint_value(fg, mode(fg, res)), brute_force_max_value(fg))
    v = fg.variables()[0]
    vals = np.arange(fg.cards[v], dtype=float)
    assert np.allclose(moments(fg, res, v, vals), brute_force_moments(fg, v, vals))


def test_single_variable_single_factor():
    fg = FactorGraph({"a": 3}, [Factor(0, ("a",), np.array([1.0, 2.0, 1.0]))])
    res = optimize(fg)
    assert np.allclose(marginals(fg, res)["a"], [0.25, 0.5, 0.25]) and mode(fg, res) == {"a": 1}


def test_tied_mode_is_a_maximizer():
    fg = FactorGraph({"a": 2, "b": 2}, [Factor(0, ("a", "b"), np.ones((2, 2)))])
    res = optimize(fg)
    assert np.isclose(joint_value(fg, mode(fg, res)), 1.0)


def test_tied_mode_xor():
    # 同時最大は (0,0) と (1,1) の 2 つ。変数ごとに独立に argmax すると (0,1) になりうる。
    t = np.array([[1.0, 0.1], [0.1, 1.0]])
    fg = FactorGraph({"a": 2, "b": 2}, [Factor(0, ("a", "b"), t)])
    res = optimize(fg)
    assert np.isclose(joint_value(fg, mode(fg, res)), 1.0)


def test_limit_hit_still_correct():
    fg = chain(6, 2, seed=7)
    res = optimize(fg, node_limit=200)
    assert res.saturation.hit_limit
    bm = brute_force_marginals(fg)
    assert all(np.allclose(marginals(fg, res)[v], bm[v]) for v in fg.variables())


def test_tree_extractor_option():
    fg = chain(4, 2, seed=8)
    res = optimize(fg, extractor="tree")
    assert res.extraction.optimal is None
    assert np.allclose(marginals(fg, res)["x1"], brute_force_marginals(fg)["x1"])

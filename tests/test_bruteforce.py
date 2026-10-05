import numpy as np

from egfg.baselines import (
    brute_force_joint,
    brute_force_marginals,
    brute_force_max_value,
    brute_force_moments,
    joint_value,
)
from egfg.generators import chain
from helpers import chain3


def test_brute_force_marginals_sum_to_one():
    m = brute_force_marginals(chain(4, 3, seed=1))
    assert all(np.isclose(p.sum(), 1.0) for p in m.values())


def test_joint_matches_product():
    fg = chain3()
    J = brute_force_joint(fg)
    f0, f1 = fg.factor(0).table, fg.factor(1).table
    assert J.shape == (2, 3, 4)
    assert np.isclose(J[1, 2, 3], f0[1, 2] * f1[2, 3])
    assert np.isclose(joint_value(fg, {"a": 1, "b": 2, "c": 3}), J[1, 2, 3])
    assert np.isclose(brute_force_max_value(fg), J.max())


def test_moments():
    fg = chain3()
    vals = np.array([0.0, 2.0, 5.0])
    pb = brute_force_marginals(fg)["b"]
    E, V = brute_force_moments(fg, "b", vals)
    assert np.isclose(E, (pb * vals).sum()) and np.isclose(V, (pb * vals**2).sum() - E**2)

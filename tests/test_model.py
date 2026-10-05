import numpy as np
import pytest

from egfg.model import Factor, FactorGraph


def test_valid_graph():
    fg = FactorGraph({"a": 2, "b": 3}, [Factor(0, ("a", "b"), np.ones((2, 3)))])
    assert fg.variables() == ["a", "b"] and fg.size(["a", "b"]) == 6 and fg.size([]) == 1
    assert fg.factor(0).scope == ("a", "b")


def test_uncovered_variable_rejected():
    with pytest.raises(ValueError):
        FactorGraph({"a": 2, "z": 2}, [Factor(0, ("a",), np.ones(2))])


def test_unknown_variable_rejected():
    with pytest.raises(ValueError):
        FactorGraph({"a": 2}, [Factor(0, ("a", "q"), np.ones((2, 2)))])


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        FactorGraph({"a": 2, "b": 3}, [Factor(0, ("a", "b"), np.ones((3, 2)))])


def test_negative_rejected():
    with pytest.raises(ValueError):
        FactorGraph({"a": 2}, [Factor(0, ("a",), np.array([1.0, -1.0]))])


def test_duplicate_factor_id_rejected():
    with pytest.raises(ValueError):
        FactorGraph({"a": 2}, [Factor(0, ("a",), np.ones(2)), Factor(0, ("a",), np.ones(2))])

import numpy as np

from egfg.generators import chain, cycle, grid, random_small, random_tree, star
from egfg.model import FactorGraph


def _connected(fg: FactorGraph) -> bool:
    vs = fg.variables()
    seen = {vs[0]}
    changed = True
    while changed:
        changed = False
        for f in fg.factors:
            if seen & set(f.scope) and not set(f.scope) <= seen:
                seen |= set(f.scope)
                changed = True
    return seen == set(vs)


def test_sizes():
    assert len(chain(5, 3).factors) == 4
    assert len(cycle(4, 2).factors) == 4
    assert len(grid(2, 3, 2).factors) == 7
    assert len(star(5, 2).factors) == 4
    t = random_tree(6, 2)
    assert len(t.factors) == 5 and _connected(t)


def test_cards_and_names():
    fg = chain(3, 4)
    assert fg.variables() == ["x0", "x1", "x2"] and set(fg.cards.values()) == {4}


def test_random_small_valid():
    for s in range(50):
        fg = random_small(s)
        assert isinstance(fg, FactorGraph)
        assert 1 <= len(fg.cards) <= 5 and 1 <= len(fg.factors) <= 5


def test_deterministic():
    a, b = random_tree(6, 3, seed=9), random_tree(6, 3, seed=9)
    assert [f.scope for f in a.factors] == [f.scope for f in b.factors]
    assert all(np.array_equal(x.table, y.table) for x, y in zip(a.factors, b.factors))

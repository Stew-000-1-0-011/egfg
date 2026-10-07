import time

import numpy as np
import pytest

from egfg.baselines import brute_force_marginals, junction_tree_dag
from egfg.cost import dag_cost
from egfg.decompose import stitch
from egfg.evaluate import SUM_PRODUCT, evaluate
from egfg.generators import cycle, grid, random_tree
from egfg.jtree import JunctionTree
from egfg.model import Factor, FactorGraph
from egfg.pipeline import marginals, optimize
from egfg.partition import PartitionInfo, PartitionSearch, initial_state


@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("fg", [cycle(8, 3), grid(3, 3, 2), random_tree(9, 3)], ids=["cycle", "grid", "tree"])
def test_searched_partition_is_exact_and_never_worse_than_the_junction_tree(fg, boundary):
    res = optimize(fg, extractor="greedy", partition="search", partition_time_s=20, boundary=boundary)
    ref = brute_force_marginals(fg)
    got = marginals(fg, res)
    for v in fg.variables():
        np.testing.assert_allclose(got[v], ref[v], atol=1e-12)
    info = res.partition
    assert res.extraction.cost <= dag_cost(junction_tree_dag(fg)[0], fg)
    # the search only accepts strictly cheaper states, and the phases never increase the cost
    costs = [c for _, c in info.accepted]
    assert all(b < a for a, b in zip(costs, costs[1:]))
    phases = list(info.phase_costs.values())
    assert all(b <= a for a, b in zip(phases, phases[1:]))
    assert res.extraction.cost == phases[-1]


def test_a_factorizing_message_is_sent_as_pieces():
    """Clique tree {a,x} - {a,b} - {b,y} | {a,b} - {a,b,c,d}: the message over {a,b} is
    (Σx f)(Σy g), and the receiver's factors h1(a,c), h2(b,d) never meet, so sending two pieces
    avoids the joint table over a, b, c, d."""
    rng = np.random.default_rng(0)
    cards = dict(a=4, b=4, c=4, d=4, x=4, y=4)
    fs = [Factor(0, ("a", "x"), rng.random((4, 4))), Factor(1, ("b", "y"), rng.random((4, 4))),
          Factor(2, ("a", "c"), rng.random((4, 4))), Factor(3, ("b", "d"), rng.random((4, 4)))]
    fg = FactorGraph(cards, fs)
    cliques = [frozenset("ax"), frozenset("ab"), frozenset("by"), frozenset("abcd")]
    jt = JunctionTree(cliques, {0: [1], 1: [0, 2, 3], 2: [1], 3: [1]}, {0: [0], 1: [], 2: [1], 3: [2, 3]})
    ps = PartitionSearch(fg, dict(strategy="staged", max_iters=30, node_limit=50_000, rules="full", time_limit_s=20),
                         dict(extractor="greedy", time_limit_s=20))
    st = initial_state(fg, jt).with_parts([{0, 1, 2}, {3}])
    joint = ps.evaluate(jt, st)
    info = PartitionInfo(1, [], {})
    st2, cur = ps.descend(jt, st, joint, ("pieces",), time.perf_counter() + 60, info)
    assert (1, 3) in st2.split and cur.cost < joint.cost
    tables, _ = evaluate(stitch(cur.problems, [s.extraction.dag for s in cur.solved]), fg, SUM_PRODUCT)
    ref = brute_force_marginals(fg)
    for v in fg.variables():
        np.testing.assert_allclose(tables[v].data / tables[v].data.sum(), ref[v], atol=1e-12)


def test_parallel_search_accepts_the_same_moves():
    fg = grid(3, 3, 2)
    kw = dict(extractor="greedy", partition="search", partition_time_s=120)
    one, two = optimize(fg, **kw), optimize(fg, partition_jobs=2, **kw)
    assert one.partition.accepted and one.partition.accepted == two.partition.accepted
    assert one.extraction.cost == two.extraction.cost

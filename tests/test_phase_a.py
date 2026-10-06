"""Phase A: rule sets, decomposition, seeding and greedy DAG extraction."""

import numpy as np
import pytest

from egfg.baselines import brute_force_marginals, junction_tree_dag
from egfg.cost import dag_cost
from egfg.decompose import clusters, local_problems
from egfg.egraph import saturate
from egfg.evaluate import SUM_PRODUCT, evaluate
from egfg.extract import extract_dag_greedy, extract_tree
from egfg.generators import chain, cycle, grid, random_small, random_sparse, random_tree, star
from egfg.ir import Dag, Input, Leaf, Mul, all_marginal_queries, dag_scopes, to_dag
from egfg.jtree import junction_tree
from egfg.pipeline import marginals, optimize
from helpers import chain3


def _matches_brute_force(fg, res) -> bool:
    bm = brute_force_marginals(fg)
    m = marginals(fg, res)
    return set(m) == set(fg.variables()) and all(np.allclose(m[v], bm[v]) for v in fg.variables())


def _acyclic(dag: Dag) -> bool:
    state: dict[str, int] = {}

    def go(nid: str) -> bool:
        if state.get(nid) == 1:
            return False
        if state.get(nid) == 2:
            return True
        state[nid] = 1
        ok = all(go(c) for c in dag.nodes[nid].children)
        state[nid] = 2
        return ok

    return all(go(r) for r in dag.roots.values())


# --- input leaves -------------------------------------------------------------


def test_input_leaf_scope_and_cost():
    fg = chain3()
    dag = to_dag({"q": Mul(Leaf(0), Input("m"))}, inputs={"m": frozenset({"b"})})
    assert dag_scopes(dag, fg)[dag.roots["q"]] == frozenset({"a", "b"})
    assert dag_cost(dag, fg) == 6  # the input leaf itself costs nothing


def test_evaluate_rejects_input_leaves():
    fg = chain3()
    dag = to_dag({"q": Mul(Leaf(0), Input("m"))}, inputs={"m": frozenset({"b"})})
    with pytest.raises(ValueError):
        evaluate(dag, fg, SUM_PRODUCT)


# --- rule sets ----------------------------------------------------------------


@pytest.mark.parametrize("rules", ["full", "no_reverse", "minimal"])
@pytest.mark.parametrize("seed", range(8))
def test_rule_sets_match_brute_force(rules, seed):
    # correctness of the rule sets; greedy extraction keeps the test fast (ILP is tested elsewhere)
    fg = random_small(seed)
    assert _matches_brute_force(fg, optimize(fg, rules=rules, extractor="greedy"))


def test_smaller_rule_sets_give_smaller_egraphs():
    fg = chain(5, 2)
    sizes = [saturate(fg, all_marginal_queries(fg), rules=r).num_nodes for r in ("full", "no_reverse", "minimal")]
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_unknown_rule_set_rejected():
    with pytest.raises(ValueError):
        saturate(chain3(), all_marginal_queries(chain3()), rules="nope")


# --- decomposition ------------------------------------------------------------

SMALL = [random_tree(8, 2, 3), cycle(7, 2, 1), grid(2, 4, 2, 2), grid(3, 3, 2, 5), random_sparse(8, 2, 1)]


@pytest.mark.parametrize("budget", [3, 5, 8])
@pytest.mark.parametrize("k", range(len(SMALL)))
def test_decomposition_matches_brute_force(budget, k):
    fg = SMALL[k]
    res = optimize(fg, cluster_budget=budget, extractor="tree")
    assert all(n.op != "input" for n in res.extraction.dag.nodes.values())
    assert _matches_brute_force(fg, res)


def test_budget_controls_cluster_count():
    fg = chain(10, 2)
    jt = junction_tree(fg)
    assert len(clusters(jt, None)) == 1
    assert len(clusters(jt, 2)) == len(jt.cliques)  # chain cliques have 2 variables: no merge fits
    assert len(clusters(jt, 3)) < len(jt.cliques)
    assert len(clusters(jt, 100)) == 1


def test_one_cluster_queries_are_phase1_queries():
    fg = grid(2, 3, 2)
    jt = junction_tree(fg)
    for parts in (clusters(jt, None), clusters(jt, 100)):
        (p,) = local_problems(fg, jt, parts)
        assert p.queries == all_marginal_queries(fg) and p.inputs == {}


def test_cluster_size_respects_budget():
    fg = grid(3, 6, 2)
    jt = junction_tree(fg)
    for budget in (3, 5, 8):
        for cl in clusters(jt, budget):
            size = len(frozenset().union(*(jt.cliques[i] for i in cl)))
            assert size <= budget or len(cl) == 1


# --- seeding ------------------------------------------------------------------


@pytest.mark.parametrize("extractor", ["ilp", "greedy"])
@pytest.mark.parametrize("budget", [None, 3, 5])
@pytest.mark.parametrize("fg", [star(7, 3), grid(2, 4, 3), random_tree(9, 3, 1)])
def test_seed_bounds_cost_by_junction_tree(fg, budget, extractor):
    jt_cost = dag_cost(junction_tree_dag(fg)[0], fg)
    res = optimize(fg, cluster_budget=budget, seed=True, extractor=extractor, node_limit=20)
    assert res.hit_limit
    assert res.extraction.cost <= jt_cost
    assert _matches_brute_force(fg, res)


def test_seed_is_unioned_with_query():
    fg = chain3()
    jt = junction_tree(fg)
    (p,) = local_problems(fg, jt, clusters(jt, None), seed=True)
    g = saturate(fg, p.queries, seeds=p.seeds, max_iters=0).graph
    # with no rewriting at all, each query's e-class already holds the seed's root node
    assert set(p.seeds) == set(p.queries)
    assert all(g.roots[q] in g.seed for q in p.queries)


# --- greedy DAG extraction ----------------------------------------------------


@pytest.mark.parametrize("fg", [chain(5, 2, 6), random_tree(6, 3, 5), star(6, 2), cycle(5, 2, 3)])
def test_greedy_acyclic_not_worse_than_tree_and_correct(fg):
    g = saturate(fg, all_marginal_queries(fg)).graph
    tree = extract_tree(g, fg)
    gr = extract_dag_greedy(g, fg)
    assert _acyclic(gr.dag)
    assert gr.cost <= tree.cost
    tables, _ = evaluate(gr.dag, fg, SUM_PRODUCT)
    bm = brute_force_marginals(fg)
    for v, t in tables.items():
        assert np.allclose(t.data / t.data.sum(), bm[v])


def test_greedy_improves_star():
    # tree extraction picks each marginal's cheapest form on its own; trading
    # some of that for shared subterms is cheaper overall
    fg = star(6, 2)
    g = saturate(fg, all_marginal_queries(fg)).graph
    assert extract_dag_greedy(g, fg).cost < extract_tree(g, fg).cost


# --- combined settings --------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(rules="minimal", cluster_budget=3, seed=True, extractor="greedy"),
        dict(rules="no_reverse", cluster_budget=5, seed=True, extractor="greedy"),
        dict(rules="no_reverse", cluster_budget=5, seed=True, extractor="ilp"),
    ],
)
@pytest.mark.parametrize("fg", [random_tree(8, 2, 0), grid(2, 4, 2, 1), random_sparse(8, 2, 0)])
def test_combined_settings_match_brute_force(fg, kw):
    assert _matches_brute_force(fg, optimize(fg, time_limit_s=5, **kw))


def test_random_sparse_shape():
    fg = random_sparse(16, 3, 0)
    assert len(fg.cards) == 16 and len(fg.factors) == 15 + 4
    assert len({frozenset(f.scope) for f in fg.factors}) == len(fg.factors)

from egfg.ir import Leaf, Mul, Sum, all_marginal_queries, dag_scopes, marginal_query, term_scope, to_dag
from helpers import chain3


def test_marginal_query_shape():
    q = marginal_query(chain3(), "b")
    assert q == Sum("a", Sum("c", Mul(Leaf(0), Leaf(1))))
    assert term_scope(q, chain3()) == frozenset({"b"})


def test_to_dag_shares_identical_subterms():
    dag = to_dag(all_marginal_queries(chain3()))
    assert set(dag.roots) == {"a", "b", "c"}
    assert sum(1 for n in dag.nodes.values() if n.op == "mul") == 1  # Mul(Leaf0, Leaf1) は 1 個
    scopes = dag_scopes(dag, chain3())
    assert scopes[dag.roots["c"]] == frozenset({"c"})


def test_to_dag_children_exist_and_leaf_args():
    dag = to_dag(all_marginal_queries(chain3()))
    leaves = sorted(n.arg for n in dag.nodes.values() if n.op == "leaf")
    assert leaves == [0, 1]
    assert all(c in dag.nodes for n in dag.nodes.values() for c in n.children)

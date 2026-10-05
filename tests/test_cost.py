from egfg.cost import dag_cost, max_intermediate_size, node_cost
from egfg.ir import ENode, all_marginal_queries, to_dag
from helpers import chain3


def test_naive_chain3_cost():
    fg = chain3()
    dag = to_dag(all_marginal_queries(fg))
    # Mul(f0,f1)=24、Σc(Mul)=24（p(a) と p(b) で共有）、p(a) の Σb=6、p(b) の Σa=6、
    # p(c) の Σb(Mul)=24 と Σa=|a||c|=8
    assert dag_cost(dag, fg) == 24 + 24 + 6 + 6 + 24 + 8
    assert max_intermediate_size(dag, fg) == 24


def test_node_cost_formula():
    fg = chain3()
    ab, bc, b = frozenset("ab"), frozenset("bc"), frozenset("b")
    assert node_cost(ENode("leaf", 0, ()), ab, [], fg) == 0
    assert node_cost(ENode("mul", None, ("x", "y")), frozenset("abc"), [ab, bc], fg) == 24
    assert node_cost(ENode("sum", "c", ("x",)), b, [bc], fg) == 12

from egfg.egraph import saturate
from egfg.generators import chain
from egfg.ir import all_marginal_queries, marginal_query
from helpers import chain3


def test_pushed_form_present():
    fg = chain3()
    res = saturate(fg, {"a": marginal_query(fg, "a")})
    root = res.graph.roots["a"]
    # Σb (f0 * Σc f1) が表現されている：root の「sum b」ノードの子 e-class に mul ノードがある
    g = res.graph
    assert any(
        n.op == "sum" and n.arg == "b" and any(m.op == "mul" for m in g.classes[n.children[0]])
        for n in g.classes[root]
    )
    # さらに、その mul の一方の子が「sum c」を含む（Σc f1 が独立した e-class になっている）
    assert any(
        m.op == "sum" and m.arg == "c"
        for nodes in g.classes.values()
        for m in nodes
    )
    assert not res.hit_limit


def test_node_limit_reported():
    fg = chain(6, 2)
    res = saturate(fg, all_marginal_queries(fg), node_limit=50)
    assert res.hit_limit


def test_children_reference_existing_classes():
    fg = chain3()
    g = saturate(fg, all_marginal_queries(fg)).graph
    assert all(c in g.classes for nodes in g.classes.values() for n in nodes for c in n.children)
    assert set(g.roots) == {"a", "b", "c"}
    # 3 つのクエリはスコープが違うので別々の e-class
    assert len(set(g.roots.values())) == 3


def test_leaves_present():
    fg = chain3()
    g = saturate(fg, all_marginal_queries(fg)).graph
    leaves = {n.arg for nodes in g.classes.values() for n in nodes if n.op == "leaf"}
    assert leaves == {0, 1}

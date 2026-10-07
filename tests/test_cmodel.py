import json
import re

import numpy as np

from egfg.baselines import brute_force_marginals, junction_tree_dag
from egfg.ccodegen import generate_c
from egfg.cmodel import dag_features
from egfg.generators import chain, cycle, grid, star
from egfg.pipeline import marginals, optimize


def test_nests_and_writes_match_the_generated_code():
    for fg, dag in [(grid(3, 3, 2), junction_tree_dag(grid(3, 3, 2))[0]),
                    (cycle(6, 3), optimize(cycle(6, 3), extractor="greedy").extraction.dag),
                    (star(6, 4), optimize(star(6, 4), extractor="greedy").extraction.dag),
                    (chain(5, 3), optimize(chain(5, 3), extractor="tree").extraction.dag)]:
        src = generate_c(dag, fg)
        body = src[src.index("void infer"):src.index("    double z")]
        f = dag_features(dag, fg)
        nmarg = len(fg.variables())
        assert f["nests"] - 1 - nmarg == len(re.findall(r"^    for \(", body, re.M))
        sizes = sum(int(n) for n in re.findall(r"^static double t\d+\[(\d+)\];", src, re.M))
        assert f["writes"] - sum(fg.cards.values()) == sizes


def test_without_a_calibration_cost_c_is_the_operation_count(monkeypatch, tmp_path):
    monkeypatch.setenv("EGFG_COST_CALIBRATION", str(tmp_path / "missing.json"))
    fg = cycle(6, 3)
    a, b = optimize(fg, extractor="greedy"), optimize(fg, extractor="greedy", cost="c")
    assert a.extraction.dag == b.extraction.dag and b.model_ns is None


def test_extraction_by_a_cost_model_is_exact(monkeypatch, tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"coef": {"flops_k": 0.5, "writes": 1.0, "strided": 0.3, "nests": 20.0,
                                      "loops": 0.2, "spill": 0.0, "const": 30.0}}))
    monkeypatch.setenv("EGFG_COST_CALIBRATION", str(p))
    fg = grid(2, 3, 3)
    for kw in ({}, {"partition": "search", "partition_time_s": 20, "partition_jobs": 1}):
        res = optimize(fg, extractor="greedy", cost="c", **kw)
        assert res.model_ns > 0
        ref, got = brute_force_marginals(fg), marginals(fg, res)
        for v in fg.variables():
            np.testing.assert_allclose(got[v], ref[v], atol=1e-12)

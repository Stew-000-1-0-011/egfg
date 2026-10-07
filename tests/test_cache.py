"""The opt-in cache: a second run reuses the saturation and the compiled program."""

import numpy as np

from egfg.ccodegen import compile_c_program, generate_c_for
from egfg.generators import chain
from egfg.pipeline import optimize


def test_cache_reuses_saturation_and_shared_library(tmp_path, monkeypatch):
    monkeypatch.setenv("EGFG_CACHE", str(tmp_path))
    fg = chain(5, 3)
    tables = {f.id: f.table for f in fg.factors}
    outs = []
    for _ in range(2):
        res = optimize(fg, extractor="greedy")
        run = compile_c_program(generate_c_for(fg, res), fg.variables(), fg.cards)
        outs.append((res.extraction.cost, run(tables)))
    assert len(list((tmp_path / "saturation").glob("*.pkl"))) == 1
    assert len(list((tmp_path / "so").glob("*.so"))) == 1
    (c1, o1), (c2, o2) = outs
    assert c1 == c2 and all(np.array_equal(o1[v], o2[v]) for v in fg.variables())

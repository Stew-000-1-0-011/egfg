"""Phase G: code generation and the arena."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from egfg.arena import (
    FAMILIES,
    draw_tables,
    egfg_program,
    holdout_problems,
    public_problems,
    reference_marginals,
    to_factor_graph,
)
from egfg.baselines import brute_force_marginals
from egfg.codegen import compile_program, generate_for
from egfg.generators import chain, cycle, grid, random_sparse, star
from egfg.pipeline import optimize
from egfg.structure import low_rank_tables

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fg", [chain(6, 3), star(6, 3), grid(2, 3, 3), random_sparse(8, 2, 1)])
def test_generated_code_matches_brute_force(fg):
    res = optimize(fg, extractor="greedy")
    infer = compile_program(generate_for(fg, res))
    out = infer({f.id: f.table for f in fg.factors})
    bm = brute_force_marginals(fg)
    assert set(out) == set(fg.variables()) and all(np.allclose(out[v], bm[v]) for v in fg.variables())


def test_generated_code_works_on_fresh_tables():
    # the program depends only on the structure: new tables give the new marginals
    fg = cycle(5, 3)
    infer = compile_program(generate_for(fg, optimize(fg, extractor="greedy")))
    fresh = cycle(5, 3, seed=9)
    out = infer({f.id: f.table for f in fresh.factors})
    bm = brute_force_marginals(fresh)
    assert all(np.allclose(out[v], bm[v]) for v in fresh.variables())


def test_generated_low_rank_code_factorizes_at_runtime():
    fg = low_rank_tables(cycle(6, 6), 2)
    res = optimize(fg, extractor="greedy", structure=("lowrank",))
    src = generate_for(fg, res)
    assert "svd" in src
    fresh = low_rank_tables(cycle(6, 6), 2, seed=4)
    out = compile_program(src)({f.id: f.table for f in fresh.factors})
    bm = brute_force_marginals(fresh)
    assert all(np.allclose(out[v], bm[v]) for v in fresh.variables())


def test_public_and_holdout_problems():
    pub = public_problems()
    assert {p["family"] for p in pub} == set(FAMILIES) and len(pub) == 2 * len(FAMILIES)
    hold = holdout_problems(5)
    assert not {p["name"] for p in hold} & {p["name"] for p in pub}
    assert holdout_problems(5) == hold  # deterministic in the seed


@pytest.mark.parametrize("k", range(0, 16, 3))
def test_egfg_programs_are_correct_on_public_problems(k):
    p = public_problems()[k]
    src, info = egfg_program(p)
    tables = draw_tables(p, 123)
    out = compile_program(src)(tables)
    ref = reference_marginals(p, tables)
    assert all(np.allclose(out[v], ref[v], atol=1e-8) for v in ref)
    assert info["flops"] > 0


def test_draw_tables_low_rank():
    p = next(q for q in public_problems() if q["family"] == "lowrank")
    t = draw_tables(p, 0)
    r = p["factors"][0]["table"]["rank"]
    assert np.linalg.matrix_rank(t[0]) == r


def test_score_rejects_forbidden_imports_and_scores_programs(tmp_path):
    p = public_problems()[2]
    probs = tmp_path / "problems"
    probs.mkdir()
    (probs / f"{p['name']}.json").write_text(json.dumps(p))
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    src, _ = egfg_program(p)
    (good / f"{p['name']}.py").write_text(src)
    (bad / f"{p['name']}.py").write_text("import egfg\n" + src)
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(ROOT / "arena" / "score.py"), "--solutions", str(good), str(bad),
                    "--problems", str(probs), "--out", str(out), "--seed", "3"], check=True, cwd=ROOT)
    res = json.loads(out.read_text())["problems"][p["name"]]
    assert res["good"]["status"] == "ok" and res["bad"]["status"] == "rejected"


# --- C ------------------------------------------------------------------------

from egfg.arena import check_c_source, egfg_c_program  # noqa: E402
from egfg.ccodegen import compile_c_program, generate_c_for  # noqa: E402


@pytest.mark.parametrize("fg", [chain(6, 3), star(6, 3), grid(2, 3, 3), random_sparse(8, 2, 1), cycle(5, 4)])
def test_generated_c_matches_brute_force(fg):
    res = optimize(fg, extractor="greedy")
    run = compile_c_program(generate_c_for(fg, res), fg.variables(), fg.cards)
    out = run({f.id: f.table for f in fg.factors})
    bm = brute_force_marginals(fg)
    assert all(np.allclose(out[v], bm[v]) for v in fg.variables())


def test_generated_c_on_fresh_tables_and_ternary():
    p = next(q for q in public_problems() if q["family"] == "ternary")
    src, info = egfg_c_program(p, overheads=(0,))
    run = compile_c_program(src, list(p["variables"]), p["variables"])
    tables = draw_tables(p, 77)
    out = run(tables)
    ref = reference_marginals(p, tables)
    assert all(np.allclose(out[v], ref[v], atol=1e-10) for v in ref)


def test_c_source_check():
    assert check_c_source("#include <math.h>\nvoid infer(){}") is None
    assert check_c_source("#include <stdio.h>\n") == "includes stdio.h"
    assert check_c_source("#pragma omp parallel\n") == "uses threads"


def test_c_sum_chains_over_shared_nodes():
    # every marginal is a chain of two sums over the same shared product: each chain must be
    # folded into one reduction of that product (all marginals come from one joint, as required)
    from egfg.ccodegen import generate_c
    from egfg.ir import Leaf, Mul, Sum, to_dag
    from egfg.model import Factor, FactorGraph

    rng = np.random.default_rng(0)
    t0, t1 = rng.uniform(0.1, 1, (2, 3, 2)), rng.uniform(0.1, 1, 3)
    fg = FactorGraph({"a": 2, "b": 3, "c": 2}, [Factor(0, ("a", "b", "c"), t0), Factor(1, ("b",), t1)])
    x = Mul(Leaf(0), Leaf(1))
    terms = {"a": Sum("b", Sum("c", x)), "b": Sum("a", Sum("c", x)), "c": Sum("a", Sum("b", x))}
    run = compile_c_program(generate_c(to_dag(terms), fg), ["a", "b", "c"], fg.cards)
    out = run({0: t0, 1: t1})
    joint = t0 * t1[None, :, None]
    for v, ax in (("a", (1, 2)), ("b", (0, 2)), ("c", (0, 1))):
        assert np.allclose(out[v], joint.sum(ax) / joint.sum())


def test_c_low_rank_holdout_chain():
    p = next(q for q in holdout_problems(7) if q["name"].startswith("holdout_lowrank_chain"))
    src, _ = egfg_c_program(p, overheads=(0,))
    run = compile_c_program(src, list(p["variables"]), p["variables"])
    tables = draw_tables(p, 3)
    out, ref = run(tables), reference_marginals(p, tables)
    assert all(np.allclose(out[v], ref[v], atol=1e-9) for v in ref)


@pytest.mark.parametrize("make,kw", [(lambda: cycle(6, 3), {}), (lambda: grid(2, 3, 3), {}), (lambda: star(6, 3), {}),
                                     (lambda: chain(20, 3), {"cluster_budget": 6}),
                                     (lambda: low_rank_tables(cycle(6, 6), 2), {"structure": ("lowrank",)})])
def test_compact_c_matches_plain_c(make, kw):
    from egfg.ccodegen import compile_c_program, generate_c_for

    fg = make()
    res = optimize(fg, extractor="greedy", **kw)
    tables = {f.id: f.table for f in fg.factors}
    plain = compile_c_program(generate_c_for(fg, res), fg.variables(), fg.cards)(tables)
    compact = compile_c_program(generate_c_for(fg, res, compact=True), fg.variables(), fg.cards)(tables)
    assert all(np.allclose(plain[v], compact[v], rtol=1e-12, atol=0) for v in fg.variables())


def test_compact_c_rolls_a_chain_into_loops():
    from egfg.ccodegen import generate_c_for

    lines = []
    for n in (24, 48):
        fg = chain(n, 3)
        lines.append(generate_c_for(fg, optimize(fg, extractor="greedy", cluster_budget=6), compact=True).count("\n"))
    assert lines[1] < 1.3 * lines[0]  # twice the chain, about the same code

"""egfg against existing tools on the same discrete problems (all marginals).

Usage:
  uv run --group libs python arena/compare_existing.py --seed 401 --out arena/results/existing_seed401.json

Two comparisons:
- plans: the contraction plans of opt_einsum ('dp') and cotengra (hyper-optimized, flops), one
  per marginal, are turned into egfg terms (identical subterms shared, as with opt_einsum's
  shared_intermediates) and go through egfg's cost model and C code generation, like egfg's own
  plan and the best junction tree. Only the plan differs.
- numpy: each tool as it is used from Python: egfg's generated numpy program, opt_einsum
  (precompiled expressions, shared intermediates), and Funsor (numpy backend: sum_product built lazily, then its optimizer).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egfg import arena  # noqa: E402
from egfg.baselines import best_junction_tree  # noqa: E402
from egfg.ccodegen import compile_c_program, generate_c, generate_c_for  # noqa: E402
from egfg.codegen import compile_program  # noqa: E402
from egfg.cost import dag_cost  # noqa: E402
from egfg.ir import Leaf, Mul, Sum, to_dag  # noqa: E402
from egfg.pipeline import optimize  # noqa: E402

TOL = 1e-6


# ---------------------------------------------------------------------------
# contraction paths -> egfg terms
# ---------------------------------------------------------------------------


def _mul(a, b):
    """Mul with its operands in a canonical order, so that equal subterms are shared."""
    return Mul(a, b) if repr(a) <= repr(b) else Mul(b, a)


def _sum_out(t, scope, keep):
    for x in sorted(scope - keep, reverse=True):
        t = Sum(x, t)
    return t, scope & keep


def path_term(fg, v: str, path):
    """The term of a linear contraction path (opt_einsum's format) for the marginal of v."""
    ops = [(Leaf(f.id), frozenset(f.scope)) for f in fg.factors]

    def needed(skip):
        out = {v}
        for k, (_, sc) in enumerate(ops):
            if k not in skip:
                out |= sc
        return out

    ops = [_sum_out(t, sc, needed({k})) for k, (t, sc) in enumerate(ops)]
    for step in path:  # a step contracts one or more operands
        picked = [ops[k] for k in step]
        for k in sorted(step, reverse=True):
            ops.pop(k)
        t, sc = picked[0]
        for tb, sb in picked[1:]:
            t, sc = _mul(t, tb), sc | sb
        keep = needed(set())  # the output and every remaining operand
        ops.append(_sum_out(t, sc, keep))
    (t, sc), = ops
    return _sum_out(t, sc, {v})[0]


def _einsum_spec(fg, v):
    sym = {x: k for k, x in enumerate(fg.variables())}
    inputs = [tuple(f.scope) for f in fg.factors]
    return inputs, sym


def oe_paths(fg) -> dict:
    import opt_einsum as oe

    inputs, sym = _einsum_spec(fg, None)
    s = lambda vs: "".join(oe.get_symbol(sym[x]) for x in vs)  # noqa: E731
    shapes = [tuple(fg.cards[x] for x in sc) for sc in inputs]
    out = {}
    for v in fg.variables():
        eq = ",".join(s(sc) for sc in inputs) + "->" + s((v,))
        out[v] = oe.contract_path(eq, *shapes, shapes=True, optimize="dp")[0]
    return out


def ctg_paths(fg) -> dict:
    import cotengra as ctg

    inputs, _ = _einsum_spec(fg, None)
    out = {}
    for v in fg.variables():
        opt = ctg.HyperOptimizer(minimize="flops", max_repeats=32, parallel=False, progbar=False)
        tree = ctg.array_contract_tree(inputs=inputs, output=(v,), size_dict=dict(fg.cards), optimize=opt)
        out[v] = tree.get_path()
    return out


def plan_dag(fg, paths):
    return to_dag({v: path_term(fg, v, paths[v]) for v in fg.variables()})


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_c(problem, src, seeds) -> dict:
    run = compile_c_program(src, list(problem["variables"]), problem["variables"])
    err, t = 0.0, None
    for k, seed in enumerate(seeds):
        tables = arena.draw_tables(problem, seed)
        out = run(tables)
        if k == 0:
            t = arena.time_c(run, tables)
        for v, r in arena.reference_marginals(problem, tables).items():
            err = max(err, float(np.max(np.abs(out[v] - r))))
    return {"status": "ok" if err <= TOL else "wrong", "max_abs_err": err, "time_s": t}


def _median_time(f, repeats=7, min_seconds=0.005) -> float:
    f()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        if time.perf_counter() - t0 >= min_seconds:
            break
        n *= 4
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        samples.append((time.perf_counter() - t0) / n)
    return statistics.median(samples)


def score_py(problem, make, seeds) -> dict:
    """make(tables) -> infer() returning {variable: normalized marginal}."""
    err, t = 0.0, None
    for k, seed in enumerate(seeds):
        tables = arena.draw_tables(problem, seed)
        infer = make(tables)
        out = infer()
        if k == 0:
            t = _median_time(infer)
        for v, r in arena.reference_marginals(problem, tables).items():
            err = max(err, float(np.max(np.abs(np.asarray(out[v]) - r))))
    return {"status": "ok" if err <= TOL else "wrong", "max_abs_err": err, "time_s": t}


def egfg_numpy(problem):
    src, _ = arena.egfg_program(problem)
    infer = compile_program(src)
    return lambda tables: (lambda: infer({i: t for i, t in tables.items()}))


def oe_numpy(problem, fg, paths):
    import opt_einsum as oe

    inputs, sym = _einsum_spec(fg, None)
    s = lambda vs: "".join(oe.get_symbol(sym[x]) for x in vs)  # noqa: E731
    shapes = [tuple(fg.cards[x] for x in sc) for sc in inputs]
    exprs = {v: oe.contract_expression(",".join(s(sc) for sc in inputs) + "->" + s((v,)), *shapes,
                                       optimize=paths[v]) for v in fg.variables()}
    ids = [f.id for f in fg.factors]

    def make(tables):
        arrays = [tables[i] for i in ids]

        def infer():
            out = {}
            with oe.shared_intermediates():
                for v, e in exprs.items():
                    m = e(*arrays)
                    out[v] = m / m.sum()
            return out

        return infer

    return make


def funsor_numpy(problem, fg):
    from collections import OrderedDict

    import funsor
    from funsor import interpretations
    from funsor.domains import Bint
    from funsor.optimizer import apply_optimizer
    from funsor.sum_product import sum_product

    funsor.set_backend("numpy")
    ids = [f.id for f in fg.factors]
    scopes = {f.id: tuple(f.scope) for f in fg.factors}
    names = fg.variables()

    def make(tables):
        factors = [funsor.Tensor(np.asarray(tables[i]), OrderedDict((x, Bint[fg.cards[x]]) for x in scopes[i]))
                   for i in ids]

        def infer():
            # built lazily, then contracted by Funsor's optimizer (opt_einsum); memoize shares
            # equal subexpressions across the marginals
            out = {}
            with interpretations.memoize():
                for v in names:
                    with interpretations.lazy:
                        e = sum_product(funsor.ops.add, funsor.ops.mul, factors, frozenset(names) - {v}, frozenset())
                    d = np.asarray(apply_optimizer(e).data)
                    out[v] = d / d.sum()
            return out

        return infer

    return make


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*", help="problem names (default: public and larger)")
    args = ap.parse_args()
    seeds = [args.seed + k for k in range(3)]
    problems = [arena.load(p) for p in sorted((ROOT / "arena" / "problems").glob("*.json"))] + arena.larger_problems()
    if args.only:
        problems = [p for p in problems if p["name"] in args.only]
    results = {"seeds": seeds, "problems": {}}
    for problem in problems:
        fg = arena.to_factor_graph(problem, arena.draw_tables(problem, 0))
        row = {"plans": {}, "numpy": {}}
        t0 = time.perf_counter()
        res = optimize(fg, extractor="greedy")
        egfg_s = time.perf_counter() - t0
        plans = {"egfg": (res.extraction.dag, egfg_s, generate_c_for(fg, res))}
        t0 = time.perf_counter()
        jt = best_junction_tree(fg, share_products=True)[0]
        plans["best_jt+"] = (jt, time.perf_counter() - t0, None)
        t0 = time.perf_counter()
        oep = oe_paths(fg)
        plans["opt_einsum_dp"] = (plan_dag(fg, oep), time.perf_counter() - t0, None)
        t0 = time.perf_counter()
        plans["cotengra"] = (plan_dag(fg, ctg_paths(fg)), time.perf_counter() - t0, None)
        for name, (dag, secs, src) in plans.items():
            r = score_c(problem, src or generate_c(dag, fg), seeds)
            r.update(flops=dag_cost(dag, fg), plan_s=round(secs, 3))
            row["plans"][name] = r
        src, info = arena.egfg_c_program(problem)
        r = score_c(problem, src, seeds)
        r.update(flops=info["flops"], plan_s=info["compile_s"], note="autotuned; may use low-rank splits")
        row["plans"]["egfg_tuned"] = r
        for name, make in (("egfg_numpy", egfg_numpy(problem)), ("opt_einsum", oe_numpy(problem, fg, oep)),
                           ("funsor", funsor_numpy(problem, fg))):
            row["numpy"][name] = score_py(problem, make, seeds)
        results["problems"][problem["name"]] = row
        print(problem["name"],
              {k: (v["status"], v["flops"], round(v["time_s"] * 1e9)) for k, v in row["plans"].items()}, "(flops, ns)",
              {k: (v["status"], round(v["time_s"] * 1e6, 1)) for k, v in row["numpy"].items()}, "(us)", flush=True)
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))  # noqa: E731
    rows = list(results["problems"].values())
    summary = {}
    for k in ("best_jt+", "opt_einsum_dp", "cotengra", "egfg_tuned"):
        summary[f"flops {k}/egfg"] = gm([r["plans"][k]["flops"] / r["plans"]["egfg"]["flops"] for r in rows])
        summary[f"time {k}/egfg"] = gm([r["plans"][k]["time_s"] / r["plans"]["egfg"]["time_s"] for r in rows])
    for k in ("opt_einsum", "funsor"):
        summary[f"numpy {k}/egfg_numpy"] = gm([r["numpy"][k]["time_s"] / r["numpy"]["egfg_numpy"]["time_s"] for r in rows])
    results["summary"] = summary
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

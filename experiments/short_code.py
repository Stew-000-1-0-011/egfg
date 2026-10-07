"""Phase J experiment: the compact C code (kernels, rolled loops) against the plain one.

For each problem the same extraction is written both ways and compared on lines, bytes, gcc
time, run time (as in the arena), correctness on fresh tables, kernels and the share of calls
inside loops; then the shape penalty α of the extraction is varied for the compact code.
Runs sequentially (the times are measured). The cache must be off (EGFG_CACHE unset).

Usage: uv run python experiments/short_code.py --out results/short_code.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egfg import arena  # noqa: E402
from egfg.ccodegen import compile_c_program, generate_c_for  # noqa: E402
from egfg.generators import chain, grid  # noqa: E402
from egfg.pipeline import optimize  # noqa: E402

ALPHAS = (0.0, 10.0, 100.0)


def problems():
    out = [(p["name"], p, {}) for p in (arena.load(f) for f in sorted((ROOT / "arena" / "problems").glob("*.json")))]
    out += [(p["name"], p, {}) for p in arena.larger_problems()]
    for name, fg, budget in (("chain64_k4", chain(64, 4), 8), ("chain128_k4", chain(128, 4), 8), ("grid6x6_k2", grid(6, 6, 2), 8)):
        out.append((name, fg, {"cluster_budget": budget}))
    return out


def measure(fg, res, compact, tables_list):
    src = generate_c_for(fg, res, compact=compact)
    t0 = time.perf_counter()
    run = compile_c_program(src, fg.variables(), fg.cards)
    gcc = time.perf_counter() - t0
    err = 0.0
    for tables, ref in tables_list:
        out = run(tables)
        err = max(err, max(float(np.max(np.abs(out[v] - ref[v]))) for v in ref))
    body = src[src.index("void infer"):]
    calls = len(re.findall(r"\bk\d+\(", body))
    looped = len(re.findall(r"for \(int r = 0; r < \d+; r\+\+\) k\d+\(", body))
    return {"lines": src.count("\n"), "bytes": len(src), "gcc_s": round(gcc, 3),
            "run_ns": round(arena.time_c(run, tables_list[0][0]) * 1e9, 1), "max_abs_err": err,
            "kernels": len(re.findall(r"^static void k\d+\(", src, re.M)), "call_lines": calls, "loop_lines": looped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/short_code.csv")
    args = ap.parse_args()
    if os.environ.get("EGFG_CACHE"):
        raise SystemExit("unset EGFG_CACHE: gcc times are measured")
    rows = []
    for name, prob, kw in problems():
        if isinstance(prob, dict):
            fg = arena.to_factor_graph(prob, arena.draw_tables(prob, 0))
            if any(f["table"]["kind"] == "lowrank" for f in prob["factors"]):
                kw = {"structure": ("lowrank",)}
            tables_list = []
            for seed in (401, 402):
                tables = arena.draw_tables(prob, seed)
                tables_list.append((tables, arena.reference_marginals(prob, tables)))
        else:
            fg = prob
            tables_list = [({f.id: f.table for f in fg.factors}, None)]
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            res = optimize(fg, extractor="greedy", shape_penalty=alpha, **kw)
            opt_s = time.perf_counter() - t0
            if tables_list[0][1] is None:  # no arena reference: the plain program of α = 0 is the reference
                ref_run = compile_c_program(generate_c_for(fg, res), fg.variables(), fg.cards)
                tables_list = [(tables_list[0][0], ref_run(tables_list[0][0]))]
            for compact in ((False, True) if alpha == 0 else (True,)):
                m = measure(fg, res, compact, tables_list)
                row = {"problem": name, "vars": len(fg.variables()), "code": "compact" if compact else "plain",
                       "alpha": alpha, "flops": res.extraction.cost, "optimize_s": round(opt_s, 2), **m}
                rows.append(row)
                print(row, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

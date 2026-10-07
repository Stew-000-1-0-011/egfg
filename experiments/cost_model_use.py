"""Phase M: does extracting by the calibrated C cost model (cost="c") give faster C than the
operation count (cost="flops")?

Problems: phase K's 11 and the arena's public and larger ones (used in the calibration), and the
arena's holdout problems (not used). Small problems (<= 12 variables) are solved without a
partition, the others with the partition search. Optimizations run in parallel (one per
process); the C run times are measured one after another.

Usage: uv run python experiments/cost_model_use.py --out results/cost_model_use.csv [--jobs N] [--time 120]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from egfg import arena, cmodel  # noqa: E402
from egfg.ccodegen import compile_c_program, generate_c  # noqa: E402
from egfg.pipeline import optimize  # noqa: E402

SMALL = 12


def problems():
    from cost_calibration import problems as calibrated

    seen = {name for name, _ in calibrated()}
    out = [(name, fg, "calibrated") for name, fg in calibrated()]
    for p in arena.holdout_problems(7):
        if any(f["table"]["kind"] == "lowrank" for f in p["factors"]) or p["name"] in seen:
            continue
        out.append((p["name"], arena.to_factor_graph(p, arena.draw_tables(p, 0)), "holdout"))
    return out


def run_task(args) -> dict:
    pi, cost, limit = args
    name, fg, group = problems()[pi]
    kw = dict(extractor="greedy", time_limit_s=60, cost=cost)
    if len(fg.variables()) > SMALL:
        kw.update(partition="search", partition_time_s=limit, partition_jobs=1, boundary=False)
    t0 = time.perf_counter()
    res = optimize(fg, **kw)
    model = cmodel.load()
    return {"pi": pi, "dag": res.extraction.dag,
            "row": {"problem": name, "group": group, "vars": len(fg.variables()), "cost": cost,
                    "method": "search" if "partition" in kw else "none", "flops": res.extraction.cost,
                    "model_ns": round(model.dag_ns(res.extraction.dag, fg), 2),
                    "optimize_s": round(time.perf_counter() - t0, 2)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cost_model_use.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--time", type=float, default=120)
    args = ap.parse_args()
    if os.environ.get("EGFG_CACHE"):
        raise SystemExit("unset EGFG_CACHE: times are measured")
    if cmodel.load() is None:
        raise SystemExit("no calibration: run experiments/cost_calibration.py first")
    probs = problems()
    tasks = [(pi, c, args.time) for pi in range(len(probs)) for c in ("flops", "c")]
    tasks.sort(key=lambda t: -len(probs[t[0]][1].variables()))
    with ProcessPoolExecutor(args.jobs, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(run_task, tasks))

    def build(r):
        fg = probs[r["pi"]][1]
        return compile_c_program(generate_c(r["dag"], fg), fg.variables(), fg.cards)

    with ThreadPoolExecutor(args.jobs) as tp:
        runs = list(tp.map(build, results))
    rows = []
    for r in results:
        r["row"]["run_ns"] = float("inf")
    for _ in range(2):  # two passes, the faster of the two (drifts of the machine hit both settings)
        for r, run in zip(results, runs):
            fg = probs[r["pi"]][1]
            ns = arena.time_c(run, {f.id: f.table for f in fg.factors}) * 1e9
            r["row"]["run_ns"] = round(min(r["row"]["run_ns"], ns), 2)
    for r in results:
        rows.append(r["row"])
        print(r["row"], flush=True)
    rows.sort(key=lambda r: (r["group"], r["problem"], r["cost"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

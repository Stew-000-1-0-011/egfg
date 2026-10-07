"""Phase K experiment: the partition chosen by measured cost against fixed budgets.

Per problem: the junction tree, fixed cluster budgets 4-8 (the phase A partition), the searched
partition after merges only, after exchanges, and after the boundary choices, and (for small
problems) no partition at all. Optimizations run in parallel (one per process, every search
from scratch); the C run times are then measured one after another.

Usage: uv run python experiments/partition_search.py --out results/partition_search.csv [--jobs N] [--time 300]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egfg import arena  # noqa: E402
from egfg.baselines import junction_tree_dag  # noqa: E402
from egfg.ccodegen import compile_c_program, generate_c  # noqa: E402
from egfg.cost import dag_cost  # noqa: E402
from egfg.dynamic import unroll  # noqa: E402
from egfg.generators import chain, coupled_hmm, cycle, factorial_hmm, grid, random_observations, random_sparse  # noqa: E402
from egfg.pipeline import optimize  # noqa: E402

BUDGETS = (4, 5, 6, 7, 8)
SMALL = 12  # variables: also solved without a partition


def problems():
    out = [("grid3x3_k3", grid(3, 3, 3)), ("cycle10_k4", cycle(10, 4)), ("sparse10_k3", random_sparse(10, 3)),
           ("grid4x4_k3", grid(4, 4, 3)), ("grid5x5_k2", grid(5, 5, 2)), ("grid6x6_k2", grid(6, 6, 2)),
           ("cycle16_k16", cycle(16, 16)), ("sparse20_k3", random_sparse(20, 3))]
    for name, m in (("coupled3_k3_t6", coupled_hmm(3, 3)), ("factorial3_k3_t6", factorial_hmm(3, 3))):
        out.append((name, unroll(m, random_observations(m, 6))))
    out.append(("chain64_k4", chain(64, 4)))
    return out


def run_task(args) -> dict:
    pi, method, limit = args
    name, fg = problems()[pi]
    t0 = time.perf_counter()
    info = None
    kw = dict(extractor="greedy", time_limit_s=60)
    if method == "jt":
        dag = junction_tree_dag(fg)[0]
        res = None
    elif method.startswith("budget"):
        res = optimize(fg, cluster_budget=int(method[6:]), **kw)
    elif method == "none":
        res = optimize(fg, **kw)
    else:
        res = optimize(fg, partition="search", partition_time_s=limit, exchange=method != "merge",
                       boundary=method == "boundary", **kw)
        info = res.partition
    secs = time.perf_counter() - t0
    if res is not None:
        dag = res.extraction.dag
    row = {"problem": name, "vars": len(fg.variables()), "method": method, "flops": dag_cost(dag, fg),
           "optimize_s": round(secs, 2), "clusters": len(res.clusters) if res else "",
           "max_cluster_vars": max(len(c) for c in res.clusters) if res else "",
           "hit_limit": res.hit_limit if res else "", "max_nodes": res.max_cluster_nodes if res else ""}
    if info is not None:
        kinds = [k for k, _ in info.accepted]
        row.update({"trees": info.trees, "tries": info.tries, "solves": info.solves,
                    "phases": " ".join(f"{k}={v}" for k, v in info.phase_costs.items()),
                    "accepted": " ".join(f"{k}:{kinds.count(k)}" for k in dict.fromkeys(kinds)),
                    "split_msgs": len(info.state.split)})
    return {"row": row, "dag": dag, "pi": pi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/partition_search.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--time", type=float, default=300, help="time limit of one partition search (s)")
    args = ap.parse_args()
    if os.environ.get("EGFG_CACHE"):
        raise SystemExit("unset EGFG_CACHE: times are measured")
    probs = problems()
    tasks = []
    for pi, (_, fg) in enumerate(probs):
        methods = ["jt", *(f"budget{b}" for b in BUDGETS), "merge", "exchange", "boundary"]
        if len(fg.variables()) <= SMALL:
            methods.append("none")
        tasks += [(pi, m, args.time) for m in methods]
    order = {"boundary": 0, "exchange": 1, "merge": 2, "none": 3}  # the slowest first
    tasks.sort(key=lambda t: (order.get(t[1], 4), -len(probs[t[0]][1].variables())))
    results = []
    with ProcessPoolExecutor(args.jobs, mp_context=get_context("spawn")) as pool:
        for r in pool.map(run_task, tasks):
            print(r["row"], flush=True)
            results.append(r)
    rows = []
    for r in sorted(results, key=lambda r: (r["pi"], r["row"]["method"])):  # C run times, one at a time
        fg = probs[r["pi"]][1]
        try:
            run = compile_c_program(generate_c(r["dag"], fg), fg.variables(), fg.cards)
            r["row"]["run_ns"] = round(arena.time_c(run, {f.id: f.table for f in fg.factors}) * 1e9, 1)
        except Exception as e:  # noqa: BLE001
            print("C failed", r["row"]["problem"], r["row"]["method"], e, flush=True)
            r["row"]["run_ns"] = ""
        rows.append(r["row"])
    fields = list(dict.fromkeys(k for row in rows for k in row))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

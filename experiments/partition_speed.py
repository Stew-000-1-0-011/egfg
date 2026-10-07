"""Phase L experiment: a faster partition search (narrowing the trees, parallel candidates, fewer moves).

Per problem (the 11 of phase K), one after another so that the times are comparable; the
parallelism is only inside a search (4 processes):

- phaseK: every tree to the end, no parallelism, phase K's moves
- halving: successive halving of the trees
- halving_par: + 4 processes
- halving_par_moves: + the reduced moves (no split, factors of crossing separators only)
- first_par_moves: the min-fill tree only, 4 processes, reduced moves
- budgets: the fixed budgets 4-8 one after another (best cost, summed time)

All searches stop after the exchanges (no boundary phase).

Usage: uv run python experiments/partition_speed.py --out results/partition_speed.csv [--time 300]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egfg.partition import EXCHANGE, PHASE_K_EXCHANGE  # noqa: E402
from egfg.pipeline import optimize  # noqa: E402
from partition_search import BUDGETS, problems  # noqa: E402

CONFIGS = {
    "phaseK": dict(partition_trees="all", partition_jobs=1, partition_moves=PHASE_K_EXCHANGE),
    "halving": dict(partition_trees="halving", partition_jobs=1, partition_moves=PHASE_K_EXCHANGE),
    "halving_par": dict(partition_trees="halving", partition_jobs=4, partition_moves=PHASE_K_EXCHANGE),
    "halving_par_moves": dict(partition_trees="halving", partition_jobs=4, partition_moves=EXCHANGE),
    "first_par_moves": dict(partition_trees="first", partition_jobs=4, partition_moves=EXCHANGE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/partition_speed.csv")
    ap.add_argument("--time", type=float, default=300, help="time limit of one partition search (s)")
    args = ap.parse_args()
    if os.environ.get("EGFG_CACHE"):
        raise SystemExit("unset EGFG_CACHE: times are measured")
    rows = []
    for name, fg in problems():
        kw = dict(extractor="greedy", time_limit_s=60)
        for cfg, ckw in CONFIGS.items():
            t0 = time.perf_counter()
            res = optimize(fg, partition="search", partition_time_s=args.time, boundary=False, **kw, **ckw)
            secs = time.perf_counter() - t0
            info = res.partition
            row = {"problem": name, "vars": len(fg.variables()), "config": cfg, "flops": res.extraction.cost,
                   "optimize_s": round(secs, 2), "search_s": round(info.seconds, 2), "trees": info.trees,
                   "tree_costs": " ".join(map(str, info.tree_costs)),
                   "phases": " ".join(f"{k}={v}" for k, v in info.phase_costs.items()),
                   "tries": info.tries, "solves": info.solves,
                   "accepted": " ".join(f"{k}:{c}" for k, c in info.accepted), "clusters": len(res.clusters)}
            rows.append(row)
            print(row, flush=True)
        t0, best = time.perf_counter(), None
        for b in BUDGETS:
            r = optimize(fg, cluster_budget=b, **kw)
            best = r.extraction.cost if best is None else min(best, r.extraction.cost)
        row = {"problem": name, "vars": len(fg.variables()), "config": "budgets", "flops": best,
               "optimize_s": round(time.perf_counter() - t0, 2)}
        rows.append(row)
        print(row, flush=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

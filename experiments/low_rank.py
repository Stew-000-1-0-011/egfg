"""Phase F: low-rank factorization equalities vs the plain search and known-trick baselines.

Usage: uv run python experiments/low_rank.py --out results/low_rank.csv [--quick] [--jobs 3]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COLUMNS = (
    "family,n,K,rank,splits,plain_cost,lowrank_cost,best_jtp,best_jtp_replaced,bp,bp_replaced,"
    "lowrank_vs_plain,lowrank_vs_best_known,uses_split,plain_s,lowrank_s,correct,max_abs_err"
).split(",")


def cases(quick: bool):
    fams = ["chain", "star", "cycle"] if quick else ["chain", "star", "random_tree", "cycle", "grid"]
    for K in ([6] if quick else [8, 16]):
        n = 6 if K >= 16 or quick else 8
        for fam in fams:
            for r in ([1, 2] if quick else [1, 2, 4, K]):
                yield fam, n, K, r


def run(case) -> dict:
    import numpy as np

    from egfg import generators as g
    from egfg.baselines import best_junction_tree, brute_force_marginals, factor_graph_bp_dag
    from egfg.cost import dag_cost
    from egfg.pipeline import marginals, optimize
    from egfg.structure import low_rank, low_rank_tables

    fam, n, K, r = case
    base = g.grid(2, n // 2, K) if fam == "grid" else {"chain": g.chain, "star": g.star, "random_tree": g.random_tree,
                                                       "cycle": g.cycle}[fam](n, K)
    fg = low_rank_tables(base, r) if r < K else base
    t0 = time.perf_counter()
    plain = optimize(fg, extractor="greedy")
    t1 = time.perf_counter()
    lr = optimize(fg, extractor="greedy", structure=("lowrank",))
    t2 = time.perf_counter()
    st = low_rank(fg)
    bjp = best_junction_tree(fg, share_products=True)[1]
    bjr = best_junction_tree(st.replaced, share_products=True)[1] if st.replaced else ""
    tree = fam in ("chain", "star", "random_tree")
    bp = dag_cost(factor_graph_bp_dag(fg)[0], fg) if tree else ""
    bpr = dag_cost(factor_graph_bp_dag(st.replaced)[0], st.replaced) if tree and st.replaced else ""
    known = min(x for x in (bjp, bjr, bp, bpr) if x != "")
    bm = brute_force_marginals(fg)
    m = marginals(fg, lr)
    err = max(float(np.max(np.abs(m[v] - bm[v]))) for v in fg.variables())
    leaves = {nd.arg for nd in lr.extraction.dag.nodes.values() if nd.op == "leaf"}
    uses = any(sp.u in leaves for sp in st.splits)
    c = lr.extraction.cost
    return {"family": fam, "n": n, "K": K, "rank": r, "splits": len(st.splits), "plain_cost": plain.extraction.cost,
            "lowrank_cost": c, "best_jtp": bjp, "best_jtp_replaced": bjr, "bp": bp, "bp_replaced": bpr,
            "lowrank_vs_plain": round(c / plain.extraction.cost, 4), "lowrank_vs_best_known": round(c / known, 4),
            "uses_split": uses, "plain_s": round(t1 - t0, 2), "lowrank_s": round(t2 - t1, 2),
            "correct": err < 1e-8, "max_abs_err": f"{err:.2e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    with Pool(args.jobs) as pool:
        rows = []
        for r in pool.imap(run, list(cases(args.quick))):
            rows.append(r)
            print(f"{r['family']:12s} n={r['n']} K={r['K']:<2} r={r['rank']:<2} plain={r['plain_cost']} "
                  f"lowrank={r['lowrank_cost']} known={r['lowrank_vs_best_known']} correct={r['correct']}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

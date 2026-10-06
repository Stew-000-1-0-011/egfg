"""Phase A: how search time and result cost grow with problem size, per search setting.

Usage: uv run python experiments/scaling.py --out results/scaling.csv [--quick] [--jobs 3]

Each run happens in its own child process with a time limit. Within one
(setting, family) series, n grows until a run hits a wall (time limit or
e-graph node limit); larger n are then skipped. Rows are appended as they
finish; when everything is done the file is rewritten with the comparison to
the baseline setting filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TIME_LIMIT_S = 120  # saturation + extraction, per run
K = 3

CONFIGS: dict[str, dict] = {
    "base": dict(strategy="bfs", rules="full", cluster_budget=None, seed=False, extractor="ilp"),
    "R=no_reverse": dict(strategy="bfs", rules="no_reverse", cluster_budget=None, seed=False, extractor="ilp"),
    "R=minimal": dict(strategy="bfs", rules="minimal", cluster_budget=None, seed=False, extractor="ilp"),
    "D=3": dict(strategy="bfs", rules="full", cluster_budget=3, seed=False, extractor="ilp"),
    "D=5": dict(strategy="bfs", rules="full", cluster_budget=5, seed=False, extractor="ilp"),
    "D=8": dict(strategy="bfs", rules="full", cluster_budget=8, seed=False, extractor="ilp"),
    "S=on": dict(strategy="bfs", rules="full", cluster_budget=None, seed=True, extractor="ilp"),
    "X=tree": dict(strategy="bfs", rules="full", cluster_budget=None, seed=False, extractor="tree"),
    "X=greedy": dict(strategy="bfs", rules="full", cluster_budget=None, seed=False, extractor="greedy"),
    "light": dict(strategy="bfs", rules="minimal", cluster_budget=3, seed=True, extractor="greedy"),
    "mid": dict(strategy="bfs", rules="no_reverse", cluster_budget=5, seed=True, extractor="greedy"),
    "precise": dict(strategy="bfs", rules="no_reverse", cluster_budget=5, seed=True, extractor="ilp"),
}
QUICK_CONFIGS = ["base", "D=3", "S=on", "X=greedy", "light"]

FAMILIES = ["chain", "star", "random_tree", "cycle", "grid2", "grid3", "random_sparse"]
QUICK_FAMILIES = ["chain", "cycle", "grid2"]
NS = [4, 6, 8, 12, 16, 24, 32, 48, 64]
QUICK_NS = [4]
SEEDS = {"random_tree": [0, 1], "random_sparse": [0, 1]}

COLUMNS = (
    "config,family,n,seed,K,num_vars,num_factors,rules,cluster_budget,seed_jt,extractor,"
    "status,wall,wall_reason,hit_limit,saturate_s,extract_s,total_s,"
    "egraph_nodes,max_cluster_nodes,num_clusters,max_cluster_vars,"
    "cost,optimal,jt_cost,cost_vs_jt,bp_cost,cost_vs_bp,base_cost,cost_vs_base,"
    "correct,checked_against,max_abs_err,error"
).split(",")


def grid_shape(rows: int, n: int) -> tuple[int, int]:
    return rows, max(2, round(n / rows))


def make_graph(family: str, n: int, seed: int):
    from egfg import generators as g

    if family in ("grid2", "grid3"):
        rows, cols = grid_shape(int(family[-1]), n)
        return g.grid(rows, cols, K, seed)
    return {
        "chain": g.chain,
        "star": g.star,
        "random_tree": g.random_tree,
        "cycle": g.cycle,
        "random_sparse": g.random_sparse,
    }[family](n, K, seed)


def actual_n(family: str, n: int) -> int:
    if family in ("grid2", "grid3"):
        r, c = grid_shape(int(family[-1]), n)
        return r * c
    return n


# ---------------------------------------------------------------------------
# worker: one run, in a child process
# ---------------------------------------------------------------------------


def worker(spec: dict) -> dict:
    import numpy as np

    from egfg.baselines import brute_force_marginals, factor_graph_bp_dag, junction_tree_dag
    from egfg.cost import dag_cost
    from egfg.evaluate import SUM_PRODUCT, evaluate
    from egfg.pipeline import marginals, optimize

    fg = make_graph(spec["family"], spec["n"], spec["seed"])
    t0 = time.perf_counter()
    res = optimize(fg, **CONFIGS[spec["config"]])  # strategy pinned to "bfs": the phase A setting
    total = time.perf_counter() - t0

    jt, _ = junction_tree_dag(fg)
    jt_cost = dag_cost(jt, fg)
    try:
        bp, _ = factor_graph_bp_dag(fg)
        bp_cost = dag_cost(bp, fg)
    except ValueError:  # loopy graph
        bp_cost = None
    if len(fg.cards) <= 12:
        ref, against = brute_force_marginals(fg), "brute_force"
    else:
        tables, _ = evaluate(jt, fg, SUM_PRODUCT)
        ref, against = {v: t.data / t.data.sum() for v, t in tables.items()}, "junction_tree"
    got = marginals(fg, res)
    err = max(float(np.max(np.abs(got[v] - ref[v]))) for v in fg.variables())
    cost = res.extraction.cost
    return {
        "num_vars": len(fg.cards),
        "num_factors": len(fg.factors),
        "hit_limit": res.hit_limit,
        "saturate_s": round(res.saturate_s, 3),
        "extract_s": round(res.extract_s, 3),
        "total_s": round(total, 3),
        "egraph_nodes": res.num_nodes,
        "max_cluster_nodes": res.max_cluster_nodes,
        "num_clusters": len(res.clusters),
        "max_cluster_vars": max(len(c) for c in res.clusters),
        "cost": cost,
        "optimal": res.extraction.optimal,
        "jt_cost": jt_cost,
        "cost_vs_jt": round(cost / jt_cost, 4),
        "bp_cost": "" if bp_cost is None else bp_cost,
        "cost_vs_bp": "" if bp_cost is None else round(cost / bp_cost, 4),
        "correct": err < 1e-8,
        "checked_against": against,
        "max_abs_err": f"{err:.2e}",
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_one(spec: dict) -> dict:
    row = {
        "config": spec["config"],
        "family": spec["family"],
        "n": actual_n(spec["family"], spec["n"]),
        "seed": spec["seed"],
        "K": K,
        "rules": CONFIGS[spec["config"]]["rules"],
        "cluster_budget": CONFIGS[spec["config"]]["cluster_budget"] or "",
        "seed_jt": CONFIGS[spec["config"]]["seed"],
        "extractor": CONFIGS[spec["config"]]["extractor"],
    }
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_S + 30,  # child start-up and the reference computations
            cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return {**row, "status": "timeout", "wall": True, "wall_reason": "time", "total_s": f">{TIME_LIMIT_S}"}
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or ["exit code %d" % proc.returncode])[-1][:200]
        return {**row, "status": "error", "wall": True, "wall_reason": "error", "error": msg}
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    reasons = []
    if out["total_s"] > TIME_LIMIT_S:
        reasons.append("time")
    if out["hit_limit"]:
        reasons.append("node_limit")
    return {**row, **out, "status": "ok", "wall": bool(reasons), "wall_reason": "+".join(reasons)}


def series(config: str, family: str, ns: list[int], emit) -> None:
    done: set[int] = set()
    for n in ns:
        if actual_n(family, n) in done:
            continue  # grid sizes round to the same shape
        done.add(actual_n(family, n))
        rows = [run_one({"config": config, "family": family, "n": n, "seed": s}) for s in SEEDS.get(family, [0])]
        for r in rows:
            emit(r)
        if any(r["wall"] for r in rows):
            return


def fill_base(rows: list[dict]) -> None:
    base = {
        (r["family"], r["n"], r["seed"]): r["cost"]
        for r in rows
        if r["config"] == "base" and r["status"] == "ok"
    }
    for r in rows:
        b = base.get((r["family"], r["n"], r["seed"]))
        if b is not None and r.get("cost") not in (None, ""):
            r["base_cost"] = b
            r["cost_vs_base"] = round(int(r["cost"]) / int(b), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--worker")
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))))
        return
    if not args.out:
        ap.error("--out is required")
    configs = QUICK_CONFIGS if args.quick else list(CONFIGS)
    families = QUICK_FAMILIES if args.quick else FAMILIES
    ns = QUICK_NS if args.quick else NS
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    lock = threading.Lock()
    start = time.perf_counter()
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()

        def emit(r: dict) -> None:
            with lock:
                rows.append(r)
                w.writerow(r)
                fh.flush()
                print(
                    f"[{time.perf_counter() - start:7.0f}s] {r['config']:12s} {r['family']:13s} n={r['n']:<3} "
                    f"seed={r['seed']} {r['status']:7s} wall={r['wall_reason'] or '-':15s} "
                    f"cost={r.get('cost', '')} jt={r.get('jt_cost', '')} t={r.get('total_s', '')}",
                    flush=True,
                )

        # one series per (setting, family); the baseline settings are submitted first
        jobs = [(c, f) for c in configs for f in families]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for fut in [pool.submit(series, c, f, ns, emit) for c, f in jobs]:
                fut.result()
    fill_base(rows)
    order = {c: i for i, c in enumerate(configs)}
    fam = {f: i for i, f in enumerate(families)}
    rows.sort(key=lambda r: (order[r["config"]], fam[r["family"]], int(r["n"]), int(r["seed"])))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

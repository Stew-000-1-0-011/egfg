"""Phase M: calibrate the cost model of the generated C code (cmodel.py) on measured run times.

1. Per problem (in parallel processes), several computations of all marginals: junction trees
   of a few elimination orders, fixed budgets, the partition search, and for small problems no
   partition (greedy, tree, and random variants drawn from the e-graph).
2. Every distinct program is compiled (threads) and timed one after another (as in the arena).
3. Non-negative least squares on the relative error, 5-fold cross-validation by problem
   (held-out relative errors, pairwise orders against the operation count), then a fit on all
   data written to results/cost_calibration.json.

Usage: uv run python experiments/cost_calibration.py [--jobs N] [--time 120]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from egfg import arena  # noqa: E402
from egfg.baselines import candidate_orders, junction_tree_dag  # noqa: E402
from egfg.ccodegen import compile_c_program, generate_c  # noqa: E402
from egfg.cmodel import FEATURES, CostModel, dag_features  # noqa: E402
from egfg.cost import dag_cost, max_intermediate_size  # noqa: E402
from egfg.extract import _build, _costs, _reachable_under, _start_choice, class_scopes  # noqa: E402
from egfg.generators import chain, cycle, grid, star  # noqa: E402
from egfg.pipeline import optimize  # noqa: E402

SMALL = 12
FOLDS = 5
MAX_TABLE = 1 << 22  # skip programs with a larger intermediate table (random variants can blow up)


def problems():
    out = []
    for p in arena.public_problems() + arena.larger_problems():
        if any(f["table"]["kind"] == "lowrank" for f in p["factors"]):
            continue
        out.append((p["name"], arena.to_factor_graph(p, arena.draw_tables(p, 0))))
    from partition_search import problems as phase_k

    out += phase_k()
    out += [(f"chain16_k{k}", chain(16, k)) for k in (2, 4, 8, 16, 32)]
    out += [(f"cycle8_k{k}", cycle(8, k)) for k in (2, 4, 8, 16)]
    out += [(f"grid3x3_k{k}", grid(3, 3, k)) for k in (2, 4, 6)]
    out += [(f"star8_k{k}", star(8, k)) for k in (4, 16)]
    seen, uniq = set(), []
    for name, fg in out:
        if name not in seen:
            seen.add(name)
            uniq.append((name, fg))
    return uniq


def random_variants(res, fg, n: int, seed: int):
    """Programs near the extracted one: random switches of e-nodes that keep the Dag acyclic."""
    g = res.saturation.graph
    scopes = class_scopes(g, fg)
    base = {**_start_choice(g, _costs(g, fg, scopes)), **res.extraction.dag.nodes}
    rng = np.random.default_rng(seed)
    roots = list(dict.fromkeys(g.roots.values()))
    from egfg.extract import _reach_cost

    out = []
    for k in range(n):
        choice = dict(base)
        for _ in range(4 * (k + 1)):
            cids = [c for c in _reachable_under(roots, choice) if len(g.classes[c]) > 1]
            if not cids:
                break
            cid = cids[int(rng.integers(len(cids)))]
            n_ = g.classes[cid][int(rng.integers(len(g.classes[cid])))]
            if not all(c in choice for c in n_.children):
                continue
            keep = choice[cid]
            choice[cid] = n_
            if _reach_cost(roots, choice, {(c, m): 0 for c, ms in g.classes.items() for m in ms}) is None:
                choice[cid] = keep
        out.append(_build(g, fg, choice, None, time.perf_counter()).dag)
    return out


def programs(args) -> list[tuple]:
    pi, limit = args
    name, fg = problems()[pi]
    dags = []
    for n, order in enumerate(candidate_orders(fg, 1)[:3]):
        dags.append((f"jt{n}", junction_tree_dag(fg, order)[0]))
    kw = dict(extractor="greedy", time_limit_s=60)
    for b in (4, 6, 8):
        dags.append((f"budget{b}", optimize(fg, cluster_budget=b, **kw).extraction.dag))
    dags.append(("tree_budget6", optimize(fg, cluster_budget=6, extractor="tree").extraction.dag))
    dags.append(("search", optimize(fg, partition="search", partition_time_s=limit, partition_jobs=1,
                                    boundary=False, **kw).extraction.dag))
    if len(fg.variables()) <= SMALL:
        res = optimize(fg, node_limit=20_000, **kw)
        dags.append(("none_greedy", res.extraction.dag))
        dags.append(("none_tree", optimize(fg, extractor="tree", node_limit=20_000).extraction.dag))
        dags += [(f"random{k}", d) for k, d in enumerate(random_variants(res, fg, 4, zlib.crc32(name.encode())))]
    out, seen = [], set()
    for label, dag in dags:
        if max_intermediate_size(dag, fg) > MAX_TABLE:
            continue
        f = dag_features(dag, fg)
        key = (dag_cost(dag, fg), tuple(f[k] for k in FEATURES))
        if key not in seen:
            seen.add(key)
            out.append((pi, name, label, dag, dag_cost(dag, fg), f))
    return out


def fit(rows) -> CostModel:
    from scipy.optimize import nnls

    A = np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)
    y = np.array([r["run_ns"] for r in rows], dtype=float)
    scale = A.max(axis=0)
    scale[scale == 0] = 1
    coef, _ = nnls(A / scale / y[:, None], np.ones(len(y)))  # relative error
    return CostModel({k: float(c / s) for k, c, s in zip(FEATURES, coef, scale)})


def pair_stats(rows, key) -> tuple[int, int]:
    """Pairs of programs of one problem whose run times differ by at least 10%: (ordered right by key, total)."""
    right = total = 0
    by: dict = {}
    for r in rows:
        by.setdefault(r["problem"], []).append(r)
    for rs in by.values():
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                if max(a["run_ns"], b["run_ns"]) < 1.1 * min(a["run_ns"], b["run_ns"]):
                    continue
                total += 1
                right += (a[key] - b[key]) * (a["run_ns"] - b["run_ns"]) > 0
    return right, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--time", type=float, default=120, help="time limit of one partition search (s)")
    ap.add_argument("--out", default="results/cost_model.csv")
    ap.add_argument("--calibration", default="results/cost_calibration.json")
    args = ap.parse_args()
    if os.environ.get("EGFG_CACHE"):
        raise SystemExit("unset EGFG_CACHE: times are measured")
    probs = problems()
    tasks = sorted(((pi, args.time) for pi in range(len(probs))), key=lambda t: -len(probs[t[0]][1].variables()))
    progs = []
    with ProcessPoolExecutor(args.jobs, mp_context=get_context("spawn")) as pool:
        for out in pool.map(programs, tasks):
            progs += out
            print(f"{out[0][1]}: {len(out)} programs", flush=True)

    def build(p):
        pi, _, _, dag, _, _ = p
        fg = probs[pi][1]
        try:
            return compile_c_program(generate_c(dag, fg), fg.variables(), fg.cards)
        except Exception as e:  # noqa: BLE001
            print("compile failed:", p[1], p[2], type(e).__name__, flush=True)
            return None

    rows = []
    with ThreadPoolExecutor(args.jobs) as tp:
        runs = list(tp.map(build, progs))
    for p, run in zip(progs, runs):  # timed one after another
        if run is None:
            continue
        pi, name, label, dag, flops, f = p
        fg = probs[pi][1]
        ns = arena.time_c(run, {fc.id: fc.table for fc in fg.factors}) * 1e9
        rows.append({"problem": name, "label": label, "vars": len(fg.variables()), "flops": flops, **f,
                     "run_ns": round(ns, 2), "fold": zlib.crc32(name.encode()) % FOLDS})
    print(f"{len(rows)} programs from {len(probs)} problems", flush=True)

    for fold in range(FOLDS):  # held-out predictions
        model = fit([r for r in rows if r["fold"] != fold])
        for r in rows:
            if r["fold"] == fold:
                r["cv_ns"] = round(model.predict(r), 2)
    model = fit(rows)
    for r in rows:
        r["fit_ns"] = round(model.predict(r), 2)
    rel = np.array([abs(r["cv_ns"] - r["run_ns"]) / r["run_ns"] for r in rows])
    summary = {"programs": len(rows), "problems": len(probs), "cv_median_rel_err": float(np.median(rel)),
               "cv_p90_rel_err": float(np.quantile(rel, 0.9)),
               "pairs_cv_model": pair_stats(rows, "cv_ns"), "pairs_flops": pair_stats(rows, "flops")}
    print(json.dumps(summary), flush=True)
    Path(args.calibration).write_text(json.dumps({"features": list(FEATURES), "coef": model.coef,
                                                  "summary": summary}, indent=1) + "\n")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

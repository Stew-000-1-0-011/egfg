"""Phase E: search strategies at a fixed node budget, and the cost reached along the way.

Usage: uv run python experiments/search_order.py --out results/search_order.csv [--quick] [--jobs 3]

Writes the final results to --out and the cost-vs-nodes curves next to it
(`*_curve.csv`, recorded for the larger node limit only).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TIME_LIMIT_S = 120
K = 3

STRATS = {
    "bfs": dict(strategy="bfs"),
    "staged": dict(strategy="staged"),
    "backoff": dict(strategy="backoff"),
    "backoff_tight": dict(strategy="backoff", match_limit=200, ban_length=2),
    "seeds": dict(strategy="seeds"),
    "restart": dict(strategy="restart"),
    "seeds+staged": dict(strategy="seeds+staged"),
    "seeds+restart": dict(strategy="seeds+restart"),
}
SEEDLESS = ["bfs", "staged", "backoff", "backoff_tight", "restart"]  # for templates (they carry their own seed)

COLUMNS = (
    "case,family,n,rules,node_limit,strategy,status,hit_limit,steps,nodes,search_s,extract_s,cost,"
    "jt_cost,best_jt_cost,best_jtp_cost,cost_vs_jt,cost_vs_best_jtp,correct,max_abs_err,error"
).split(",")
CURVE_COLUMNS = "case,family,n,rules,node_limit,strategy,step,nodes,seconds,cost".split(",")


def cases(quick: bool) -> list[tuple[str, str, int]]:
    if quick:
        return [("fg", "star", 6), ("fg", "cycle", 6)]
    out = []
    for fam in ("chain", "star", "random_tree", "cycle", "grid2", "grid3", "random_sparse"):
        seen = set()
        for n in (6, 8, 12, 16):
            m = _actual_n(fam, n)
            if m not in seen:
                seen.add(m)
                out.append(("fg", fam, n))
    out += [("template", "chmm4", 4), ("gaussian", "coupled4x2", 8)]
    return out


def _grid(fam, n):
    rows = int(fam[-1])
    return rows, max(2, round(n / rows))


def _actual_n(fam, n):
    if fam.startswith("grid"):
        r, c = _grid(fam, n)
        return r * c
    return n


def make_fg(fam: str, n: int):
    from egfg import generators as g

    if fam.startswith("grid"):
        r, c = _grid(fam, n)
        return g.grid(r, c, K, 0)
    return {"chain": g.chain, "star": g.star, "random_tree": g.random_tree, "cycle": g.cycle,
            "random_sparse": g.random_sparse}[fam](n, K, 0)


def worker(spec: dict) -> dict:
    import numpy as np

    from egfg.baselines import best_junction_tree, brute_force_marginals, junction_tree_dag
    from egfg.cost import dag_cost
    from egfg.evaluate import SUM_PRODUCT, evaluate
    from egfg.ir import all_marginal_queries
    from egfg.search import SearchConfig, extract_result, search

    cfg = SearchConfig(rules=spec["rules"], node_limit=spec["node_limit"], time_limit_s=TIME_LIMIT_S,
                       **STRATS[spec["strategy"]])
    trace = spec["trace"]
    if spec["kind"] == "fg":
        fg = make_fg(spec["family"], spec["n"])
        res = search(fg, all_marginal_queries(fg), cfg, trace=trace)
        t0 = time.perf_counter()
        ex = extract_result(res, fg)
        ext = time.perf_counter() - t0
        jt = dag_cost(junction_tree_dag(fg)[0], fg)
        bj = best_junction_tree(fg)[1]
        bjp = best_junction_tree(fg, share_products=True)[1]
        tables, _ = evaluate(ex.dag, fg, SUM_PRODUCT)
        if len(fg.cards) <= 12:
            ref = brute_force_marginals(fg)
        else:
            t2, _ = evaluate(junction_tree_dag(fg)[0], fg, SUM_PRODUCT)
            ref = {v: t.data / t.data.sum() for v, t in t2.items()}
        err = max(float(np.max(np.abs(tables[v].data / tables[v].data.sum() - ref[v]))) for v in fg.variables())
        cost = ex.cost
    elif spec["kind"] == "template":
        from egfg.dynamic import compile_filter, jt_program, local_step
        from egfg.generators import coupled_hmm

        model = coupled_hmm(4, K)
        loc = local_step(model, "step")
        res = search(loc.fg, loc.queries, cfg, inputs=loc.inputs, seeds=loc.seeds, trace=trace)
        t0 = time.perf_counter()
        ex = extract_result(res, loc.fg)
        ext = time.perf_counter() - t0
        jt = jt_program(model).step.cost
        bj = bjp = ""
        # correctness: the extracted step equals the junction-tree step on random inputs
        err = _template_error(model, loc, ex)
        cost = ex.cost
    else:
        res, cost, jt, ext, err = _gaussian(cfg, trace)
        bj = bjp = ""
    curve = [dict(step=p.step, nodes=p.nodes, seconds=p.seconds, cost=p.cost) for p in res.trace]
    return {
        "hit_limit": res.hit_limit,
        "steps": res.steps,
        "nodes": res.num_nodes,
        "search_s": round(res.seconds, 3),
        "extract_s": round(ext, 3),
        "cost": cost,
        "jt_cost": jt,
        "best_jt_cost": bj,
        "best_jtp_cost": bjp,
        "cost_vs_jt": round(cost / jt, 4),
        "cost_vs_best_jtp": round(cost / bjp, 4) if bjp != "" else "",
        "correct": err < 1e-8,
        "max_abs_err": f"{err:.2e}",
        "curve": curve,
    }


def _template_error(model, loc, ex) -> float:
    """Compare the extracted step with the junction-tree step on random observations and input."""
    import numpy as np

    from egfg.dynamic import FilterProgram, Template, filter_marginals, jt_program
    from egfg.generators import random_observations

    jt = jt_program(model)
    prog = FilterProgram(model, jt.head, Template(loc, ex.dag, ex.cost))
    obs = random_observations(model, 5, seed=3)
    a, b = filter_marginals(prog, obs), filter_marginals(jt, obs)
    return max(float(np.max(np.abs(a[t][n] - b[t][n]))) for t in range(5) for n in model.states)


def _gaussian(cfg, trace):
    """The step template of the coupled 4-block Gaussian model, with representation-aware extraction."""
    import numpy as np

    from egfg.dynamic import MSG, local_step
    from egfg.gdynamic import (
        MOMENT,
        REPS,
        GFilterProgram,
        GTemplate,
        _template,
        compile_gaussian_filter,
        gaussian_filter,
        implementations,
        kalman_program,
        kalman_reference,
    )
    from egfg.generators import gaussian_coupled, gaussian_observations
    from egfg.repextract import extract_rep
    from egfg.search import search

    model = gaussian_coupled(4, 2)
    loc = local_step(model.structure(), "step")
    res = search(loc.fg, loc.queries, cfg, inputs=loc.inputs, seeds=loc.seeds, trace=False)
    dims = model.local_dims()
    facs = model.factors("step")
    t0 = time.perf_counter()
    best = None
    for r in REPS["both"]:
        impls = implementations(res.graph, loc, facs, dims, r)
        roots = {q: (cid, r if q == MSG else MOMENT) for q, cid in res.graph.roots.items()}
        ch = extract_rep(res.graph, impls, roots, "greedy", None, 60)
        if best is None or ch.cost < best[0].cost:
            best = (ch, r)
    ext = time.perf_counter() - t0
    # correctness: run the filter with this step template (head from the standard compile)
    base = compile_gaussian_filter(model, extractor="greedy", rules="minimal")
    ch, r = best
    if r != base.fwd_rep:  # the head must hand over the message in the same representation
        head = compile_gaussian_filter(model, extractor="greedy", rules="minimal", reps="moment" if r == MOMENT else "info").head
    else:
        head = base.head
    step = GTemplate(loc, facs, ch, r, res)
    prog = GFilterProgram(model, head, step, 0.0)
    obs = gaussian_observations(model, 6)
    got, ref = gaussian_filter(prog, obs), kalman_reference(model, obs)
    err = max(float(np.max(np.abs(a.mean[n] - b.mean[n]))) for a, b in zip(got, ref) for n in model.dims)
    return res, ch.cost, kalman_program(model).step.cost, ext, err


def run_one(c, rules, node_limit, strategy) -> tuple[dict, list[dict]]:
    kind, fam, n = c
    head = {"case": kind, "family": fam, "n": _actual_n(fam, n) if kind == "fg" else n, "rules": rules,
            "node_limit": node_limit, "strategy": strategy}
    spec = {"kind": kind, "family": fam, "n": n, "rules": rules, "node_limit": node_limit, "strategy": strategy,
            "trace": node_limit >= 50_000 and kind != "gaussian"}
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
                              capture_output=True, text=True, timeout=TIME_LIMIT_S + 240, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {**head, "status": "timeout"}, []
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or [f"exit code {proc.returncode}"])[-1][:200]
        return {**head, "status": "error", "error": msg}, []
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    curve = [{**head, **p} for p in r.pop("curve")]
    return {**head, **r, "status": "ok"}, curve


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
    rules_list = ["no_reverse"] if args.quick else ["full", "no_reverse"]
    limits = [10_000] if args.quick else [10_000, 50_000]
    strats = ["bfs", "staged", "seeds", "restart"] if args.quick else list(STRATS)
    jobs = []
    for c in cases(args.quick):
        for rules in rules_list:
            for lim in limits:
                for s in strats:
                    if c[0] != "fg" and s not in SEEDLESS:
                        continue
                    jobs.append((c, rules, lim, s))
    start = time.perf_counter()
    rows, curves = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        for j, fut in zip(jobs, futures):
            r, cv = fut.result()
            rows.append(r)
            curves += cv
            print(f"[{time.perf_counter() - start:6.0f}s] {j[0][1]:13s} n={r['n']:<3} {j[1]:10s} {j[2]:6d} "
                  f"{j[3]:14s} {r['status']:7s} cost={r.get('cost', '')} bestJT+={r.get('best_jtp_cost', '')} "
                  f"nodes={r.get('nodes', '')} correct={r.get('correct', '')}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with out.with_name(out.stem + "_curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CURVE_COLUMNS)
        w.writeheader()
        w.writerows(curves)


if __name__ == "__main__":
    main()

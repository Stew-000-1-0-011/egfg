"""Phase D: linear-Gaussian filtering with representation-aware extraction.

Usage: uv run python experiments/gaussian.py --out results/gaussian.csv [--quick] [--jobs 3]

Each row is one (model, setting). Settings "KF" and "IF" are the standard Kalman
filter and the information filter (the stacked model in moment / information
form only). The other settings search the templates with the e-graph. Every
setting is run with and without amortizing the parameter-only work.
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

SEARCH_LIMIT_S = 120
ILP_LIMIT_S = 20
T = 64


def models(quick: bool) -> list[tuple[str, int, int]]:
    if quick:
        return [("vec", 2, 1), ("vec", 2, 4), ("blocks", 2, 1), ("coupled", 2, 1)]
    out = []
    for d in (2, 4, 8):
        for m in sorted({1, max(1, d // 2), d, 2 * d}):
            out.append(("vec", d, m))
    for kind in ("blocks", "coupled"):
        for k in (2, 3, 4):
            for b in (1, 2, 4):
                out.append((kind, k, b))
    return out


def settings(quick: bool) -> list[dict]:
    out = []
    for am in (False, True):
        out += [dict(name="KF", amortize=am), dict(name="IF", amortize=am)]
        for reps in ("both", "info"):
            for obj in ("total", "latency"):
                out.append(dict(name="light", search="light", reps=reps, objective=obj, amortize=am))
        if not quick:
            for obj in ("total", "latency"):
                out.append(dict(name="default", search="default", reps="both", objective=obj, amortize=am))
    return out


SEARCH = {
    "light": dict(rules="minimal", seed=True, extractor="greedy", strategy="bfs"),
    "default": dict(rules="full", seed=True, extractor="ilp", time_limit_s=ILP_LIMIT_S, strategy="bfs"),
}

COLUMNS = (
    "model,p1,p2,state_dim,obs_dim,setting,reps,objective,amortize,status,wall,search_s,step_nodes,hit_limit,"
    "fwd_rep,head_cost,step_cost,step_prep,step_latency,kf_step_cost,if_step_cost,step_vs_kf,step_vs_if,"
    "step_vs_best_baseline,ops,correct,max_rel_err,error"
).split(",")


def make(kind: str, p1: int, p2: int):
    from egfg.generators import gaussian_blocks, gaussian_coupled, gaussian_vec

    return {"vec": gaussian_vec, "blocks": gaussian_blocks, "coupled": gaussian_coupled}[kind](p1, p2)


def worker(spec: dict) -> dict:
    import numpy as np

    from egfg.gdynamic import (
        compile_gaussian_filter,
        gaussian_filter,
        information_program,
        kalman_program,
        kalman_reference,
        stack_obs,
    )
    from egfg.generators import gaussian_observations

    model = make(spec["model"], spec["p1"], spec["p2"])
    st = spec["setting"]
    am = st["amortize"]
    kf = kalman_program(model, extractor="tree", amortize_constants=am, strategy="bfs")
    inf = information_program(model, extractor="tree", amortize_constants=am, strategy="bfs")
    if st["name"] == "KF":
        prog = kf
    elif st["name"] == "IF":
        prog = inf
    else:
        prog = compile_gaussian_filter(model, reps=st["reps"], objective=st["objective"], amortize_constants=am,
                                       **SEARCH[st["search"]])
    obs = gaussian_observations(model, T, seed=0)
    ref = kalman_reference(model, obs)
    names = sorted(model.dims)
    stacked = st["name"] in ("KF", "IF")
    res = gaussian_filter(prog, stack_obs(obs) if stacked else obs)
    err = 0.0
    for r, q in zip(res, ref):
        if stacked:
            pairs = [(r.mean["s"], np.concatenate([q.mean[n] for n in names]))]
        else:
            pairs = [(r.mean[n], q.mean[n]) for n in names] + [(r.cov[n], q.cov[n]) for n in names]
        pairs.append((np.array([r.loglik]), np.array([q.loglik])))
        for a, b in pairs:
            err = max(err, float(np.max(np.abs(a - b) / (1e-12 + np.abs(b) + 1.0))))
    sat = prog.step.saturation
    best = min(kf.step.cost, inf.step.cost)
    return {
        "state_dim": sum(model.dims.values()),
        "obs_dim": sum(len(f.b) for f in model.observation),
        "search_s": round(prog.search_s, 3),
        "step_nodes": sat.num_nodes,
        "hit_limit": prog.head.saturation.hit_limit or sat.hit_limit,
        "fwd_rep": prog.fwd_rep,
        "head_cost": prog.head.cost,
        "step_cost": prog.step.cost,
        "step_prep": prog.step.prep_cost,
        "step_latency": prog.step.latency_cost,
        "kf_step_cost": kf.step.cost,
        "if_step_cost": inf.step.cost,
        "step_vs_kf": round(prog.step.cost / kf.step.cost, 4),
        "step_vs_if": round(prog.step.cost / inf.step.cost, 4),
        "step_vs_best_baseline": round(prog.step.cost / best, 4),
        "ops": ";".join(f"{k}:{v}" for k, v in sorted(prog.step.counts.items())),
        "correct": err < 1e-8,
        "max_rel_err": f"{err:.2e}",
    }


def run_one(kind: str, p1: int, p2: int, st: dict) -> dict:
    head = {"model": kind, "p1": p1, "p2": p2, "setting": st["name"], "reps": st.get("reps", ""),
            "objective": st.get("objective", ""), "amortize": st["amortize"]}
    spec = {"model": kind, "p1": p1, "p2": p2, "setting": st}
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
            capture_output=True, text=True, timeout=SEARCH_LIMIT_S + 300, cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return {**head, "status": "timeout", "wall": True}
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or [f"exit code {proc.returncode}"])[-1][:200]
        return {**head, "status": "error", "wall": True, "error": msg}
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    return {**head, **r, "status": "ok", "wall": r["search_s"] > SEARCH_LIMIT_S or bool(r["hit_limit"])}


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
    jobs = [(k, p1, p2, st) for (k, p1, p2) in models(args.quick) for st in settings(args.quick)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        rows = []
        for j, fut in zip(jobs, futures):
            r = fut.result()
            rows.append(r)
            print(
                f"[{time.perf_counter() - start:6.0f}s] {j[0]:7s} {j[1]} {j[2]} {r['setting']:7s} {r['reps']:4s} "
                f"{r['objective']:7s} am={int(r['amortize'])} {r['status']:7s} step={r.get('step_cost', '')} "
                f"kf={r.get('kf_step_cost', '')} if={r.get('if_step_cost', '')} correct={r.get('correct', '')}",
                flush=True,
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

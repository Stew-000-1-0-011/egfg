"""Phase C: preparation cost vs latency cost of the filtering step, per objective.

Usage: uv run python experiments/incremental.py --out results/incremental.csv [--quick] [--jobs 3]

Each row is one (model, search, objective). The rows with search "-" are the
forward algorithm and the per-step junction tree (no search). Templates are
searched in a child process; filtering is then run for T steps with
prepare() before and update() after each observation, timing both.
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
sys.path.insert(0, str(ROOT / "experiments"))

from template import checked_times, make_model, reference  # noqa: E402

SEARCH_LIMIT_S = 120
K = 3
T = 64

MODELS = ["hmm", "fhmm2", "fhmm3", "fhmm4", "chmm2", "chmm3", "chmm4"]
QUICK_MODELS = ["hmm", "fhmm2", "chmm2"]
SEARCHES = {
    "light": dict(rules="minimal", seed=True, extractor="greedy", strategy="bfs"),
    "default": dict(rules="full", seed=True, extractor="ilp", strategy="bfs"),
}
QUICK_SEARCHES = ["light"]
OBJECTIVES = ["total", "latency", "weighted:2", "weighted:4", "weighted:10"]
QUICK_OBJECTIVES = ["total", "latency"]

COLUMNS = (
    "model,m,K,search,objective,status,wall,search_s,step_nodes,hit_limit,"
    "head_prep,head_latency,head_cost,step_prep,step_latency,step_cost,"
    "fwd_step_prep,fwd_step_latency,fwd_step_cost,latency_vs_forward,cost_vs_forward,"
    "latency_vs_total,cost_vs_total,prepare_ms,update_ms,T,checked_times,checked_against,correct,max_abs_err,error"
).split(",")


def worker(spec: dict) -> dict:
    import numpy as np

    from egfg.dynamic import Filter, compile_filter, forward_program, jt_program
    from egfg.generators import random_observations

    model = make_model(spec["model"])
    fwd = forward_program(model)
    if spec["search"] == "-":
        prog = fwd if spec["objective"] == "forward" else jt_program(model)
    else:
        prog = compile_filter(model, objective=spec["objective"], **SEARCHES[spec["search"]])
    obs = random_observations(model, spec["T"], seed=0)
    f = Filter(prog)
    prep = upd = 0.0
    got = []
    for tabs in obs:
        t0 = time.perf_counter()
        f.prepare()
        t1 = time.perf_counter()
        out = f.update(tabs)
        t2 = time.perf_counter()
        prep, upd = prep + (t1 - t0), upd + (t2 - t1)
        got.append({n: out[n].data / out[n].data.sum() for n in model.states})
    err, against = 0.0, set()
    ts = checked_times(spec["T"])
    for t in ts:
        ref, how = reference(model, obs, t)
        against.add(how)
        err = max(err, max(float(np.max(np.abs(got[t][n] - ref[n]))) for n in model.states))
    row = {
        "search_s": round(prog.search_s, 3),
        "head_prep": prog.head.prep_cost,
        "head_latency": prog.head.latency_cost,
        "head_cost": prog.head.cost,
        "step_prep": prog.step.prep_cost,
        "step_latency": prog.step.latency_cost,
        "step_cost": prog.step.cost,
        "fwd_step_prep": fwd.step.prep_cost,
        "fwd_step_latency": fwd.step.latency_cost,
        "fwd_step_cost": fwd.step.cost,
        "latency_vs_forward": round(prog.step.latency_cost / fwd.step.latency_cost, 4),
        "cost_vs_forward": round(prog.step.cost / fwd.step.cost, 4),
        "prepare_ms": round(1000 * prep / spec["T"], 4),
        "update_ms": round(1000 * upd / spec["T"], 4),
        "T": spec["T"],
        "checked_times": len(ts),
        "checked_against": "+".join(sorted(against)),
        "correct": err < 1e-8,
        "max_abs_err": f"{err:.2e}",
    }
    if prog.step.saturation is not None:
        row["step_nodes"] = prog.step.saturation.num_nodes
        row["hit_limit"] = prog.head.saturation.hit_limit or prog.step.saturation.hit_limit
    return row


def run_one(model: str, search: str, objective: str) -> dict:
    head = {"model": model, "m": 1 if model == "hmm" else int(model[-1]), "K": K, "search": search,
            "objective": objective}
    spec = {"model": model, "search": search, "objective": objective, "T": T}
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=SEARCH_LIMIT_S + 300,
            cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return {**head, "status": "timeout", "wall": True}
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or [f"exit code {proc.returncode}"])[-1][:200]
        return {**head, "status": "error", "wall": True, "error": msg}
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    return {**head, **r, "status": "ok", "wall": r["search_s"] > SEARCH_LIMIT_S or bool(r.get("hit_limit"))}


def fill_vs_total(rows: list[dict]) -> None:
    total = {(r["model"], r["search"]): r for r in rows if r["objective"] == "total" and r["status"] == "ok"}
    for r in rows:
        b = total.get((r["model"], r["search"]))
        if b is not None and r["status"] == "ok":
            r["latency_vs_total"] = round(r["step_latency"] / b["step_latency"], 4)
            r["cost_vs_total"] = round(r["step_cost"] / b["step_cost"], 4)


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
    models = QUICK_MODELS if args.quick else MODELS
    searches = QUICK_SEARCHES if args.quick else list(SEARCHES)
    objectives = QUICK_OBJECTIVES if args.quick else OBJECTIVES
    jobs = [(m, "-", o) for m in models for o in ("forward", "jt")]
    jobs += [(m, s, o) for m in models for s in searches for o in objectives]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        rows = []
        for j, fut in zip(jobs, futures):
            r = fut.result()
            rows.append(r)
            print(
                f"[{time.perf_counter() - start:6.0f}s] {j[0]:6s} {j[1]:8s} {j[2]:12s} {r['status']:7s} "
                f"prep={r.get('step_prep', '')} latency={r.get('step_latency', '')} "
                f"search={r.get('search_s', '')} correct={r.get('correct', '')}",
                flush=True,
            )
    fill_vs_total(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

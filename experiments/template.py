"""Phase B: filtering with time-step templates.

Usage: uv run python experiments/template.py --out results/template.csv [--quick] [--jobs 3]

For each model and setting, the head and step templates are searched once (in a
child process), then reused for every T. Each row is one (model, setting, T).
The settings "forward" and "jt" are the forward algorithm and the per-step
junction tree, built without any search.
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
K = 3

MODELS = ["hmm", "fhmm2", "fhmm3", "fhmm4", "chmm2", "chmm3", "chmm4"]
QUICK_MODELS = ["hmm", "fhmm2", "chmm2"]
SETTINGS = {
    "forward": None,
    "jt": None,
    "default": dict(rules="full", seed=True, extractor="ilp", strategy="bfs"),
    "light": dict(rules="minimal", seed=True, extractor="greedy", strategy="bfs"),
}
QUICK_SETTINGS = ["forward", "default", "light"]
TS = [1, 2, 4, 8, 16, 64, 256, 1024]
QUICK_TS = [1, 2, 4, 16]

COLUMNS = (
    "model,m,K,setting,T,status,wall,search_s,head_saturate_s,head_extract_s,step_saturate_s,step_extract_s,"
    "head_nodes,step_nodes,hit_limit,head_cost,step_cost,total_cost,forward_step_cost,step_vs_forward,"
    "jt_step_cost,step_vs_jt,total_vs_forward,eval_s,eval_s_per_step,checked_times,checked_against,"
    "correct,max_abs_err,error"
).split(",")


def make_model(name: str):
    from egfg.generators import coupled_hmm, factorial_hmm, hmm

    if name == "hmm":
        return hmm(K)
    m = int(name[-1])
    return factorial_hmm(m, K) if name.startswith("fhmm") else coupled_hmm(m, K)


def checked_times(T: int) -> list[int]:
    """The first 8 times, the last one, and about 8 in between."""
    ts = set(range(min(8, T))) | {T - 1}
    if T > 9:
        ts |= {round(8 + i * (T - 9) / 8) for i in range(9)}
    return sorted(t for t in ts if t < T)


def reference(model, obs, t: int):
    """Filtering answer at time t, and how it was obtained."""
    import numpy as np

    from egfg.baselines import brute_force_marginals, junction_tree_dag
    from egfg.dynamic import reference_filter, unroll
    from egfg.evaluate import SUM_PRODUCT, evaluate

    fg = unroll(model, obs[: t + 1])
    if len(fg.cards) <= 12:
        bm = brute_force_marginals(fg)
        return {n: bm[f"{n}@{t}"] for n in model.states}, "brute_force"
    if len(fg.cards) <= 64:
        order = sorted(fg.cards, key=lambda v: (int(v.rsplit("@", 1)[1]), v))
        dag, _ = junction_tree_dag(fg, order)
        tables, _ = evaluate(dag, fg, SUM_PRODUCT)
        return {n: tables[f"{n}@{t}"].data / tables[f"{n}@{t}"].data.sum() for n in model.states}, "junction_tree"
    ref = reference_filter(model, obs[: t + 1])[t]
    return {n: np.asarray(ref[n]) for n in model.states}, "joint_state"


def worker(spec: dict) -> list[dict]:
    import numpy as np

    from egfg.dynamic import compile_filter, filter_marginals, forward_program, jt_program
    from egfg.generators import random_observations

    model = make_model(spec["model"])
    fwd, jt = forward_program(model), jt_program(model)
    setting = SETTINGS[spec["setting"]]
    if setting is None:
        prog = fwd if spec["setting"] == "forward" else jt
    else:
        prog = compile_filter(model, **setting)
    base = {
        "search_s": round(prog.search_s, 3),
        "head_cost": prog.head.cost,
        "step_cost": prog.step.cost,
        "forward_step_cost": fwd.step.cost,
        "step_vs_forward": round(prog.step.cost / fwd.step.cost, 4),
        "jt_step_cost": jt.step.cost,
        "step_vs_jt": round(prog.step.cost / jt.step.cost, 4),
    }
    if setting is not None:
        hs, he = prog.head.saturation, prog.head.extraction
        ss, se = prog.step.saturation, prog.step.extraction
        base.update(
            head_saturate_s=round(hs.seconds, 3),
            head_extract_s=round(he.seconds, 3),
            step_saturate_s=round(ss.seconds, 3),
            step_extract_s=round(se.seconds, 3),
            head_nodes=hs.num_nodes,
            step_nodes=ss.num_nodes,
            hit_limit=hs.hit_limit or ss.hit_limit,
        )
    rows = []
    for T in spec["Ts"]:
        obs = random_observations(model, T, seed=0)
        t0 = time.perf_counter()
        fm = filter_marginals(prog, obs)
        ev = time.perf_counter() - t0
        err, against = 0.0, set()
        ts = checked_times(T)
        for t in ts:
            ref, how = reference(model, obs, t)
            against.add(how)
            err = max(err, max(float(np.max(np.abs(fm[t][n] - ref[n]))) for n in model.states))
        rows.append(
            {
                **base,
                "T": T,
                "total_cost": prog.total_cost(T),
                "total_vs_forward": round(prog.total_cost(T) / fwd.total_cost(T), 4),
                "eval_s": round(ev, 3),
                "eval_s_per_step": f"{ev / T:.2e}",
                "checked_times": len(ts),
                "checked_against": "+".join(sorted(against)),
                "correct": err < 1e-8,
                "max_abs_err": f"{err:.2e}",
            }
        )
    return rows


def run_series(model: str, setting: str, Ts: list[int]) -> list[dict]:
    head = {"model": model, "m": 1 if model == "hmm" else int(model[-1]), "K": K, "setting": setting}
    spec = {"model": model, "setting": setting, "Ts": Ts}
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=SEARCH_LIMIT_S + 900,  # search limit, plus evaluating and checking every T
            cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return [{**head, "T": T, "status": "timeout", "wall": True} for T in Ts]
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or [f"exit code {proc.returncode}"])[-1][:200]
        return [{**head, "T": T, "status": "error", "wall": True, "error": msg} for T in Ts]
    rows = json.loads(proc.stdout.strip().splitlines()[-1])
    out = []
    for r in rows:
        wall = r["search_s"] > SEARCH_LIMIT_S or bool(r.get("hit_limit"))
        out.append({**head, **r, "status": "ok", "wall": wall})
    return out


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
    settings = QUICK_SETTINGS if args.quick else list(SETTINGS)
    Ts = QUICK_TS if args.quick else TS
    jobs = [(m, s) for m in models for s in settings]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_series, m, s, Ts) for m, s in jobs]
        results = []
        for (m, s), fut in zip(jobs, futures):
            rows = fut.result()
            results += rows
            last = rows[-1]
            print(
                f"[{time.perf_counter() - start:6.0f}s] {m:6s} {s:8s} {last['status']:7s} "
                f"search={last.get('search_s', '')} step_cost={last.get('step_cost', '')} "
                f"fwd={last.get('forward_step_cost', '')} correct={all(r.get('correct') for r in rows)}",
                flush=True,
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(results)


if __name__ == "__main__":
    main()

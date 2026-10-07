"""Phase I experiment: choosing where to merge with the expected error computed from the model.

For every model, the best program over all message forms is compiled for several prices μ
(operations per nat of expected KL) and, for comparison, several phase H penalties λ. Each
program's summed expected bound (computed before running) is compared with the collapse KL
bounds measured at run time on observations drawn from the model, and with the actual error
(the exact posterior by enumeration; a long run against GPB2).

Usage: uv run python experiments/expected_error.py --out results/expected_error.csv [--quick] [--jobs N]
Compilation (per model and message form) and evaluation (per program) run in parallel; every
extraction starts afresh (the greedy extraction is a local search: a warm start would make the
choices depend on the order of the settings).
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from approximation import SEEDS, T_LONG, errors, label, models  # noqa: E402
from egfg.switching import compile_switching_filter, message_groups, run_filter  # noqa: E402
from egfg.switching_bound import ExpectedBound  # noqa: E402
from egfg.switching_ref import exact_filter, gpb1, gpb2, imm, simulate, stack  # noqa: E402

PRICES = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e4)
LAMS = (0.0, 1.0, 10.0, 100.0, 1e6)


def _forms(model):
    forms = message_groups(model)
    big = len(model.components()) >= 3
    return (forms[0], forms[-1]) if big and len(forms) > 1 else tuple(forms), big


def compile_task(args) -> list[tuple]:
    """One model and one message form: the program for every price and λ (one process; the
    saturation is shared, every extraction starts afresh)."""
    quick, mi, fi = args
    name, model = models(quick)[mi]
    forms, big = _forms(model)
    kw = dict(strategy="staged", rules="full") if big else {}
    t0 = time.perf_counter()
    eb = ExpectedBound(model)
    bound_setup = time.perf_counter() - t0
    cache: dict = {}
    out = []
    for selector, settings in (("price", [dict(price=p) for p in PRICES]), ("lam", [dict(lam=x) for x in LAMS])):
        for st in settings:
            t1 = time.perf_counter()
            p = compile_switching_filter(model, groups=forms[fi], bound=eb, cache=cache, **st, **kw)
            out.append((mi, fi, selector, st.get("price", st.get("lam")), p.step.choice.cost, p,
                        time.perf_counter() - t1, bound_setup, len(eb.cache), eb.L))
    return out


def evaluate_task(args) -> dict:
    """One program: the measured collapse bounds and the actual errors (one process)."""
    quick, mi, p, info = args
    name, model = models(quick)[mi]
    st = stack(model)
    nj = len(st.modes)
    T_ex = max(3, min(8, int(np.log(2e5) / np.log(max(nj, 2)))))
    labs, tvs, mes, meas, long_me = [], [], [], [], []
    for seed in SEEDS:
        obs = simulate(model, T_LONG, seed=seed)
        short = obs[:T_ex]
        refs = {"imm": imm(st, short), "gpb2": gpb2(st, short), "gpb1": gpb1(st, short)}
        res = run_filter(p, short)
        labs.append(label(model, res, refs))
        tv, me = errors(model, res, exact_filter(st, short))
        tvs.append(tv)
        mes.append(me)
        long = run_filter(p, obs)
        meas.append(float(np.mean([sum(r.collapse_kl) for r in long[5:]])))
        long_me.append(errors(model, long, gpb2(st, obs))[1])
    return {"model": name, "groups": " ".join("+".join(g) for g in p.groups), "flops": p.step.flops,
            "collapses": p.collapses, "chosen_by_price": " ".join(f"{v:g}" for v in info["price"]),
            "chosen_by_lam": " ".join(f"{v:g}" for v in info["lam"]), "matches": max(set(labs), key=labs.count),
            "eps_static": round(p.step.eps, 5), "kl_measured": round(float(np.mean(meas)), 5),
            "mode_tv": round(float(np.mean(tvs)), 5), "mean_err_sd": round(float(np.mean(mes)), 5),
            "long_mean_err_vs_gpb2": round(float(np.mean(long_me)), 5), "window": info["window"],
            "bound_setup_s": round(info["bound_setup_s"], 2), "bound_sites": info["sites"],
            "compile_s": round(info["compile_s"], 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/expected_error.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    ms = models(args.quick)
    tasks = [(args.quick, mi, fi) for mi, (_, model) in enumerate(ms) for fi in range(len(_forms(model)[0]))]
    tasks.sort(key=lambda t: -len(ms[t[1]][1].components()))  # the slowest first
    with ProcessPoolExecutor(args.jobs, mp_context=get_context("spawn")) as pool:
        compiled = [r for rs in pool.map(compile_task, tasks) for r in rs]
        # per model and setting, the cheapest program over the message forms
        best: dict = {}
        for mi, fi, selector, value, cost, p, secs, bsetup, sites, window in compiled:
            k = (mi, selector, value)
            if k not in best or cost < best[k][0]:
                best[k] = (cost, p, secs, bsetup, sites, window)
        progs: dict = {}
        for (mi, selector, value), (cost, p, secs, bsetup, sites, window) in sorted(best.items(), key=lambda kv: kv[0]):
            key = (mi, tuple(p.groups), tuple(sorted(p.step.counts.items())), p.step.flops)
            info = progs.setdefault(key, {"p": p, "mi": mi, "price": [], "lam": [], "compile_s": 0.0, "window": window,
                                          "bound_setup_s": bsetup, "sites": sites})
            info[selector].append(value)
            info["compile_s"] = max(info["compile_s"], secs)
        todo = [(args.quick, info["mi"], info["p"], {k: v for k, v in info.items() if k != "p"})
                for _, info in sorted(progs.items(), key=lambda kv: (kv[0][0], kv[1]["p"].step.flops))]
        rows = []
        for row in pool.map(evaluate_task, todo):
            print(row, flush=True)
            rows.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

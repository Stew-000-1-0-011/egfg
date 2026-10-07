"""Phase I experiment: choosing where to merge with the expected error computed from the model.

For every model, the best program over all message forms is compiled for several prices μ
(operations per nat of expected KL) and, for comparison, several phase H penalties λ. Each
program's summed expected bound (computed before running) is compared with the collapse KL
bounds measured at run time on observations drawn from the model, and with the actual error
(the exact posterior by enumeration; a long run against GPB2).

Usage: uv run python experiments/expected_error.py --out results/expected_error.csv [--quick]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
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


def run_model(name, model) -> list[dict]:
    st = stack(model)
    nj = len(st.modes)
    T_ex = max(3, min(8, int(np.log(2e5) / np.log(max(nj, 2)))))
    t0 = time.perf_counter()
    eb = ExpectedBound(model)
    bound_setup = time.perf_counter() - t0
    big = len(model.components()) >= 3
    kw = dict(strategy="staged", rules="full") if big else {}
    forms = message_groups(model)
    if big:
        forms = [forms[0], forms[-1]]
    cache: dict = {}
    progs = {}
    for selector, settings in [("price", [dict(price=p) for p in PRICES]), ("lam", [dict(lam=x) for x in LAMS])]:
        for s in settings:
            best = None
            t1 = time.perf_counter()
            for grp in forms:
                p = compile_switching_filter(model, groups=grp, bound=eb, cache=cache, **s, **kw)
                c = p.step.choice.cost
                if best is None or c < best[0]:
                    best = (c, p)
            p = best[1]
            key = (tuple(p.groups), tuple(sorted(p.step.counts.items())), p.step.flops)
            value = s.get("price", s.get("lam"))
            progs.setdefault(key, {"p": p, "price": [], "lam": [], "compile_s": 0.0})
            progs[key][selector].append(value)
            progs[key]["compile_s"] = max(progs[key]["compile_s"], time.perf_counter() - t1)
    rows = []
    for key, info in progs.items():
        p = info["p"]
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
        row = {"model": name, "groups": " ".join("+".join(g) for g in p.groups), "flops": p.step.flops,
               "collapses": p.collapses, "chosen_by_price": " ".join(f"{v:g}" for v in info["price"]),
               "chosen_by_lam": " ".join(f"{v:g}" for v in info["lam"]), "matches": max(set(labs), key=labs.count),
               "eps_static": round(p.step.eps, 5), "kl_measured": round(float(np.mean(meas)), 5),
               "mode_tv": round(float(np.mean(tvs)), 5), "mean_err_sd": round(float(np.mean(mes)), 5),
               "long_mean_err_vs_gpb2": round(float(np.mean(long_me)), 5), "window": eb.L,
               "bound_setup_s": round(bound_setup, 2), "bound_sites": len(eb.cache), "compile_s": round(info["compile_s"], 2)}
        rows.append(row)
        print(row, flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/expected_error.csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rows = []
    for name, model in models(args.quick):
        rows += run_model(name, model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

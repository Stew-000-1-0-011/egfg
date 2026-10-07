"""Phase H experiment: where to merge mixtures in switching models.

For every model, egfg's candidate programs (every message form, λ = 0 and large) and the
state-wide GPB1 / GPB2 / IMM (the same search on the stacked model: one joint mode, one stacked
state) are run on observations drawn from the model and compared with the exact posterior
(every mode history; short runs) and over a long run with GPB2.

Usage: uv run python experiments/approximation.py --out results/approximation.csv [--quick]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egfg.switching import compile_switching_programs, run_filter  # noqa: E402
from egfg.switching_ref import (  # noqa: E402
    exact_filter,
    gpb1,
    gpb2,
    imm,
    maneuver,
    marginals,
    message_joint,
    outlier,
    simulate,
    stack,
    stacked_model,
    stacked_obs,
    targets,
    tv_1d,
)

SEEDS = (11, 12, 13)
T_LONG = 50


def models(quick: bool):
    out = [("maneuver(2,1)", maneuver(2, 1)), ("maneuver(3,1)", maneuver(3, 1)), ("maneuver(2,2)", maneuver(2, 2)),
           ("maneuver(3,2,obs)", maneuver(3, 2, obs_mode=True)), ("outlier(1)", outlier(1)), ("outlier(2)", outlier(2)),
           ("targets(2,2,1)", targets(2, 2, 1)), ("targets(2,2,2)", targets(2, 2, 2)), ("targets(3,2,1)", targets(3, 2, 1))]
    return out[:2] + out[6:7] if quick else out


def label(model, res, refs):
    """The references the program reproduces (several when they coincide on this model)."""
    return "=".join(name for name, ref in refs.items() if _diff(model, res, ref) < 1e-8) or "-"


def _diff(model, res, ref):
    st = stack(model)
    err = 0.0
    for r, e in zip(res, ref):
        probs, cont = marginals(st, e, model)
        for d in model.cards:
            err = max(err, float(np.max(np.abs(r.probs[d] - probs[d]))))
        for x in model.dims:
            err = max(err, float(np.max(np.abs(r.mean[x] - cont[x][0]))), float(np.max(np.abs(r.cov[x] - cont[x][1]))))
    return err


def errors(model, res, ref):
    """(mode TV, mean error in units of the exact standard deviation): the largest over steps and states."""
    st = stack(model)
    tv, me = 0.0, 0.0
    for r, e in zip(res, ref):
        probs, cont = marginals(st, e, model)
        for d in model.cards:
            tv = max(tv, 0.5 * float(np.sum(np.abs(r.probs[d] - probs[d]))))
        for x in model.dims:
            m, S = cont[x]
            me = max(me, float(np.sqrt((r.mean[x] - m) @ np.linalg.solve(S, r.mean[x] - m))))
    return tv, me


def run_model(name, model, uniform) -> list[dict]:
    st = stack(model)
    nj = len(st.modes)
    T_ex = max(3, min(8, int(math.log(2e5) / math.log(max(nj, 2)))))
    single = len(model.components()) == 1
    rows = []
    sets = [("egfg", model, lambda o: o)]
    if not single:
        sets.append(("state-wide", stacked_model(model), stacked_obs))
    for kind, m, conv in sets:
        t0 = time.perf_counter()
        # three or more components: breadth-first saturation hits the node limit early; the staged
        # search with every rule finds cheaper programs
        kw = dict(strategy="staged", rules="full") if uniform and kind == "egfg" else {}
        progs = compile_switching_programs(m, uniform=uniform and kind == "egfg", **kw)
        search = time.perf_counter() - t0
        for p in progs:
            row = {"model": name, "kind": kind, "groups": " ".join("+".join(g) for g in p.groups), "lam": p.lam,
                   "message_rep": p.step.fwd_rep, "flops": p.step.flops, "collapses": p.collapses,
                   "search_s": round(search, 2), "joint_modes": nj, "T_exact": T_ex}
            labs, tvs, mes, taus, kls, sat, tv1, long_me = [], [], [], [], [], [], [], []
            for seed in SEEDS:
                obs = simulate(model, T_LONG, seed=seed)
                short = obs[:T_ex]
                refs = {"imm": imm(st, short), "gpb2": gpb2(st, short), "gpb1": gpb1(st, short)}
                raw = run_filter(p, conv(short))
                res = _unstack(model, m, raw) if kind == "state-wide" else raw  # compared on the original names
                labs.append(label(model, res, refs))
                ex = exact_filter(st, short)
                tv, me = errors(model, res, ex)
                tvs.append(tv)
                mes.append(me)
                tau_t = [min(1.0, sum(x.tau for x in r.messages)) for r in raw]
                taus.append(tau_t[-1])
                sat.append(next((t for t, v in enumerate(tau_t) if v >= 1.0), len(tau_t)))
                kls.append(sum(x.klsum for x in raw[-1].messages))
                if kind == "egfg" and single and len(model.cards) == 1 and all(d == 1 for d in model.dims.values()):
                    worst = 0.0
                    for r, e in zip(raw, ex):
                        modes, approx = message_joint(model, r.messages)
                        worst = max(worst, tv_1d(e.comps, approx, modes))
                    tv1.append(worst)
                long = run_filter(p, conv(obs))
                if kind == "state-wide":
                    long = _unstack(model, m, long)
                long_me.append(errors(model, long, gpb2(st, obs))[1])
            row.update(matches=max(set(labs), key=labs.count), mode_tv=round(float(np.mean(tvs)), 5),
                       mean_err_sd=round(float(np.mean(mes)), 5), tv_1d=round(float(np.mean(tv1)), 5) if tv1 else "",
                       tau_final=round(float(np.mean(taus)), 4), tau_saturates_at=round(float(np.mean(sat)), 1),
                       klsum_final=round(float(np.mean(kls)), 4), long_mean_err_vs_gpb2=round(float(np.mean(long_me)), 5))
            rows.append(row)
            print(row, flush=True)
    return rows


def _unstack(model, stacked, res):
    """Results on the stacked model mapped back to the original names."""
    from egfg.switching import SResult

    st = stack(model)
    out = []
    for r in res:
        probs = {}
        for k, d in enumerate(st.dnames):
            p = np.zeros(model.cards[d])
            for J, pj in zip(st.modes, r.probs["m"]):
                p[J[k]] += pj
            probs[d] = p
        mean = {x: r.mean["x"][st.off[x]] for x in st.cnames}
        cov = {x: r.cov["x"][st.off[x], st.off[x]] for x in st.cnames}
        out.append(SResult(probs, mean, cov, r.tau, r.messages, r.collapse_kl))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/approximation.csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rows = []
    for name, model in models(args.quick):
        rows += run_model(name, model, uniform=len(model.components()) >= 3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

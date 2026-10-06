"""Linear-Gaussian filtering in C: egfg's programs against the Kalman and information filters
(the same C code generation) and GTSAM (C++).

Usage:
  uv run python arena/gaussian_bench/run.py --gtsam <GTSAM install prefix> --out arena/results/gaussian_c.json
The models are phase D's (vec, blocks, coupled), plus larger ones. Every program is checked
against the numpy Kalman filter (`kalman_reference`) on the first steps; the time is per step,
averaged over a run of T steps (the median of 7 runs).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from egfg import dynamic_gaussian_models as dgm  # noqa: E402
from egfg.dynamic import split  # noqa: E402
from egfg.gccodegen import generate_gaussian_c  # noqa: E402
from egfg.gdynamic import (  # noqa: E402
    compile_gaussian_filter,
    information_program,
    kalman_program,
    kalman_reference,
    stack,
    stack_obs,
)
from egfg.generators import gaussian_observations  # noqa: E402

HERE = Path(__file__).resolve().parent
T, TC = 200, 5
TOL = 1e-8
CFLAGS = ["-O3", "-march=native", "-ffp-contract=fast"]
LIGHT = dict(rules="minimal", seed=True, extractor="greedy", strategy="bfs")
MODELS = [("vec", 2, 4), ("vec", 4, 4), ("vec", 8, 16), ("blocks", 2, 4), ("blocks", 4, 4), ("blocks", 8, 4),
          ("coupled", 3, 4), ("coupled", 4, 4), ("coupled", 6, 4)]


def write_problem(model, obs, path: Path) -> None:
    names = sorted(model.dims)
    bi = {n: i for i, n in enumerate(names)}
    out = [str(len(names)), " ".join(str(model.dims[n]) for n in names)]
    facs = model.initial + model.transition + model.observation
    out.append(str(len(facs)))
    num = lambda a: " ".join(repr(float(x)) for x in np.asarray(a).ravel())  # noqa: E731
    for f in facs:
        kind = {"prior": 0, "cond": 1, "obs": 2}[f.kind]
        child = bi[split(f.child)[0]] if f.child else -1
        ps = " ".join(f"{bi[split(p)[0]]} {split(p)[1]}" for p in f.parents)
        out.append(f"{kind} {child} {len(f.parents)} {ps} {len(f.b)}")
        out.append(" ".join([num(A) for A in f.A] + [num(f.b), num(f.Q)]))
    out.append(str(len(obs)))
    out += [" ".join(num(y) for y in ys) for ys in obs]
    out.append(str(TC))
    path.write_text("\n".join(out) + "\n")


def run_exe(cmd) -> tuple[np.ndarray, float]:
    lines = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return np.array([float(x) for x in lines[0].split()]), float(lines[1])


def check(model, obs, flat: np.ndarray, stacked: bool) -> float:
    """Largest relative error of the means and covariances against the reference Kalman filter."""
    ref = kalman_reference(model, obs[:TC])
    names = sorted(model.dims)
    err = 0.0
    per = len(flat) // TC
    N = sum(model.dims.values())
    for t, r in enumerate(ref):
        o = flat[t * per:(t + 1) * per]
        if stacked:
            mu, S = o[:N], o[N:].reshape(N, N)
            off = np.cumsum([0] + [model.dims[n] for n in names])
            means = [mu[off[i]:off[i + 1]] for i in range(len(names))]
            covs = [S[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(names))]
        else:
            k, means, covs = 0, [], []
            for n in names:
                means.append(o[k:k + model.dims[n]])
                k += model.dims[n]
            for n in names:
                d = model.dims[n]
                covs.append(o[k:k + d * d].reshape(d, d))
                k += d * d
        for n, m, c in zip(names, means, covs):
            scale = 1 + max(np.max(np.abs(r.mean[n])), np.max(np.abs(r.cov[n])))
            err = max(err, float(np.max(np.abs(m - r.mean[n]))) / scale, float(np.max(np.abs(c - r.cov[n]))) / scale)
    return err


def c_program(prog, data: Path, work: Path, tag: str) -> dict:
    src = work / f"{tag}.c"
    src.write_text(generate_gaussian_c(prog))
    exe = work / tag
    subprocess.run(["gcc", *CFLAGS, "-std=c11", str(src), str(HERE / "driver.c"), "-o", str(exe), "-lm"], check=True)
    flat, t = run_exe([str(exe), str(data)])
    return {"flat": flat, "time_s": t}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtsam", required=True, type=Path)
    ap.add_argument("--out")
    ap.add_argument("--models", nargs="*", help="e.g. blocks:4:4 (default: all)")
    args = ap.parse_args()
    models = [tuple([m.split(":")[0]] + [int(x) for x in m.split(":")[1:]]) for m in args.models] if args.models else MODELS
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        gexe = work / "gtsam_filter"
        prefix = args.gtsam
        subprocess.run(["g++", *CFLAGS, "-std=c++17", str(HERE / "gtsam_filter.cpp"), "-o", str(gexe),
                        f"-I{prefix / 'include'}", f"-I{prefix / 'include' / 'gtsam' / '3rdparty' / 'Eigen'}",
                        f"-L{prefix / 'lib'}", "-Wl,--disable-new-dtags", f"-Wl,-rpath,{prefix / 'lib'}", "-lgtsam"],
                       check=True)
        for kind, p1, p2 in models:
            name = f"{kind}({p1},{p2})"
            model = getattr(dgm, kind)(p1, p2)
            obs = gaussian_observations(model, T)
            data, sdata = work / f"{kind}_{p1}_{p2}.txt", work / f"{kind}_{p1}_{p2}_stacked.txt"
            write_problem(model, obs, data)
            write_problem(stack(model), stack_obs(obs), sdata)
            row = {"model": name}
            t0 = time.perf_counter()
            prog = compile_gaussian_filter(model, reps="both", amortize_constants=True, **LIGHT)
            row["egfg_compile_s"] = round(time.perf_counter() - t0, 2)
            row["egfg_flops"] = prog.step.cost
            progs = {"egfg_c": (prog, data, False),
                     "kf_c": (kalman_program(model, amortize_constants=True), sdata, True),
                     "if_c": (information_program(model, amortize_constants=True), sdata, True)}
            for tag, (pr, d, stacked) in progs.items():
                r = c_program(pr, d, work, tag)
                row[tag] = {"time_s": r["time_s"], "max_rel_err": check(model, obs, r["flat"], stacked),
                            "flops": pr.step.cost, "ops": pr.step.counts}
            for tag in ("kf", "fg"):
                flat, t = run_exe([str(gexe), str(data), tag])
                row[f"gtsam_{tag}"] = {"time_s": t, "max_rel_err": check(model, obs, flat, False)}
            for k, v in row.items():
                if isinstance(v, dict):
                    v["status"] = "ok" if v["max_rel_err"] <= TOL else "wrong"
            results.append(row)
            print(name, {k: (v["status"], round(v["time_s"] * 1e9)) for k, v in row.items() if isinstance(v, dict)},
                  "(ns/step)", f"search {row['egfg_compile_s']}s", flush=True)
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))  # noqa: E731
    summary = {f"{a}/egfg_c": gm([r[a]["time_s"] / r["egfg_c"]["time_s"] for r in results])
               for a in ("kf_c", "if_c", "gtsam_kf", "gtsam_fg")}
    print(json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps({"T": T, "models": results, "summary": summary}, indent=1, default=str))


if __name__ == "__main__":
    main()

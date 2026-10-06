"""Existing libraries on the public problems: GTSAM (discrete) and pyAgrum (Shafer-Shenoy).

Usage:
  uv run --group libs python arena/score_libs.py --seed 401 --against arena/results/round2b_c_seed401.json \
      --out arena/results/libs_seed401.json
Both libraries are called from Python, so every number includes the cost of the bindings
(`binding_floor_s` is the time of a trivial call). The graph is built once; only inference
(all marginals) is timed. For a fair timing, see arena/gtsam_bench (C++).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egfg.arena import draw_tables, load, reference_marginals  # noqa: E402

N_DRAWS = 3
N_TIMING = 7
TOL = 1e-6


def _median_time(f, repeats: int = N_TIMING, min_seconds: float = 0.005) -> float:
    f()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        if time.perf_counter() - t0 >= min_seconds:
            break
        n *= 4
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        samples.append((time.perf_counter() - t0) / n)
    return statistics.median(samples)


def gtsam_solver(problem, tables):
    """Returns (infer, noop): infer() gives all marginals of a prebuilt DiscreteFactorGraph."""
    import gtsam

    names = sorted(problem["variables"])
    key = {v: i for i, v in enumerate(names)}
    g = gtsam.DiscreteFactorGraph()
    for f in problem["factors"]:
        # the table is row-major in scope order (the last key varies fastest)
        g.add([(key[v], problem["variables"][v]) for v in f["scope"]], list(np.asarray(tables[f["id"]]).ravel()))
    keys = {v: (key[v], problem["variables"][v]) for v in names}

    def infer():
        m = gtsam.DiscreteMarginals(g)
        return {v: np.asarray(m.marginalProbabilities(k)) for v, k in keys.items()}

    return infer, g.size


def agrum_solver(problem, tables):
    import pyagrum as gum

    mrf = gum.MarkovRandomField()
    for v, k in problem["variables"].items():
        mrf.add(gum.RangeVariable(v, v, 0, k - 1))
    for f in problem["factors"]:
        # pyAgrum enumerates with the first variable varying fastest
        t = np.asarray(tables[f["id"]]).transpose(range(len(f["scope"]) - 1, -1, -1))
        mrf.addFactor(f["scope"]).fillWith(list(t.ravel()))

    def infer():
        ie = gum.ShaferShenoyMRFInference(mrf)  # a fresh engine: makeInference caches its result
        ie.makeInference()
        return {v: np.asarray(ie.posterior(v).toarray()) for v in problem["variables"]}

    return infer, mrf.size


SOLVERS = {"gtsam_py": gtsam_solver, "pyagrum_py": agrum_solver}


def score(problem, make, seeds) -> dict:
    err, t, floor = 0.0, None, None
    for k, seed in enumerate(seeds):
        tables = draw_tables(problem, seed)
        infer, noop = make(problem, tables)
        out = infer()
        if k == 0:
            t, floor = _median_time(infer), _median_time(noop)
        for v, r in reference_marginals(problem, tables).items():
            err = max(err, float(np.max(np.abs(out[v] - r))))
    return {"status": "ok" if err <= TOL else "wrong", "max_abs_err": err, "time_s": t, "binding_floor_s": floor}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(ROOT / "arena" / "problems"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--against", help="a score.py result whose programs are shown alongside")
    ap.add_argument("--out")
    args = ap.parse_args()
    seeds = [args.seed + k for k in range(N_DRAWS)]
    other = json.loads(Path(args.against).read_text())["problems"] if args.against else {}
    results: dict = {"seeds": seeds, "problems": {}}
    for prob in sorted(Path(args.problems).glob("*.json")):
        problem = load(prob)
        row = {name: score(problem, make, seeds) for name, make in SOLVERS.items()}
        for label, r in other.get(prob.stem, {}).items():
            row[label] = r
        results["problems"][prob.stem] = row
        print(prob.stem, {k: (v["status"], round(v.get("time_s", float("nan")) * 1e6, 3)) for k, v in row.items()},
              "(us)", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

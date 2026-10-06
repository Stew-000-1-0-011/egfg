"""GTSAM in C++ against egfg's C programs, on the public problems and a few larger ones.

Usage:
  uv run python arena/gtsam_bench/run.py --gtsam <install prefix> --seed 401 \
      --against arena/results/round2b_c_seed401.json --out arena/results/gtsam_cpp_seed401.json
GTSAM is built from source (the Python wheel has no headers), e.g. 4.3.0 with
-DGTSAM_USE_BOOST_FEATURES=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON, Release.
Only inference is timed (the graph is built once); `time_s` gives all marginals,
`eliminate_s` the multifrontal elimination alone, `eliminate_fixed_order_s` the same with
the ordering computed beforehand. For the larger problems, which have no
LLM programs, egfg's C program is generated here and timed the same way as in score.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from egfg import arena  # noqa: E402

HERE = Path(__file__).resolve().parent
TOL = 1e-6


def larger_problems() -> list[dict]:
    rng = np.random.default_rng(5)
    p = arena._pairwise_spec
    return [
        p("big_chain32_k32", 32, 32, [(i, i + 1) for i in range(31)], unary=True),
        p("big_cycle16_k16", 16, 16, [(i, (i + 1) % 16) for i in range(16)]),
        p("big_tree30_k8", 30, 8, arena._tree_edges(30, rng)),
        p("big_grid4x4_k3", 16, 3, arena._grid_edges(4, 4)),
        p("big_lowrank_cycle12_k32_r2", 12, 32, [(i, (i + 1) % 12) for i in range(12)],
          {"kind": "lowrank", "rank": 2}),
    ]


def build(prefix: Path, out: Path) -> Path:
    exe = out / "bench"
    subprocess.run(["g++", "-O3", "-march=native", "-std=c++17", str(HERE / "bench.cpp"), "-o", str(exe),
                    f"-I{prefix / 'include'}", f"-I{prefix / 'include' / 'gtsam' / '3rdparty' / 'Eigen'}",
                    f"-L{prefix / 'lib'}", "-Wl,--disable-new-dtags", f"-Wl,-rpath,{prefix / 'lib'}",
                    "-lgtsam"], check=True)
    return exe


def run_gtsam(exe: Path, problem: dict, tables, work: Path) -> dict:
    names = sorted(problem["variables"])
    idx = {v: i for i, v in enumerate(names)}
    ref = arena.reference_marginals(problem, tables)
    lines = [str(len(names)), " ".join(str(problem["variables"][v]) for v in names), str(len(problem["factors"]))]
    for f in problem["factors"]:
        lines.append(" ".join(map(str, [len(f["scope"])] + [idx[v] for v in f["scope"]])))
        lines.append(" ".join(repr(float(x)) for x in np.asarray(tables[f["id"]]).ravel()))
    lines.append(" ".join(repr(float(x)) for v in names for x in ref[v]))
    data = work / f"{problem['name']}.txt"
    data.write_text("\n".join(lines) + "\n")
    r = json.loads(subprocess.run([str(exe), str(data)], capture_output=True, text=True, check=True).stdout)
    r.pop("sink")
    r["status"] = "ok" if r["max_abs_err"] <= TOL else "wrong"
    return r


def run_egfg(problem: dict, seed: int) -> dict:
    from egfg.ccodegen import compile_c_program

    src, info = arena.egfg_c_program(problem)
    tables = arena.draw_tables(problem, seed)
    run = compile_c_program(src, list(problem["variables"]), problem["variables"])
    out = run(tables)
    err = max(float(np.max(np.abs(out[v] - r))) for v, r in arena.reference_marginals(problem, tables).items())
    return {"status": "ok" if err <= TOL else "wrong", "max_abs_err": err,
            "time_s": arena.time_c(run, tables), "compile_s": info["compile_s"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtsam", required=True, type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--against", help="a score.py result (same seed) shown alongside the public problems")
    ap.add_argument("--out")
    args = ap.parse_args()
    other = json.loads(Path(args.against).read_text())["problems"] if args.against else {}
    results: dict = {"seed": args.seed, "problems": {}}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        exe = build(args.gtsam, work)
        public = [arena.load(p) for p in sorted((ROOT / "arena" / "problems").glob("*.json"))]
        for problem in public + larger_problems():
            tables = arena.draw_tables(problem, args.seed)
            row = {"gtsam_cpp": run_gtsam(exe, problem, tables, work)}
            if problem["name"] in other:
                row.update(other[problem["name"]])
            else:
                row["egfg_c"] = run_egfg(problem, args.seed)
            results["problems"][problem["name"]] = row
            print(problem["name"], {k: (v["status"], round(v["time_s"] * 1e6, 3)) for k, v in row.items()},
                  "(us)", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

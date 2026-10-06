"""Score programs on fresh tables: correctness against the reference, and wall-clock time.

Usage:
  uv run python arena/score.py --solutions arena/solutions/egfg arena/solutions/llm --out arena/results/r1.json
Each solutions directory holds <problem>.py modules defining infer(tables). Programs other than
egfg's may import only numpy and a few standard modules (checked before running).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALLOWED_IMPORTS = {"numpy", "math", "itertools", "functools", "collections", "operator"}
TIME_LIMIT_S = 60
MEMORY_BYTES = 4 * 1024**3
N_DRAWS = 3
N_TIMING = 7
TOL = 1e-6


def check_imports(path: Path) -> str | None:
    """None if the module imports only allowed modules, else the offending name."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for n in names:
            if n.split(".")[0] not in ALLOWED_IMPORTS:
                return n
    return None


def _limit():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))


def worker(args: dict) -> dict:
    """Run one program on one problem (in a child process)."""
    import importlib.util

    import numpy as np

    from egfg.arena import draw_tables, load, reference_marginals

    problem = load(args["problem"])
    spec = importlib.util.spec_from_file_location("solution", args["program"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    err = 0.0
    times = []
    for k, seed in enumerate(args["seeds"]):
        tables = draw_tables(problem, seed)
        if k == 0:
            mod.infer({i: t.copy() for i, t in tables.items()})  # warm-up
            for _ in range(N_TIMING):
                t0 = time.perf_counter()
                out = mod.infer({i: t.copy() for i, t in tables.items()})
                times.append(time.perf_counter() - t0)
        else:
            out = mod.infer({i: t.copy() for i, t in tables.items()})
        ref = reference_marginals(problem, tables)
        for v, r in ref.items():
            got = np.asarray(out.get(v), float) if v in out else None
            if got is None or got.shape != r.shape:
                return {"status": "wrong", "detail": f"missing or misshaped marginal for {v}"}
            err = max(err, float(np.max(np.abs(got - r))))
    return {"status": "ok" if err <= TOL else "wrong", "max_abs_err": err, "time_s": statistics.median(times)}


def run(program: Path, problem: Path, seeds: list[int], check: bool) -> dict:
    if check:
        bad = check_imports(program)
        if bad:
            return {"status": "rejected", "detail": f"imports {bad!r}"}
    spec = json.dumps({"program": str(program), "problem": str(problem), "seeds": seeds})
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", spec],
                              capture_output=True, text=True, timeout=TIME_LIMIT_S * (N_TIMING + N_DRAWS + 1),
                              cwd=ROOT, preexec_fn=_limit)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if proc.returncode != 0:
        return {"status": "error", "detail": (proc.stderr.strip().splitlines() or ["?"])[-1][:300]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", nargs="+", default=[])
    ap.add_argument("--problems", default=str(ROOT / "arena" / "problems"))
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=None, help="seed for the table draws (default: from the clock)")
    ap.add_argument("--worker")
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))))
        return
    base = args.seed if args.seed is not None else int(time.time())
    seeds = [base + k for k in range(N_DRAWS)]
    problems = sorted(Path(args.problems).glob("*.json"))
    results: dict = {"seeds": seeds, "problems": {}}
    for prob in problems:
        name = prob.stem
        row = {}
        for sol in args.solutions:
            d = Path(sol)
            prog = d / f"{name}.py"
            label = d.name
            row[label] = run(prog, prob, seeds, check=(label != "egfg")) if prog.exists() else {"status": "missing"}
        results["problems"][name] = row
        print(name, {k: (v["status"], round(v.get("time_s", float("nan")) * 1e3, 3)) for k, v in row.items()},
              flush=True)
    labels = [Path(s).name for s in args.solutions]
    if len(labels) == 2:
        a, b = labels
        ratios, wins = [], {a: 0, b: 0, "tie": 0}
        for name, row in results["problems"].items():
            ra, rb = row[a], row[b]
            oka, okb = ra.get("status") == "ok", rb.get("status") == "ok"
            if oka and okb:
                ratios.append(ra["time_s"] / rb["time_s"])
                wins[a if ra["time_s"] < rb["time_s"] else b] += 1
            elif oka:
                wins[a] += 1
            elif okb:
                wins[b] += 1
            else:
                wins["tie"] += 1
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else float("nan")
        results["summary"] = {"time_ratio_geomean": gm, "ratio": f"{a}/{b}", "wins": wins}
        print(json.dumps(results["summary"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

"""Gate for egfg improvements: score egfg on holdout problems it has never seen.

The score is deterministic: for each holdout problem, egfg's cost-model operations divided by
the best known junction tree (best-JT+), plus a correctness check of the generated program on
fresh tables. Usage: uv run python arena/holdout_gate.py --seed 7 [--out arena/results/gate.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def gate(seed: int) -> dict:
    import numpy as np

    from egfg.arena import draw_tables, egfg_program, holdout_problems, reference_marginals, to_factor_graph
    from egfg.baselines import best_junction_tree
    from egfg.codegen import compile_program

    rows = {}
    for p in holdout_problems(seed):
        src, info = egfg_program(p)
        infer = compile_program(src)
        tables = draw_tables(p, seed + 1000)
        ref = reference_marginals(p, tables)
        out = infer(tables)
        err = max(float(np.max(np.abs(out[v] - ref[v]))) for v in ref)
        bj = best_junction_tree(to_factor_graph(p, tables), share_products=True)[1]
        rows[p["name"]] = {"flops": info["flops"], "best_jtp": bj, "ratio": info["flops"] / bj,
                           "correct": err < 1e-6, "compile_s": info["compile_s"]}
        print(p["name"], rows[p["name"]], flush=True)
    gm = math.exp(sum(math.log(r["ratio"]) for r in rows.values()) / len(rows))
    return {"seed": seed, "geomean_ratio": gm, "all_correct": all(r["correct"] for r in rows.values()),
            "problems": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = gate(a.seed)
    print(json.dumps({k: v for k, v in res.items() if k != "problems"}))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))

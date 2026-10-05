import csv
import subprocess
import sys
from pathlib import Path

COLUMNS = (
    "family,n,K,seed,num_vars,num_factors,egraph_nodes,iterations,hit_limit,saturate_s,"
    "tree_cost,ilp_cost,ilp_optimal,ilp_s,jt_cost,opt_einsum_cost,"
    "ilp_max_intermediate,jt_max_intermediate,jt_messages_covered,bp_cost,bp_messages_covered"
).split(",")


def test_quick_run(tmp_path: Path):
    out = tmp_path / "r.csv"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "experiments" / "run.py"), "--out", str(out), "--quick"],
        check=True,
        cwd=root,
    )
    rows = list(csv.DictReader(out.open()))
    assert rows and list(rows[0].keys()) == COLUMNS
    assert all(int(r["ilp_cost"]) <= int(r["tree_cost"]) for r in rows)
    assert {r["family"] for r in rows} >= {"chain", "star", "random_tree", "cycle", "grid"}
    assert all(0.0 <= float(r["jt_messages_covered"]) <= 1.0 for r in rows)
    trees = [r for r in rows if r["family"] in ("chain", "star", "random_tree")]
    assert all(r["bp_cost"] != "" for r in trees)
    assert all(r["bp_cost"] == "" for r in rows if r["family"] in ("cycle", "grid"))

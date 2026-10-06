import csv
import subprocess
import sys
from pathlib import Path

COLUMNS = (
    "config,family,n,seed,K,num_vars,num_factors,rules,cluster_budget,seed_jt,extractor,"
    "status,wall,wall_reason,hit_limit,saturate_s,extract_s,total_s,"
    "egraph_nodes,max_cluster_nodes,num_clusters,max_cluster_vars,"
    "cost,optimal,jt_cost,cost_vs_jt,bp_cost,cost_vs_bp,base_cost,cost_vs_base,"
    "correct,checked_against,max_abs_err,error"
).split(",")


def test_quick_scaling_run(tmp_path: Path):
    out = tmp_path / "s.csv"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "experiments" / "scaling.py"), "--out", str(out), "--quick"],
        check=True,
        cwd=root,
    )
    rows = list(csv.DictReader(out.open()))
    assert rows and list(rows[0].keys()) == COLUMNS
    assert all(r["status"] == "ok" and r["correct"] == "True" for r in rows)
    assert {r["config"] for r in rows} >= {"base", "D=3", "S=on", "X=greedy", "light"}
    assert all(r["cost_vs_base"] != "" for r in rows)
    seeded = [r for r in rows if r["seed_jt"] == "True"]
    assert seeded and all(int(r["cost"]) <= int(r["jt_cost"]) for r in seeded)

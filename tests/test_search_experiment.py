import csv
import subprocess
import sys
from pathlib import Path

COLUMNS = (
    "case,family,n,rules,node_limit,strategy,status,hit_limit,steps,nodes,search_s,extract_s,cost,"
    "jt_cost,best_jt_cost,best_jtp_cost,cost_vs_jt,cost_vs_best_jtp,correct,max_abs_err,error"
).split(",")


def test_quick_search_order_run(tmp_path: Path):
    out = tmp_path / "s.csv"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "experiments" / "search_order.py"), "--out", str(out), "--quick"],
        check=True,
        cwd=root,
    )
    rows = list(csv.DictReader(out.open()))
    assert rows and list(rows[0].keys()) == COLUMNS
    assert all(r["status"] == "ok" and r["correct"] == "True" for r in rows)
    assert {r["strategy"] for r in rows} == {"bfs", "staged", "seeds", "restart"}
    assert all(int(r["cost"]) <= int(r["best_jt_cost"]) for r in rows if r["strategy"] == "seeds")
    assert (tmp_path / "s_curve.csv").exists()

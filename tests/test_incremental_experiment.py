import csv
import subprocess
import sys
from pathlib import Path

COLUMNS = (
    "model,m,K,search,objective,status,wall,search_s,step_nodes,hit_limit,"
    "head_prep,head_latency,head_cost,step_prep,step_latency,step_cost,"
    "fwd_step_prep,fwd_step_latency,fwd_step_cost,latency_vs_forward,cost_vs_forward,"
    "latency_vs_total,cost_vs_total,prepare_ms,update_ms,T,checked_times,checked_against,correct,max_abs_err,error"
).split(",")


def test_quick_incremental_run(tmp_path: Path):
    out = tmp_path / "i.csv"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "experiments" / "incremental.py"), "--out", str(out), "--quick"],
        check=True,
        cwd=root,
    )
    rows = list(csv.DictReader(out.open()))
    assert rows and list(rows[0].keys()) == COLUMNS
    assert all(r["status"] == "ok" and r["correct"] == "True" for r in rows)
    assert {r["objective"] for r in rows} == {"forward", "jt", "total", "latency"}
    for r in rows:
        assert int(r["step_prep"]) + int(r["step_latency"]) == int(r["step_cost"])
        if r["objective"] == "latency":
            assert float(r["latency_vs_forward"]) <= 1.0 and float(r["latency_vs_total"]) <= 1.0

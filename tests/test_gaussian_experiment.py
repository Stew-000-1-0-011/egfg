import csv
import subprocess
import sys
from pathlib import Path

COLUMNS = (
    "model,p1,p2,state_dim,obs_dim,setting,reps,objective,amortize,status,wall,search_s,step_nodes,hit_limit,"
    "fwd_rep,head_cost,step_cost,step_prep,step_latency,kf_step_cost,if_step_cost,step_vs_kf,step_vs_if,"
    "step_vs_best_baseline,ops,correct,max_rel_err,error"
).split(",")


def test_quick_gaussian_run(tmp_path: Path):
    out = tmp_path / "g.csv"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "experiments" / "gaussian.py"), "--out", str(out), "--quick"],
        check=True,
        cwd=root,
    )
    rows = list(csv.DictReader(out.open()))
    assert rows and list(rows[0].keys()) == COLUMNS
    assert all(r["status"] == "ok" and r["correct"] == "True" for r in rows)
    assert {r["setting"] for r in rows} == {"KF", "IF", "light"}
    for r in rows:
        assert int(r["step_prep"]) + int(r["step_latency"]) == int(r["step_cost"])
        if r["setting"] == "light" and r["reps"] == "both" and r["objective"] == "total":
            assert float(r["step_vs_best_baseline"]) <= 1.0

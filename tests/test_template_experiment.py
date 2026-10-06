import csv
import subprocess
import sys
from pathlib import Path

COLUMNS = (
    "model,m,K,setting,T,status,wall,search_s,head_saturate_s,head_extract_s,step_saturate_s,step_extract_s,"
    "head_nodes,step_nodes,hit_limit,head_cost,step_cost,total_cost,forward_step_cost,step_vs_forward,"
    "jt_step_cost,step_vs_jt,total_vs_forward,eval_s,eval_s_per_step,checked_times,checked_against,"
    "correct,max_abs_err,error"
).split(",")


def test_quick_template_run(tmp_path: Path):
    out = tmp_path / "t.csv"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "experiments" / "template.py"), "--out", str(out), "--quick"],
        check=True,
        cwd=root,
    )
    rows = list(csv.DictReader(out.open()))
    assert rows and list(rows[0].keys()) == COLUMNS
    assert all(r["status"] == "ok" and r["correct"] == "True" for r in rows)
    assert {r["setting"] for r in rows} == {"forward", "default", "light"}
    # the search is done once per model and setting: the same time for every T
    for key in {(r["model"], r["setting"]) for r in rows}:
        assert len({r["search_s"] for r in rows if (r["model"], r["setting"]) == key}) == 1
    assert all(float(r["step_vs_forward"]) <= 1.0 for r in rows if r["setting"] == "default")

"""A/B gate for egfg improvements on the holdout problems (C programs, wall-clock).

  # compile the holdout problems with the current egfg into arena/holdout_programs/<tag>/
  uv run python arena/holdout_gate_c.py compile --seed 7 --tag round2
  # (optionally with another checkout of egfg: --src /path/to/other/checkout/src)
  # score two tags against each other in the same run
  uv run python arena/holdout_gate_c.py compare --seed 7 --old round1 --new round2 [--out ...]

An improvement is accepted when every new program is correct and the geometric mean of
new/old time is not worse than 1.0 (both measured in the same run).
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "arena" / "holdout_programs"


def compile_tag(seed: int, tag: str, src: str | None) -> None:
    sys.path.insert(0, src or str(ROOT / "src"))
    from egfg.arena import egfg_c_program, holdout_problems, save

    out = BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    for p in holdout_problems(seed):
        code, info = egfg_c_program(p)
        (out / f"{p['name']}.c").write_text(code)
        print(tag, p["name"], {k: info[k] for k in ("flops", "overhead")}, flush=True)


def compare(seed: int, old: str, new: str, out: str | None) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from egfg.arena import holdout_problems, save

    probs = Path(tempfile.mkdtemp(prefix="holdout_"))
    for p in holdout_problems(seed):
        save(p, probs)
    # score.py treats directories named egfg* as trusted; copy into neutral names would check
    # headers too, which egfg programs pass anyway
    res_path = probs / "scores.json"
    subprocess.run([sys.executable, str(ROOT / "arena" / "score.py"), "--solutions", str(BASE / old), str(BASE / new),
                    "--problems", str(probs), "--out", str(res_path), "--seed", str(seed + 99)], check=True)
    res = json.loads(res_path.read_text())
    ratios, all_ok = [], True
    for name, row in res["problems"].items():
        a, b = row[old], row[new]
        all_ok &= b.get("status") == "ok"
        if a.get("status") == "ok" and b.get("status") == "ok":
            ratios.append(b["time_s"] / a["time_s"])
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    summary = {"seed": seed, "old": old, "new": new, "new_over_old_geomean": gm, "all_new_correct": all_ok,
               "accept": all_ok and gm <= 1.0}
    print(json.dumps(summary))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({**summary, "problems": res["problems"]}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compile", "compare"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag")
    ap.add_argument("--src")
    ap.add_argument("--old")
    ap.add_argument("--new")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "compile":
        compile_tag(a.seed, a.tag, a.src)
    else:
        compare(a.seed, a.old, a.new, a.out)

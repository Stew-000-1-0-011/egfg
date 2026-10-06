"""Compile every public problem to C with egfg (arena/solutions/egfg_c/).

Usage: uv run python arena/solve_egfg_c.py [--problems arena/problems] [--out arena/solutions/egfg_c]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egfg.arena import egfg_c_program, load  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(ROOT / "arena" / "problems"))
    ap.add_argument("--out", default=str(ROOT / "arena" / "solutions" / "egfg_c"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    infos = {}
    for path in sorted(Path(args.problems).glob("*.json")):
        p = load(path)
        src, info = egfg_c_program(p)
        (out / f"{p['name']}.c").write_text(src)
        infos[p["name"]] = info
        print(p["name"], {k: info[k] for k in ("compile_s", "flops", "overhead")}, flush=True)
    (out / "info.json").write_text(json.dumps(infos, indent=1))

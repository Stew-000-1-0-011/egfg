"""Compile every public problem with egfg and write the generated programs to arena/solutions/egfg/.

Usage: uv run python arena/solve_egfg.py [--problems arena/problems] [--out arena/solutions/egfg]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egfg.arena import egfg_program, load  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(ROOT / "arena" / "problems"))
    ap.add_argument("--out", default=str(ROOT / "arena" / "solutions" / "egfg"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    infos = {}
    for path in sorted(Path(args.problems).glob("*.json")):
        p = load(path)
        src, info = egfg_program(p)
        (out / f"{p['name']}.py").write_text(src)
        infos[p["name"]] = info
        print(p["name"], info, flush=True)
    (out / "info.json").write_text(json.dumps(infos, indent=1))

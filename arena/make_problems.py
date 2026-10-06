"""Write the public problem set to arena/problems/ (structures only, no table values)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egfg.arena import public_problems, save  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "arena" / "problems"
    for p in public_problems():
        print(save(p, out).name)

"""An opt-in on-disk cache of compiled C programs and saturations.

Set `EGFG_CACHE` to a directory to turn it on (off by default, so tests always compute).
Only the most recent `EGFG_CACHE_MAX` entries of each kind are kept (default 200).

Keys include everything the result depends on, and for saturations a hash of egfg's own
source, so a change to the code never returns a stale result. Saturations depend only on
the structure (variables, numbers of states, factor scopes), not on the tables. A cached
result is the first one computed: searches with time limits are then reproducible, but
fixed to that run.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path

_VERSION: str | None = None


def directory(kind: str) -> Path | None:
    root = os.environ.get("EGFG_CACHE")
    if not root:
        return None
    d = Path(root) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def code_version() -> str:
    """A hash of egfg's source files."""
    global _VERSION
    if _VERSION is None:
        h = hashlib.sha256()
        for f in sorted(Path(__file__).parent.glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        _VERSION = h.hexdigest()[:16]
    return _VERSION


def key(*parts) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _touch(p: Path) -> None:
    try:
        os.utime(p)
    except OSError:
        pass


def evict(d: Path) -> None:
    limit = int(os.environ.get("EGFG_CACHE_MAX", "200"))
    files = sorted((f for f in d.iterdir() if f.is_file() and not f.name.startswith(".")), key=lambda f: f.stat().st_mtime)
    for f in files[:-limit] if len(files) > limit else []:
        try:
            f.unlink()
        except OSError:
            pass


def path_for(kind: str, k: str, suffix: str) -> Path | None:
    d = directory(kind)
    return d / f"{k}{suffix}" if d else None


def store_file(kind: str, k: str, suffix: str, src: Path) -> Path:
    """Move `src` into the cache (atomically) and return its cached path."""
    d = directory(kind)
    dst = d / f"{k}{suffix}"
    tmp = d / f".{k}.{os.getpid()}{suffix}"
    os.replace(src, tmp)
    os.replace(tmp, dst)
    evict(d)
    return dst


def load(kind: str, k: str):
    p = path_for(kind, k, ".pkl")
    if p is None or not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
    except Exception:  # a broken entry is recomputed
        return None
    _touch(p)
    return obj


def save(kind: str, k: str, obj) -> None:
    d = directory(kind)
    if d is None:
        return
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, d / f"{k}.pkl")
    evict(d)

"""Factor graph generators for tests and experiments (pairwise factors, values in [0.1, 1))."""

from __future__ import annotations

import numpy as np

from .model import Factor, FactorGraph


def _pairwise(n: int, K: int, edges: list[tuple[int, int]], seed: int) -> FactorGraph:
    rng = np.random.default_rng(seed)
    cards = {f"x{i}": K for i in range(n)}
    factors = [
        Factor(fid, (f"x{i}", f"x{j}"), rng.uniform(0.1, 1.0, (K, K)))
        for fid, (i, j) in enumerate(edges)
    ]
    return FactorGraph(cards, factors)


def chain(n: int, K: int, seed: int = 0) -> FactorGraph:
    return _pairwise(n, K, [(i, i + 1) for i in range(n - 1)], seed)


def star(n: int, K: int, seed: int = 0) -> FactorGraph:
    return _pairwise(n, K, [(0, i) for i in range(1, n)], seed)


def random_tree(n: int, K: int, seed: int = 0) -> FactorGraph:
    rng = np.random.default_rng(seed + 10_000)
    edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
    return _pairwise(n, K, edges, seed)


def cycle(n: int, K: int, seed: int = 0) -> FactorGraph:
    return _pairwise(n, K, [(i, (i + 1) % n) for i in range(n)], seed)


def grid(rows: int, cols: int, K: int, seed: int = 0) -> FactorGraph:
    idx = lambda r, c: r * cols + c  # noqa: E731
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return _pairwise(rows * cols, K, edges, seed)


def random_small(seed: int) -> FactorGraph:
    """1-5 variables (cards 1-3), 1-5 factors of arity 1-3, every variable covered."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 6))
    names = [f"x{i}" for i in range(n)]
    cards = {v: int(rng.integers(1, 4)) for v in names}
    m = int(rng.integers(1, 6))
    scopes: list[tuple[str, ...]] = []
    for _ in range(m):
        k = int(rng.integers(1, min(3, n) + 1))
        scopes.append(tuple(str(v) for v in rng.choice(names, size=k, replace=False)))
    covered = {v for s in scopes for v in s}
    for i, v in enumerate(v for v in names if v not in covered):
        j = i % len(scopes)  # attach uncovered variables to existing factors
        if len(scopes[j]) < 3:
            scopes[j] = scopes[j] + (v,)
        else:
            scopes.append((v,))
    scopes = scopes[:5] if all(v in {u for s in scopes[:5] for u in s} for v in names) else scopes
    factors = [
        Factor(fid, s, rng.uniform(0.1, 1.0, tuple(cards[v] for v in s)))
        for fid, s in enumerate(scopes)
    ]
    return FactorGraph(cards, factors)


def random_sparse(n: int, K: int, seed: int = 0) -> FactorGraph:
    """A random tree plus n // 4 extra random edges (no self loops, no duplicates)."""
    rng = np.random.default_rng(seed + 20_000)
    edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
    present = {frozenset(e) for e in edges}
    extra = n // 4
    candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if frozenset((i, j)) not in present]
    for k in rng.permutation(len(candidates))[: min(extra, len(candidates))]:
        edges.append(candidates[int(k)])
    return _pairwise(n, K, edges, seed)

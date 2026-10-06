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


# ---------------------------------------------------------------------------
# dynamic models (scopes use relative time: @-1 previous step, @0 current step)
# ---------------------------------------------------------------------------


def _table(rng, shape) -> np.ndarray:
    return rng.uniform(0.1, 1.0, shape)


def hmm(K: int, seed: int = 0):
    from .dynamic import DynamicModel

    rng = np.random.default_rng(seed + 30_000)
    return DynamicModel(
        {"a": K}, [(("a@0",), _table(rng, (K,)))], [(("a@-1", "a@0"), _table(rng, (K, K)))], [("a@0",)]
    )


def factorial_hmm(m: int, K: int, seed: int = 0):
    """m independent chains; one observation factor over all chains."""
    from .dynamic import DynamicModel

    rng = np.random.default_rng(seed + 40_000)
    xs = [f"x{i}" for i in range(m)]
    return DynamicModel(
        {x: K for x in xs},
        [((f"{x}@0",), _table(rng, (K,))) for x in xs],
        [((f"{x}@-1", f"{x}@0"), _table(rng, (K, K))) for x in xs],
        [tuple(f"{x}@0" for x in xs)],
    )


def coupled_hmm(m: int, K: int, seed: int = 0):
    """m chains; chain i also depends on chain i-1 at the previous step; one observation per chain."""
    from .dynamic import DynamicModel

    rng = np.random.default_rng(seed + 50_000)
    xs = [f"x{i}" for i in range(m)]
    trans = [((f"{xs[0]}@-1", f"{xs[0]}@0"), _table(rng, (K, K)))]
    trans += [((f"{xs[i]}@-1", f"{xs[i - 1]}@-1", f"{xs[i]}@0"), _table(rng, (K, K, K))) for i in range(1, m)]
    return DynamicModel(
        {x: K for x in xs},
        [((f"{x}@0",), _table(rng, (K,))) for x in xs],
        trans,
        [(f"{x}@0",) for x in xs],
    )


def random_observations(model, T: int, seed: int = 0) -> list[list[np.ndarray]]:
    """Observation (likelihood) tables for T steps."""
    rng = np.random.default_rng(seed + 60_000)
    return [[_table(rng, model.obs_shape(k)) for k in range(len(model.observation))] for _ in range(T)]


# ---------------------------------------------------------------------------
# linear-Gaussian dynamic models
# ---------------------------------------------------------------------------


def gaussian_vec(d: int, m: int, seed: int = 0):
    """One d-dimensional state x; one m-dimensional observation."""
    from .dynamic_gaussian_models import vec

    return vec(d, m, seed)


def gaussian_blocks(k: int, b: int, seed: int = 0):
    """k independent b-dimensional blocks; one b-dimensional observation of all of them."""
    from .dynamic_gaussian_models import blocks

    return blocks(k, b, seed)


def gaussian_coupled(k: int, b: int, seed: int = 0):
    """k b-dimensional blocks, block i driven by itself and block i-1; one observation per block."""
    from .dynamic_gaussian_models import coupled

    return coupled(k, b, seed)


def gaussian_observations(model, T: int, seed: int = 0) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(seed + 70_000)
    return [[rng.normal(size=len(f.b)) for f in model.observation] for _ in range(T)]

"""Linear-Gaussian dynamic models for tests and experiments (see generators.gaussian_*)."""

from __future__ import annotations

import numpy as np

from .gaussian import GFactor
from .gdynamic import GaussianDynamicModel


def _spd(rng, n: int) -> np.ndarray:
    M = rng.normal(size=(n, n))
    return M @ M.T / n + 0.5 * np.eye(n)


def _stable(rng, n: int, radius: float = 0.9) -> np.ndarray:
    A = rng.normal(size=(n, n))
    return A * (radius / max(1e-9, np.max(np.abs(np.linalg.eigvals(A)))))


def _prior(rng, v: str, d: int) -> GFactor:
    return GFactor("prior", f"{v}@0", (), (), rng.normal(size=d), _spd(rng, d))


def vec(d: int, m: int, seed: int = 0) -> GaussianDynamicModel:
    rng = np.random.default_rng(seed + 80_000)
    return GaussianDynamicModel(
        {"x": d},
        [_prior(rng, "x", d)],
        [GFactor("cond", "x@0", ("x@-1",), (_stable(rng, d),), rng.normal(size=d) * 0.1, _spd(rng, d))],
        [GFactor("obs", None, ("x@0",), (rng.normal(size=(m, d)),), np.zeros(m), _spd(rng, m))],
    )


def blocks(k: int, b: int, seed: int = 0) -> GaussianDynamicModel:
    rng = np.random.default_rng(seed + 90_000)
    xs = [f"x{i}" for i in range(k)]
    return GaussianDynamicModel(
        {x: b for x in xs},
        [_prior(rng, x, b) for x in xs],
        [GFactor("cond", f"{x}@0", (f"{x}@-1",), (_stable(rng, b),), np.zeros(b), _spd(rng, b)) for x in xs],
        [GFactor("obs", None, tuple(f"{x}@0" for x in xs), tuple(rng.normal(size=(b, b)) for _ in xs),
                 np.zeros(b), _spd(rng, b))],
    )


def coupled(k: int, b: int, seed: int = 0) -> GaussianDynamicModel:
    rng = np.random.default_rng(seed + 100_000)
    xs = [f"x{i}" for i in range(k)]
    trans = [GFactor("cond", f"{xs[0]}@0", (f"{xs[0]}@-1",), (_stable(rng, b),), np.zeros(b), _spd(rng, b))]
    for i in range(1, k):
        trans.append(GFactor("cond", f"{xs[i]}@0", (f"{xs[i]}@-1", f"{xs[i - 1]}@-1"),
                             (_stable(rng, b, 0.6), _stable(rng, b, 0.3)), np.zeros(b), _spd(rng, b)))
    return GaussianDynamicModel(
        {x: b for x in xs},
        [_prior(rng, x, b) for x in xs],
        trans,
        [GFactor("obs", None, (f"{x}@0",), (rng.normal(size=(b, b)),), np.zeros(b), _spd(rng, b)) for x in xs],
    )

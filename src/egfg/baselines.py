"""Reference computations: brute force (this section) and comparison baselines."""

from __future__ import annotations

import numpy as np

from .model import FactorGraph



def brute_force_joint(fg: FactorGraph) -> np.ndarray:
    """Unnormalized joint; axes follow fg.variables()."""
    vs = fg.variables()
    operands = []
    for f in fg.factors:
        operands += [f.table, [vs.index(v) for v in f.scope]]
    return np.einsum(*operands, list(range(len(vs))))


def brute_force_marginals(fg: FactorGraph) -> dict[str, np.ndarray]:
    J = brute_force_joint(fg)
    Z = J.sum()
    vs = fg.variables()
    return {
        v: J.sum(axis=tuple(j for j in range(len(vs)) if j != i)) / Z for i, v in enumerate(vs)
    }


def brute_force_max_value(fg: FactorGraph) -> float:
    return float(brute_force_joint(fg).max())


def joint_value(fg: FactorGraph, assignment: dict[str, int]) -> float:
    val = 1.0
    for f in fg.factors:
        val *= float(f.table[tuple(assignment[v] for v in f.scope)])
    return val


def brute_force_moments(fg: FactorGraph, var: str, values: np.ndarray) -> tuple[float, float]:
    p = brute_force_marginals(fg)[var]
    E = float((p * values).sum())
    return E, float((p * values**2).sum() - E**2)

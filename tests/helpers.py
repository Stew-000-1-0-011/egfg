import numpy as np

from egfg.model import Factor, FactorGraph


def chain3() -> FactorGraph:
    """a(2) - b(3) - c(4) with factors f0(a,b), f1(b,c)."""
    rng = np.random.default_rng(123)
    return FactorGraph(
        {"a": 2, "b": 3, "c": 4},
        [
            Factor(0, ("a", "b"), rng.uniform(0.1, 1.0, (2, 3))),
            Factor(1, ("b", "c"), rng.uniform(0.1, 1.0, (3, 4))),
        ],
    )

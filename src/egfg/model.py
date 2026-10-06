"""Discrete factor graphs: variables with finite cardinalities and table factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Factor:
    """A non-negative table over `scope`; axis i of `table` is `scope[i]`."""

    id: int
    scope: tuple[str, ...]
    table: np.ndarray


class FactorGraph:
    def __init__(self, cards: dict[str, int], factors: list[Factor], allow_negative: bool = False):
        """`allow_negative` admits tables with negative entries (only for internal,
        sum-product-only graphs such as low-rank factorizations)."""
        self.cards = dict(cards)
        self.factors = list(factors)
        self._by_id: dict[int, Factor] = {}
        covered: set[str] = set()
        for f in self.factors:
            if f.id in self._by_id:
                raise ValueError(f"duplicate factor id {f.id}")
            self._by_id[f.id] = f
            if len(set(f.scope)) != len(f.scope):
                raise ValueError(f"factor {f.id} repeats a variable in its scope")
            for v in f.scope:
                if v not in self.cards:
                    raise ValueError(f"factor {f.id} uses unknown variable {v!r}")
            expected = tuple(self.cards[v] for v in f.scope)
            if tuple(f.table.shape) != expected:
                raise ValueError(f"factor {f.id} has shape {f.table.shape}, expected {expected}")
            if not allow_negative and np.any(f.table < 0):
                raise ValueError(f"factor {f.id} has negative entries")
            covered.update(f.scope)
        for v, k in self.cards.items():
            if k < 1:
                raise ValueError(f"variable {v!r} has cardinality {k} < 1")
        uncovered = set(self.cards) - covered
        if uncovered:
            raise ValueError(f"variables not in any factor: {sorted(uncovered)}")

    def factor(self, fid: int) -> Factor:
        return self._by_id[fid]

    def variables(self) -> list[str]:
        return sorted(self.cards)

    def size(self, vars: Iterable[str]) -> int:
        n = 1
        for v in vars:
            n *= self.cards[v]
        return n

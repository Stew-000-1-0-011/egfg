"""A cost model of the generated C code (plain `ccodegen.generate_c`), calibrated by measurement.

The Dag is split into the loop nests the code generator writes (same rules: `_fusion`,
`_refs`): a fused sum (one nest over its output, then its summed variables, accumulating the
product of its operands), a product (one nest writing the product of two operands), and the
output (normalizing the marginals). Per program the features are

    red       Σ iterations of the summing nests (each adds to one accumulator: a dependency chain,
              which gcc does not vectorize without -ffast-math)
    red_mul   Σ iterations × (operands − 1) of the summing nests (the multiplies before the add)
    mul       Σ iterations of the product nests (independent elements: vectorized)
    writes    Σ elements written (size of each nest's output)
    strided   Σ iterations × operands read with a stride other than 0 or 1 in the innermost loop
    nests     number of nests
    loops     Σ over loops of the number of times the loop is entered
    spill     elements of the intermediate tables beyond 256 KiB (a cache proxy)
    const     1 (the fixed cost of one call)

and the predicted time (ns) is their dot product with non-negative coefficients fitted on
measured run times (`experiments/cost_calibration.py`). The coefficients live in
`results/cost_calibration.json` (or the file named by EGFG_COST_CALIBRATION); without a
calibration, `load()` returns None and callers keep the operation count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .ccodegen import _fusion, _refs, _strides, _topo
from .ir import Dag, ENode, dag_scopes
from .model import FactorGraph

FEATURES = ("red", "red_mul", "mul", "writes", "strided", "nests", "loops", "spill", "const")
CACHE_ELEMENTS = 256 * 1024 // 8
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "results" / "cost_calibration.json"


def _size(vars_, cards) -> int:
    n = 1
    for v in vars_:
        n *= cards[v]
    return n


def features(nodes: dict[str, ENode], roots: dict[str, str], scopes: dict[str, frozenset[str]],
             fg: FactorGraph) -> dict[str, float]:
    """The features of the plain C code of the Dag given by `nodes`, `roots` and `scopes`."""
    cards = fg.cards
    dag = Dag(nodes, roots)
    order = _topo(dag)
    refs = _refs(dag, order)
    fused, skip = _fusion(dag, order, refs)
    f = dict.fromkeys(FEATURES, 0.0)
    f["const"] = 1.0
    strides: dict[str, dict[str, int]] = {}
    inter = 0
    for nid in order:
        if nid in skip:
            continue
        node = nodes[nid]
        if node.op == "leaf":
            strides[nid] = _strides(list(fg.factor(node.arg).scope), cards)
            continue
        out = sorted(scopes[nid])
        strides[nid] = _strides(out, cards)
        if node.op == "input":
            continue
        operands = fused.get(nid, node.children) if (nid in fused or node.op == "sum") else node.children
        inner = sorted(set().union(*(scopes[c] for c in operands)) - set(out)) if node.op == "sum" or nid in fused else []
        loops = out + inner
        n = _size(loops, cards)
        if node.op == "sum":
            f["red"] += n
            f["red_mul"] += n * (len(operands) - 1)
        else:
            f["mul"] += n
        w = _size(out, cards)
        f["writes"] += w
        inter += w
        if loops:
            last = loops[-1]
            f["strided"] += n * sum(1 for c in operands if strides[c].get(last, 0) > 1)
        f["nests"] += 1
        f["loops"] += sum(_size(loops[:d], cards) for d in range(len(loops)))
    # the output: the normalizer and one loop per marginal (other roots, such as the messages of
    # a cluster, are tables already counted above)
    marg = sorted((v, nid) for v, nid in roots.items() if v in cards)
    if marg:
        f["red"] += _size(scopes[marg[0][1]], cards)
        f["mul"] += sum(cards[v] for v, _ in marg)
        f["writes"] += sum(cards[v] for v, _ in marg)
        f["nests"] += 1 + len(marg)
    f["spill"] = max(0, inter - CACHE_ELEMENTS)
    return f


def dag_features(dag: Dag, fg: FactorGraph) -> dict[str, float]:
    return features(dag.nodes, dag.roots, dag_scopes(dag, fg), fg)


@dataclass
class CostModel:
    coef: dict[str, float]  # ns per unit of each feature

    def predict(self, f: dict[str, float]) -> float:
        return sum(self.coef.get(k, 0.0) * f[k] for k in FEATURES)

    def dag_ns(self, dag: Dag, fg: FactorGraph) -> float:
        return self.predict(dag_features(dag, fg))

    def node_ns(self, n: ENode, own: frozenset[str], children: list[frozenset[str]], fg: FactorGraph) -> float:
        """One node on its own (no fusion): the per-node cost for tree and ILP extraction."""
        if n.op in ("leaf", "input"):
            return 0.0
        cards = fg.cards
        out = sorted(own)
        loops = out + (sorted(children[0] - own) if n.op == "sum" else [])
        f = dict.fromkeys(FEATURES, 0.0)
        if n.op == "sum":
            f["red"] = _size(loops, cards)
        else:
            f["mul"] = _size(loops, cards)
        f["writes"] = _size(out, cards)
        f["nests"] = 1
        f["loops"] = sum(_size(loops[:d], cards) for d in range(len(loops)))
        return self.predict(f)

    def to_json(self) -> str:
        return json.dumps({"features": list(FEATURES), "coef": self.coef}, indent=1)


def load(path: str | Path | None = None) -> CostModel | None:
    """The calibrated model (EGFG_COST_CALIBRATION, else results/cost_calibration.json), or None."""
    p = Path(path or os.environ.get("EGFG_COST_CALIBRATION") or DEFAULT_PATH)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return CostModel({k: float(v) for k, v in data["coef"].items()})

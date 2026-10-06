"""Shared pieces of the arena: problem specs, table generation, references, scoring helpers.

A problem spec (JSON) describes a factor graph's structure and how its tables are
drawn, never the table values. Programs are modules with
`infer(tables: {factor id: array}) -> {variable: normalized marginal}`; they are
scored on fresh tables drawn from the spec with seeds chosen at scoring time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import Factor, FactorGraph

# ---------------------------------------------------------------------------
# problem specs
# ---------------------------------------------------------------------------


def spec(name: str, cards: dict[str, int], factors: list[tuple[tuple[str, ...], dict]], tags=("in_scope",)) -> dict:
    return {
        "name": name,
        "variables": dict(cards),
        "factors": [{"id": i, "scope": list(sc), "table": tab} for i, (sc, tab) in enumerate(factors)],
        "tags": list(tags),
    }


def _pairwise_spec(name, n, K, edges, table=None, unary=False):
    cards = {f"x{i}": K for i in range(n)}
    tab = table or {"kind": "dense"}
    facs = [((f"x{i}", f"x{j}"), dict(tab)) for i, j in edges]
    if unary:
        facs += [((f"x{i}",), {"kind": "dense"}) for i in range(n)]
    return spec(name, cards, facs)


def _tree_edges(n, rng):
    return [(int(rng.integers(0, i)), i) for i in range(1, n)]


def _grid_edges(r, c):
    idx = lambda a, b: a * c + b  # noqa: E731
    out = []
    for a in range(r):
        for b in range(c):
            if b + 1 < c:
                out.append((idx(a, b), idx(a, b + 1)))
            if a + 1 < r:
                out.append((idx(a, b), idx(a + 1, b)))
    return out


def family_spec(family: str, rng, size: str = "public") -> dict:
    """One problem of a family; the structure is drawn with `rng` (used for the holdout set)."""
    wide = size == "holdout"  # the holdout draws sizes from wider ranges, to get unseen structures
    hi = (lambda a: a + 3) if wide else (lambda a: a)
    if family == "chain":
        n, K = (int(rng.integers(8, hi(17))), int(rng.choice([4, 8])))
        return _pairwise_spec(f"chain{n}_k{K}", n, K, [(i, i + 1) for i in range(n - 1)], unary=True)
    if family == "star":
        n, K = int(rng.integers(6, hi(11))), int(rng.choice([3, 4, 6]))
        return _pairwise_spec(f"star{n}_k{K}", n, K, [(0, i) for i in range(1, n)])
    if family == "tree":
        n, K = int(rng.integers(8, hi(15))), int(rng.choice([3, 4]))
        return _pairwise_spec(f"tree{n}_k{K}", n, K, _tree_edges(n, rng))
    if family == "cycle":
        n, K = int(rng.integers(6, hi(11))), int(rng.choice([3, 4, 6]))
        return _pairwise_spec(f"cycle{n}_k{K}", n, K, [(i, (i + 1) % n) for i in range(n)])
    if family == "grid":
        r, c = int(rng.integers(2, 4)), int(rng.integers(3, hi(5) - 1 if wide else 5))
        K = int(rng.choice([2, 3]))
        return _pairwise_spec(f"grid{r}x{c}_k{K}", r * c, K, _grid_edges(r, c))
    if family == "sparse":
        n, K = int(rng.integers(8, hi(13))), int(rng.choice([2, 3]))
        edges = _tree_edges(n, rng)
        present = {frozenset(e) for e in edges}
        cand = [(i, j) for i in range(n) for j in range(i + 1, n) if frozenset((i, j)) not in present]
        for k in rng.permutation(len(cand))[: n // 4]:
            edges.append(cand[int(k)])
        return _pairwise_spec(f"sparse{n}_k{K}", n, K, edges)
    if family == "lowrank":
        n, K, r = int(rng.integers(6, hi(9))), int(rng.choice([8, 12])), int(rng.choice([1, 2, 3]))
        kind = rng.choice(["chain", "cycle"])
        edges = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if kind == "cycle" else [])
        return _pairwise_spec(f"lowrank_{kind}{n}_k{K}_r{r}", n, K, edges, {"kind": "lowrank", "rank": r})
    if family == "ternary":
        n, K = int(rng.integers(6, hi(9))), int(rng.choice([2, 3])) if wide else 3
        cards = {f"x{i}": K for i in range(n)}
        facs = []
        for i in range(n - 2):
            facs.append(((f"x{i}", f"x{i + 1}", f"x{i + 2}"), {"kind": "dense"}))
        return spec(f"ternary{n}_k{K}", cards, facs)
    raise ValueError(f"unknown family {family!r}")


FAMILIES = ("chain", "star", "tree", "cycle", "grid", "sparse", "lowrank", "ternary")


def public_problems() -> list[dict]:
    """The fixed public set (two problems per family)."""
    out, seen = [], set()
    rng = np.random.default_rng(2026)
    for fam in FAMILIES:
        tries = 0
        while sum(1 for p in out if p["family"] == fam) < 2 and tries < 200:
            tries += 1
            p = family_spec(fam, rng)
            if p["name"] not in seen:
                seen.add(p["name"])
                out.append({**p, "family": fam})
    return out


def holdout_problems(seed: int, per_family: int = 2) -> list[dict]:
    """Problems the improving side never sees: same families, structures drawn from `seed`."""
    out, seen = [], {p["name"] for p in public_problems()}
    rng = np.random.default_rng(seed)
    for fam in FAMILIES:
        k = tries = 0
        while k < per_family and tries < 200:  # some families have few distinct structures
            tries += 1
            p = family_spec(fam, rng, size="holdout")
            if p["name"] not in seen:
                seen.add(p["name"])
                out.append({**p, "family": fam, "name": "holdout_" + p["name"]})
                k += 1
    return out


def save(problem: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{problem['name']}.json"
    path.write_text(json.dumps(problem, indent=1))
    return path


def load(path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# tables and graphs
# ---------------------------------------------------------------------------


def draw_tables(problem: dict, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    cards = problem["variables"]
    out = {}
    for f in problem["factors"]:
        shape = [cards[v] for v in f["scope"]]
        tab = f["table"]
        if tab["kind"] == "dense":
            out[f["id"]] = rng.uniform(0.1, 1.0, shape)
        elif tab["kind"] == "lowrank":
            r = tab["rank"]
            left = shape[:1]
            right = shape[1:]
            U = rng.uniform(0.1, 1.0, (int(np.prod(left)), r))
            V = rng.uniform(0.1, 1.0, (r, int(np.prod(right))))
            out[f["id"]] = (U @ V).reshape(shape)
        else:
            raise ValueError(f"unknown table kind {tab['kind']!r}")
    return out


def to_factor_graph(problem: dict, tables: dict[int, np.ndarray]) -> FactorGraph:
    return FactorGraph(
        dict(problem["variables"]),
        [Factor(f["id"], tuple(f["scope"]), tables[f["id"]]) for f in problem["factors"]],
    )


def reference_marginals(problem: dict, tables) -> dict[str, np.ndarray]:
    """Brute force when the joint is small enough, otherwise the junction tree."""
    from .baselines import brute_force_marginals, junction_tree_dag
    from .evaluate import SUM_PRODUCT, evaluate

    fg = to_factor_graph(problem, tables)
    if fg.size(fg.variables()) <= 2_000_000:
        return brute_force_marginals(fg)
    dag, _ = junction_tree_dag(fg)
    t, _ = evaluate(dag, fg, SUM_PRODUCT)
    return {v: x.data / x.data.sum() for v, x in t.items()}


# ---------------------------------------------------------------------------
# the egfg side
# ---------------------------------------------------------------------------


def egfg_program(problem: dict, sample_seed: int = 0, **optimize_kw) -> tuple[str, dict]:
    """Compile a problem with egfg: (generated source, info). Low-rank factors declared in
    the spec are searched with the low-rank equalities."""
    import time

    from .codegen import generate_for
    from .pipeline import optimize

    tables = draw_tables(problem, sample_seed)
    fg = to_factor_graph(problem, tables)
    kw = dict(extractor="greedy")
    if any(f["table"]["kind"] == "lowrank" for f in problem["factors"]):
        kw["structure"] = ("lowrank",)
    kw.update(optimize_kw)
    t0 = time.perf_counter()
    res = optimize(fg, **kw)
    src = generate_for(fg, res)
    info = {"compile_s": round(time.perf_counter() - t0, 3), "flops": res.extraction.cost,
            "egraph_nodes": res.num_nodes, "hit_limit": res.hit_limit}
    header = f'"""egfg program for {problem["name"]} (cost model: {res.extraction.cost} operations)."""\n'
    return header + src, info

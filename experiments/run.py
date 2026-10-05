"""H1/H2 experiments: compare e-graph extraction with the junction tree (BP) baseline.

Usage: uv run python experiments/run.py --out results/results.csv [--quick]
Rows are appended as they finish, so a partial run still leaves usable data.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egfg.baselines import factor_graph_bp_dag, junction_tree_dag, opt_einsum_cost  # noqa: E402
from egfg.cost import dag_cost, max_intermediate_size, reachable  # noqa: E402
from egfg.egraph import saturate  # noqa: E402
from egfg.extract import extract_dag_ilp, extract_tree  # noqa: E402
from egfg.generators import chain, cycle, grid, random_tree, star  # noqa: E402
from egfg.ir import Dag, all_marginal_queries, dag_scopes  # noqa: E402
from egfg.model import FactorGraph  # noqa: E402

COLUMNS = (
    "family,n,K,seed,num_vars,num_factors,egraph_nodes,iterations,hit_limit,saturate_s,"
    "tree_cost,ilp_cost,ilp_optimal,ilp_s,jt_cost,opt_einsum_cost,"
    "ilp_max_intermediate,jt_max_intermediate,jt_messages_covered,bp_cost,bp_messages_covered"
).split(",")


def node_signatures(dag: Dag, fg: FactorGraph) -> set[tuple[frozenset[int], frozenset[str]]]:
    """(factor ids below the node, node scope) for every reachable node."""
    scopes = dag_scopes(dag, fg)
    fids: dict[str, frozenset[int]] = {}

    def go(nid: str) -> frozenset[int]:
        if nid not in fids:
            node = dag.nodes[nid]
            if node.op == "leaf":
                fids[nid] = frozenset({node.arg})
            else:
                fids[nid] = frozenset().union(*(go(c) for c in node.children))
        return fids[nid]

    return {(go(nid), scopes[nid]) for nid in reachable(dag)}


def cases(quick: bool):
    """(family, n, K, seed, factor graph). Table values never change costs, so
    seeds only matter for random_tree's shape; other families use seed 0."""
    Ks = [2] if quick else [2, 5]
    tree_ns = [3, 4] if quick else [3, 4, 5, 6, 8]
    seeds = [0] if quick else [0, 1, 2]
    for K in Ks:
        for n in tree_ns:
            yield "chain", n, K, 0, chain(n, K)
            yield "star", n, K, 0, star(n, K)
            for s in seeds:
                yield "random_tree", n, K, s, random_tree(n, K, seed=s)
        for n in ([3, 4] if quick else [3, 4, 5, 6]):
            yield "cycle", n, K, 0, cycle(n, K)
        for cols in ([2] if quick else [2, 3]):
            yield "grid", 2 * cols, K, 0, grid(2, cols, K)


def run_case(fg: FactorGraph) -> dict:
    sat = saturate(fg, all_marginal_queries(fg))
    tree = extract_tree(sat.graph, fg)
    ilp = extract_dag_ilp(sat.graph, fg)
    jt, msgs = junction_tree_dag(fg)
    sigs = node_signatures(ilp.dag, fg)
    covered = sum(1 for m in set(msgs) if m in sigs) / max(1, len(set(msgs)))
    try:
        bp, bp_msgs = factor_graph_bp_dag(fg)
        bp_cost = dag_cost(bp, fg)
        bp_cov = round(sum(1 for m in set(bp_msgs) if m in sigs) / max(1, len(set(bp_msgs))), 3)
    except ValueError:  # loopy graph: exact BP does not apply
        bp_cost, bp_cov = "", ""
    return {
        "num_vars": len(fg.cards),
        "num_factors": len(fg.factors),
        "egraph_nodes": sat.num_nodes,
        "iterations": sat.iterations,
        "hit_limit": sat.hit_limit,
        "saturate_s": round(sat.seconds, 3),
        "tree_cost": tree.cost,
        "ilp_cost": ilp.cost,
        "ilp_optimal": ilp.optimal,
        "ilp_s": round(ilp.seconds, 3),
        "jt_cost": dag_cost(jt, fg),
        "opt_einsum_cost": opt_einsum_cost(fg),
        "ilp_max_intermediate": max_intermediate_size(ilp.dag, fg),
        "jt_max_intermediate": max_intermediate_size(jt, fg),
        "jt_messages_covered": round(covered, 3),
        "bp_cost": bp_cost,
        "bp_messages_covered": bp_cov,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for family, n, K, seed, fg in cases(args.quick):
            t0 = time.perf_counter()
            row = {"family": family, "n": n, "K": K, "seed": seed, **run_case(fg)}
            w.writerow(row)
            fh.flush()
            print(f"{family:12s} n={n} K={K} seed={seed} ilp={row['ilp_cost']} bp={row['bp_cost']} jt={row['jt_cost']} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

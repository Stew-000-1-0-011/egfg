"""Dynamic models (the same shape at every time step) and template-based filtering.

Variables are named `name@t`. Inside a template, time is written relative to
the current step (`a@-1`, `a@0`), so every step after the first is the same
local problem. That problem is searched once; the resulting Dag, whose Input
leaf `fwd` stands for the previous step's message, is reused at every step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cost import dag_cost, depends_on, split_cost
from .egraph import SaturationResult, saturate
from .evaluate import MAX_PRODUCT, SUM_PRODUCT, ExpectationSemiring, Table, evaluate, value_feature
from .extract import Extraction, class_leaves
from .ir import Dag, ENode, Input, Leaf, Mul, Sum, Term, to_dag
from .jtree import Piece, TreeTerms, junction_tree, product, sum_out
from .model import Factor, FactorGraph

FWD = "fwd"
MSG = "msg"


def at(name: str, t: int) -> str:
    return f"{name}@{t}"


def split(var: str) -> tuple[str, int]:
    name, t = var.rsplit("@", 1)
    return name, int(t)


@dataclass
class DynamicModel:
    """First-order dynamic model.

    `initial` and `transition` hold (scope, table); their tables are the same at
    every step. `observation` holds scopes only: the tables (likelihoods of the
    observed values) are given per time step. Scopes use relative time:
    initial and observation scopes use `@0`; transition scopes use `@-1` and `@0`.
    """

    states: dict[str, int]
    initial: list[tuple[tuple[str, ...], np.ndarray]]
    transition: list[tuple[tuple[str, ...], np.ndarray]]
    observation: list[tuple[str, ...]]

    def __post_init__(self):
        if MSG in self.states:
            raise ValueError(f"{MSG!r} is reserved")
        for scope, _ in self.initial:
            self._check(scope, {0})
        for scope, _ in self.transition:
            self._check(scope, {-1, 0})
        for scope in self.observation:
            self._check(scope, {0})

    def _check(self, scope: tuple[str, ...], times: set[int]) -> None:
        for v in scope:
            name, t = split(v)
            if name not in self.states or t not in times:
                raise ValueError(f"bad variable {v!r} in scope {scope}")

    def boundary(self) -> list[str]:
        """State names carried from one step to the next (those at @-1 in a transition)."""
        return sorted({split(v)[0] for scope, _ in self.transition for v in scope if split(v)[1] == -1})

    def obs_shape(self, k: int) -> tuple[int, ...]:
        return tuple(self.states[split(v)[0]] for v in self.observation[k])


def _rename(var: str, t: int) -> str:
    name, dt = split(var)
    return at(name, t + dt)


def _head_size(model: DynamicModel) -> int:
    return len(model.initial) + len(model.observation)


def _step_size(model: DynamicModel) -> int:
    return len(model.transition) + len(model.observation)


def global_id(model: DynamicModel, t: int, local: int) -> int:
    """Id in `unroll` of the local factor `local` of the step at time t."""
    return local if t == 0 else _head_size(model) + (t - 1) * _step_size(model) + local


def _step_factors(model: DynamicModel, t: int, obs_tables) -> list[tuple[tuple[str, ...], np.ndarray]]:
    """(relative scope, table) of the step at time t, in local id order."""
    first = model.initial if t == 0 else model.transition
    return list(first) + [(scope, np.asarray(tab)) for scope, tab in zip(model.observation, obs_tables)]


def unroll(model: DynamicModel, obs: list[list[np.ndarray]]) -> FactorGraph:
    """The plain factor graph of times 0..T-1 (T = len(obs))."""
    cards = {at(n, t): k for t in range(len(obs)) for n, k in model.states.items()}
    factors = []
    for t, tabs in enumerate(obs):
        for i, (scope, tab) in enumerate(_step_factors(model, t, tabs)):
            factors.append(Factor(global_id(model, t, i), tuple(_rename(v, t) for v in scope), tab))
    return FactorGraph(cards, factors)


# ---------------------------------------------------------------------------
# local problems and templates
# ---------------------------------------------------------------------------


@dataclass
class LocalStep:
    kind: str  # "head" (t = 0) or "step" (t >= 1)
    fg: FactorGraph  # relative variable names; observation tables are placeholders
    inputs: dict[str, frozenset[str]]
    queries: dict[str, Term]  # MSG and one marginal per state name
    seeds: dict[str, Term]
    obs_ids: frozenset[int]  # local ids of this step's observation factors


def _local_fg(model: DynamicModel, kind: str, obs_tables=None) -> FactorGraph:
    if obs_tables is None:
        obs_tables = [np.ones(model.obs_shape(k)) for k in range(len(model.observation))]
    t = 0 if kind == "head" else 1
    facs = _step_factors(model, t, obs_tables)
    used = {v for scope, _ in facs for v in scope}
    cards = {v: model.states[split(v)[0]] for v in used}
    cards.update({at(n, 0): k for n, k in model.states.items()})
    return FactorGraph(cards, [Factor(i, scope, tab) for i, (scope, tab) in enumerate(facs)])


def _keep_msg(model: DynamicModel) -> frozenset[str]:
    return frozenset(at(n, 0) for n in model.boundary())


def _fwd_scope(model: DynamicModel) -> frozenset[str]:
    return frozenset(at(n, -1) for n in model.boundary())


def _substitute(t: Term, fid: int, by: Term) -> Term:
    if isinstance(t, Leaf):
        return by if t.fid == fid else t
    if isinstance(t, Mul):
        return Mul(_substitute(t.a, fid, by), _substitute(t.b, fid, by))
    if isinstance(t, Sum):
        return Sum(t.var, _substitute(t.a, fid, by))
    return t


def local_step(model: DynamicModel, kind: str) -> LocalStep:
    fg = _local_fg(model, kind)
    inputs = {FWD: _fwd_scope(model)} if kind == "step" and model.boundary() else {}
    pieces: list[Piece] = [(Leaf(f.id), frozenset(f.scope), frozenset({f.id})) for f in fg.factors]
    pieces += [(Input(name), s, frozenset()) for name, s in sorted(inputs.items())]
    whole = product(pieces)
    queries: dict[str, Term] = {MSG: sum_out(whole, _keep_msg(model))[0]}
    for n in sorted(model.states):
        queries[n] = sum_out(whole, {at(n, 0)})[0]

    # Seeds: the junction tree of the step, with `fwd` as a stand-in factor.
    # A helper factor over the message's variables (never assigned to a clique)
    # makes sure some clique holds all of them.
    extra = list(fg.factors)
    fwd_id = len(extra)
    if inputs:
        extra.append(Factor(fwd_id, tuple(sorted(inputs[FWD])), np.ones([fg.cards[v] for v in sorted(inputs[FWD])])))
    keep = sorted(_keep_msg(model))
    helper_id = len(extra)
    if len(keep) > 1:
        extra.append(Factor(helper_id, tuple(keep), np.ones([fg.cards[v] for v in keep])))
    aug = FactorGraph(fg.cards, extra)
    jt = junction_tree(aug)
    for i in jt.assigned:
        jt.assigned[i] = [f for f in jt.assigned[i] if f != helper_id]
    calc = TreeTerms(aug, jt)
    seeds = {MSG: calc.joint(set(keep))[0]}
    for n in sorted(model.states):
        seeds[n] = calc.marginal(at(n, 0))[0]
    if inputs:
        seeds = {q: _substitute(t, fwd_id, Input(FWD)) for q, t in seeds.items()}
    nfirst = len(model.initial) if kind == "head" else len(model.transition)
    return LocalStep(kind, fg, inputs, queries, seeds, frozenset(range(nfirst, len(fg.factors))))


@dataclass
class Template:
    local: LocalStep
    dag: Dag
    cost: int
    saturation: SaturationResult | None = None
    extraction: Extraction | None = None
    prep_cost: int = 0  # nodes not depending on this step's observations (computable in advance)
    latency_cost: int = 0  # nodes that must wait for the observations

    def __post_init__(self):
        self.prep_cost, self.latency_cost = split_cost(self.dag, self.local.fg, self.local.obs_ids)
        dep = depends_on(self.dag, self.local.obs_ids)
        self.prep_nodes = [nid for nid, d in dep.items() if not d]


@dataclass
class FilterProgram:
    model: DynamicModel
    head: Template
    step: Template
    search_s: float = 0.0

    def total_cost(self, T: int) -> int:
        return self.head.cost + (T - 1) * self.step.cost


def compile_filter(
    model: DynamicModel,
    rules: str = "full",
    seed: bool = True,
    extractor: str = "ilp",
    max_iters: int = 30,
    node_limit: int = 50_000,
    time_limit_s: float = 60,
    objective: str = "total",
) -> FilterProgram:
    """Search the head and step templates once each.

    `objective`: "total" minimizes the step cost; "latency" minimizes the cost of
    the nodes that wait for the observations, then the total; "weighted:L" counts
    those nodes L times.
    """
    from .pipeline import _extract

    start = time.perf_counter()
    fwd = forward_program(model) if objective != "total" else None
    temps = []
    for kind in ("head", "step"):
        loc = local_step(model, kind)
        sat = saturate(
            loc.fg,
            loc.queries,
            max_iters=max_iters,
            node_limit=node_limit,
            rules=rules,
            inputs=loc.inputs,
            seeds=loc.seeds if seed else None,
        )
        weights = None
        if objective != "total":
            w_after = _latency_weight(objective, (fwd.head if kind == "head" else fwd.step).cost)
            leaves = class_leaves(sat.graph)
            weights = {cid: w_after if leaves[cid] & loc.obs_ids else 1 for cid in leaves}
        ex = _extract(sat.graph, loc.fg, extractor, time_limit_s, weights)
        temps.append(Template(loc, ex.dag, ex.cost, sat, ex))
    return FilterProgram(model, temps[0], temps[1], time.perf_counter() - start)


def _latency_weight(objective: str, forward_cost: int) -> int:
    if objective == "latency":
        # larger than any preparation cost worth paying: latency first, total second
        return forward_cost + 1
    if objective.startswith("weighted:"):
        return int(objective.split(":", 1)[1])
    raise ValueError(f"unknown objective {objective!r}")


def forward_program(model: DynamicModel) -> FilterProgram:
    """The textbook forward algorithm: predict (Σ over t-1 of fwd · transitions), then update (· observations)."""
    temps = []
    for kind in ("head", "step"):
        loc = local_step(model, kind)
        leaf = {f.id: (Leaf(f.id), frozenset(f.scope), frozenset({f.id})) for f in loc.fg.factors}
        nfirst = len(model.initial) if kind == "head" else len(model.transition)
        obs = [leaf[i] for i in range(nfirst, len(leaf))]
        if kind == "head":
            updated = product([leaf[i] for i in range(nfirst)] + obs)
        else:
            parts = [(Input(FWD), loc.inputs[FWD], frozenset())] if loc.inputs else []
            parts += [leaf[i] for i in range(nfirst)]
            now = {at(n, 0) for n in model.states}
            pred = product(parts)
            pred = None if pred is None else sum_out(pred, now)
            updated = product(([pred] if pred is not None else []) + obs)
        terms = {MSG: sum_out(updated, _keep_msg(model))[0]}
        for n in sorted(model.states):
            terms[n] = sum_out(updated, {at(n, 0)})[0]
        dag = to_dag(terms, loc.inputs)
        temps.append(Template(loc, dag, dag_cost(dag, loc.fg)))
    return FilterProgram(model, temps[0], temps[1])


def jt_program(model: DynamicModel) -> FilterProgram:
    """The junction tree of each local problem (the seeds), without any search."""
    temps = []
    for kind in ("head", "step"):
        loc = local_step(model, kind)
        dag = to_dag(loc.seeds, loc.inputs)
        temps.append(Template(loc, dag, dag_cost(dag, loc.fg)))
    return FilterProgram(model, temps[0], temps[1])


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _normalize(t: Table) -> Table:
    lead = t.data.ndim - len(t.vars)
    z = t.data[0].sum() if lead else t.data.sum()
    return Table(t.vars, t.data / z)


def _shift_back(t: Table) -> Table:
    """A message over `x@0` becomes the next step's input over `x@-1` (same sorted order)."""
    return Table(tuple(at(split(v)[0], -1) for v in t.vars), t.data)


class Filter:
    """Online filtering: feed one time step's observation tables at a time."""

    def __init__(self, program: FilterProgram, semiring=SUM_PRODUCT):
        self.program = program
        self.semiring = semiring
        self.t = 0
        self._msg: Table | None = None
        self._known: dict[str, Table] | None = None
        self._sr = None

    def _template(self) -> Template:
        return self.program.head if self.t == 0 else self.program.step

    def prepare(self, semiring=None, fg: FactorGraph | None = None) -> None:
        """Before the next observations arrive: compute everything that does not depend on them."""
        tmpl = self._template()
        if fg is None:  # placeholder observation tables: no prepared node reads them
            fg = _local_fg(self.program.model, tmpl.local.kind)
        self._sr = semiring or self.semiring
        inputs = {FWD: self._msg} if tmpl.local.inputs else {}
        sub = Dag(tmpl.dag.nodes, {nid: nid for nid in tmpl.prep_nodes}, tmpl.dag.inputs)
        self._known, _ = evaluate(sub, fg, self._sr, inputs)

    def update(self, obs_tables, fg: FactorGraph | None = None) -> dict[str, Table]:
        """The observations arrived: finish the step; return each state's (unnormalized) table."""
        if self._known is None:
            self.prepare()
        tmpl = self._template()
        fg = fg or _local_fg(self.program.model, tmpl.local.kind, obs_tables)
        inputs = {FWD: self._msg} if tmpl.local.inputs else {}
        tables, _ = evaluate(tmpl.dag, fg, self._sr, inputs, known=self._known)
        self._msg = _shift_back(_normalize(tables[MSG]))
        self._known = None
        self.t += 1
        return {n: tables[n] for n in self.program.model.states}

    def step(self, obs_tables, semiring_for: Callable[[FactorGraph], object] | None = None) -> dict[str, Table]:
        """prepare() then update(): advance one step."""
        fg = _local_fg(self.program.model, self._template().local.kind, obs_tables)
        self.prepare(semiring_for(fg) if semiring_for else None, fg)
        return self.update(obs_tables, fg)


def _scalar(t: Table) -> np.ndarray:
    return t.data / t.data.sum()


def filter_marginals(program: FilterProgram, obs) -> list[dict[str, np.ndarray]]:
    f = Filter(program)
    return [{n: _scalar(t) for n, t in f.step(tabs).items()} for tabs in obs]


def filter_max_marginals(program: FilterProgram, obs) -> list[dict[str, np.ndarray]]:
    """Per time and state: max over everything else up to that time (scaled to sum 1)."""
    f = Filter(program, MAX_PRODUCT)
    return [{n: _scalar(t) for n, t in f.step(tabs).items()} for tabs in obs]


def filter_moments(program: FilterProgram, obs, name: str, t: int, values: np.ndarray) -> tuple[float, float]:
    """Mean and variance of values[name@t] given the observations up to time t."""
    f = Filter(program)
    for s, tabs in enumerate(obs[: t + 1]):
        if s < t:
            out = f.step(tabs, lambda fg: ExpectationSemiring({}))
        else:
            out = f.step(tabs, lambda fg: ExpectationSemiring(value_feature(fg, at(name, 0), values)))
    p, r, q = (float(out[name].data[i].sum()) for i in range(3))
    mean = r / p
    return mean, q / p - mean * mean


def instantiate(program: FilterProgram, T: int) -> Dag:
    """All T steps as one Dag over `unroll`'s factor graph (no rescaling).

    Roots are `name@t` for every state and time, and `msg@t` for every message.
    """
    model = program.model
    nodes: dict[str, ENode] = {}
    roots: dict[str, str] = {}
    for t in range(T):
        tmpl = program.head if t == 0 else program.step
        p = f"t{t}:"
        for nid, n in tmpl.dag.nodes.items():
            if n.op == "leaf":
                node = ENode("leaf", global_id(model, t, n.arg), ())
            elif n.op == "sum":
                node = ENode("sum", _rename(n.arg, t), (p + n.children[0],))
            elif n.op == "input":
                node = ENode("input", n.arg, ())
            else:
                node = ENode(n.op, n.arg, tuple(p + c for c in n.children))
            nodes[p + nid] = node
        for q, nid in tmpl.dag.roots.items():
            roots[at(q, t)] = p + nid

    def resolve(nid: str) -> str:
        if nodes[nid].op != "input":
            return nid
        t = int(nid[1 : nid.index(":")])
        return resolve(roots[at(MSG, t - 1)])

    out = {nid: ENode(n.op, n.arg, tuple(resolve(c) for c in n.children)) for nid, n in nodes.items()}
    roots = {q: resolve(nid) for q, nid in roots.items()}
    used = set()
    stack = list(roots.values())
    while stack:
        nid = stack.pop()
        if nid not in used:
            used.add(nid)
            stack.extend(out[nid].children)
    return Dag({nid: out[nid] for nid in used}, roots)


# ---------------------------------------------------------------------------
# independent reference: forward pass over the joint state with numpy
# ---------------------------------------------------------------------------


def reference_filter(model: DynamicModel, obs) -> list[dict[str, np.ndarray]]:
    """Filtering by brute force over the joint state, one step at a time (no egfg code)."""
    names = sorted(model.states)
    m = len(names)
    axis = {}  # einsum axis: previous step 0..m-1, current step m..2m-1
    for i, n in enumerate(names):
        axis[at(n, -1)] = i
        axis[at(n, 0)] = m + i
    now = [axis[at(n, 0)] for n in names]
    prev = [axis[at(n, -1)] for n in names]
    size = {axis[at(n, d)]: model.states[n] for n in names for d in (-1, 0)}

    def joint(facs, out: list[int]) -> np.ndarray:
        """Product of the factors as an array over the axes `out` (1 along absent ones)."""
        ops = []
        present = set()
        for scope, tab in facs:
            ops += [np.asarray(tab, float), [axis[v] for v in scope]]
            present |= {axis[v] for v in scope}
        for c in out:
            if c not in present:
                ops += [np.ones(size[c]), [c]]
        return np.einsum(*ops, out)

    result = []
    alpha = None
    for t, tabs in enumerate(obs):
        obs_f = list(zip(model.observation, tabs))
        if t == 0:
            alpha = joint(list(model.initial) + obs_f, now)
        else:
            trans = joint(list(model.transition), prev + now)
            pred = np.einsum(alpha, list(range(m)), trans, list(range(2 * m)), list(range(m, 2 * m)))
            alpha = pred * joint(obs_f, now)
        alpha = alpha / alpha.sum()
        result.append({n: alpha.sum(axis=tuple(j for j in range(m) if j != i)) for i, n in enumerate(names)})
    return result

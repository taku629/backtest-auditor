"""Dynamic leak localizer: find the source line where future information enters, by experiment.

The future-perturbation test proves THAT signals at dates <= t change when only data after t is scrambled.
This module finds WHERE. The strategy module is AST-instrumented so that after every assignment
(``Assign``/``AugAssign``/``AnnAssign``, in ``generate_signals`` and every helper function in the module, inside loops
and branches too), every ``for`` target, every in-place method call on a local, every function parameter and every
``return`` value, the value is handed to a recorder. The instrumented code runs in the sandbox on the original data
and on perturbed copies (rows <= cut bit-identical, rows after scrambled: ``perturb_after`` with the same seed and
cuts as ``run_audit``). For each recorded value the part a point-in-time computation could know at the cut is
compared:

* objects indexed by date (DataFrame/Series/DatetimeIndex/PeriodIndex/MultiIndex with a date level, ndarrays
  with one row per data date): rows whose date label is <= cut, aligned by label, so newest-first or resampled
  objects are handled;
* anything else (a per-ticker full-sample mean, a scalar threshold, a ticker list): the whole value.

Divergence is then traced like taint: a recorded statement is an *origin* when its value diverges although none of
the recorded variables it reads (nor the helper functions it calls) had diverged. Walking back from the returned
weights through the variables each statement reads gives the *chain*; the entry point is the earliest origin on
that chain.
"""
from __future__ import annotations

import ast
import time
from collections import Counter

import numpy as np

from audit.leak_tests import choose_cuts, perturb_after

KIND_LABEL = {"time": "time-indexed rows changed", "whole": "non-time object changed"}


# ----------------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------------
def _names_loaded(node) -> set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _root_name(node) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _store_names(target) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for e in target.elts:
            out += _store_names(e.value if isinstance(e, ast.Starred) else e)
        return out
    if isinstance(target, (ast.Attribute, ast.Subscript)):  # w.loc[...] = x mutates w
        r = _root_name(target)
        return [r] if r else []
    return []


class _Instrumenter:
    def __init__(self, tree: ast.Module):
        self.meta: list[dict] = []
        self.funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def _calls(self, node) -> set[str]:
        return {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                and c.func.id in self.funcs} if node is not None else set()

    def _new(self, func, line, var, kind, loads, calls) -> int:
        self.meta.append({"sid": len(self.meta), "func": func, "line": int(line), "var": var, "kind": kind,
                          "loads": sorted(loads), "calls": sorted(calls)})
        return len(self.meta) - 1

    @staticmethod
    def _rec_stmt(sid: int, var: str, like) -> ast.stmt:
        call = ast.Call(ast.Name("_bta_rec", ast.Load()), [ast.Constant(sid), ast.Name(var, ast.Load())], [])
        return ast.copy_location(ast.Expr(call), like)

    def function(self, fn: ast.FunctionDef) -> None:
        name = fn.name
        locals_: set[str] = set()
        a = fn.args
        params = [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)]
        params += [x.arg for x in (a.vararg, a.kwarg) if x is not None]
        locals_.update(params)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                locals_.add(n.id)
        pre = [self._rec_stmt(self._new(name, fn.lineno, p, "param", set(), set()), p, fn) for p in params]
        body = self.block(fn.body, name, locals_)
        doc = 1 if (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)) else 0
        fn.body = body[:doc] + pre + body[doc:] if (body[doc:] or pre) else body

    def block(self, stmts: list[ast.stmt], func: str, locals_: set[str]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for s in stmts:
            after: list[ast.stmt] = []
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function(s)
            elif isinstance(s, ast.Assign):
                loads = _names_loaded(s.value) | set().union(*[_names_loaded(t) for t in s.targets])
                calls = self._calls(s.value)
                for t in s.targets:
                    for v in _store_names(t):
                        after.append(self._rec_stmt(self._new(func, s.lineno, v, "assign", loads, calls), v, s))
            elif isinstance(s, ast.AugAssign):
                loads = _names_loaded(s.value) | _names_loaded(s.target) | set(_store_names(s.target))
                for v in _store_names(s.target):
                    after.append(self._rec_stmt(self._new(func, s.lineno, v, "augassign", loads,
                                                          self._calls(s.value)), v, s))
            elif isinstance(s, ast.AnnAssign) and s.value is not None:
                loads = _names_loaded(s.value) | _names_loaded(s.target)
                for v in _store_names(s.target):
                    after.append(self._rec_stmt(self._new(func, s.lineno, v, "assign", loads,
                                                          self._calls(s.value)), v, s))
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute):
                r = _root_name(s.value.func)
                if r and r in locals_:  # e.g. w.fillna(0, inplace=True), lst.append(x)
                    after.append(self._rec_stmt(self._new(func, s.lineno, r, "mutate", _names_loaded(s.value),
                                                          self._calls(s.value)), r, s))
            elif isinstance(s, ast.Return) and s.value is not None:
                sid = self._new(func, s.lineno, "<return>", "return", _names_loaded(s.value), self._calls(s.value))
                s.value = ast.copy_location(ast.Call(ast.Name("_bta_ret", ast.Load()), [ast.Constant(sid), s.value],
                                                     []), s.value)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                pre = [self._rec_stmt(self._new(func, s.lineno, v, "for", _names_loaded(s.iter), self._calls(s.iter)),
                                      v, s) for v in _store_names(s.target)]
                s.body = pre + self.block(s.body, func, locals_)
                s.orelse = self.block(s.orelse, func, locals_)
            elif isinstance(s, ast.While):
                s.body = self.block(s.body, func, locals_)
                s.orelse = self.block(s.orelse, func, locals_)
            elif isinstance(s, ast.If):
                s.body = self.block(s.body, func, locals_)
                s.orelse = self.block(s.orelse, func, locals_)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                s.body = self.block(s.body, func, locals_)
            elif isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
                s.body = self.block(s.body, func, locals_)
                for h in s.handlers:
                    h.body = self.block(h.body, func, locals_)
                s.orelse = self.block(s.orelse, func, locals_)
                s.finalbody = self.block(s.finalbody, func, locals_)
            out.append(s)
            out.extend(after)
        return out


def instrument(code: str) -> tuple[str, list[dict]]:
    """Return (instrumented source, statement metadata indexed by sid). Only functions are instrumented."""
    tree = ast.parse(code)
    ins = _Instrumenter(tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ins.function(node)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", ins.meta


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def _key(m: dict) -> tuple[str, str]:
    return (m["func"], m["var"])


def _load_keys(m: dict) -> set[tuple[str, str]]:
    return {(m["func"], n) for n in m["loads"]} | {(g, "<return>") for g in m["calls"]}


def analyse_job(records: list[dict], meta: list[dict], func_name: str = "generate_signals") -> dict:
    """Origins, chain and entry for one perturbation run (records in execution order)."""
    state: dict = {}
    rows = []
    for seq, r in enumerate(records):
        m = meta[r["s"]]
        d = bool(r.get("d"))
        origin = d and m["kind"] != "param" and not any(state.get(k) for k in _load_keys(m))
        state[_key(m)] = d
        rows.append({"seq": seq, "rec": r, "meta": m, "diverged": d, "origin": origin})
    ret_i = next((i for i in range(len(rows) - 1, -1, -1)
                  if rows[i]["meta"]["func"] == func_name and rows[i]["meta"]["kind"] == "return"), None)
    first_div = next((x for x in rows if x["diverged"]), None)
    if ret_i is None or not rows[ret_i]["diverged"]:
        return {"found": False, "rows": rows, "first_divergent": first_div}
    needed = set(_load_keys(rows[ret_i]["meta"]))
    chain = [rows[ret_i]]
    for i in range(ret_i - 1, -1, -1):
        x = rows[i]
        if x["diverged"] and _key(x["meta"]) in needed:
            chain.append(x)
            needed |= _load_keys(x["meta"])
    chain.reverse()
    origins = [x for x in chain if x["origin"]]
    entry = origins[0] if origins else chain[0]
    return {"found": True, "rows": rows, "chain": chain, "entry": entry, "first_divergent": first_div}


def _describe(x: dict, src: list[str], cut_date: str | None = None) -> dict:
    m, r = x["meta"], x["rec"]
    ln = m["line"]
    d = {"line": ln, "var": m["var"], "func": m["func"],
         "source_line": src[ln - 1].strip() if 0 < ln <= len(src) else "",
         "kind": KIND_LABEL.get(r.get("k"), "value changed"), "first_changed_date": r.get("f"),
         "detail": r.get("why")}
    if r.get("r") is not None:
        d["rows_changed"] = r["r"]
    if r.get("m") is not None:
        d["max_abs_change"] = round(float(r["m"]), 6)
    if r.get("o"):
        d["occurrence"] = int(r["o"]) + 1
    if cut_date:
        d["cut_date"] = cut_date
    return d


def _variants(data: dict, cut_positions: list[int] | None, seed: int, n_cuts: int, max_cuts: int):
    """Reproduce run_audit's perturbed copies (same seed, same cuts, same rng sequence)."""
    T = len(data["close"])
    cuts = choose_cuts(T, n_cuts, seed)
    rng = np.random.default_rng(seed + 1)
    audit_vars = {c: perturb_after(data, c, rng) for c in cuts}
    wanted = list(cut_positions) if cut_positions else list(cuts)
    wanted = [int(c) for c in wanted if 0 <= int(c) < T - 1][:max_cuts]
    out, extra_rng = [], np.random.default_rng(seed + 1)
    for c in wanted:
        out.append((c, audit_vars[c] if c in audit_vars else perturb_after(data, c, extra_rng)))
    return out


def localize_leak(code: str, data: dict, cut_positions: list[int] | None = None, seed: int = 7, *,
                  n_cuts: int = 6, max_cuts: int = 2, timeout: float = 120.0,
                  func_name: str = "generate_signals") -> dict:
    """Pinpoint the line where future information enters ``generate_signals``.

    ``cut_positions``: row positions of cut dates to test (e.g. the leaking cuts of the future-perturbation test).
    Default: the cuts ``run_audit`` uses with this ``seed``/``n_cuts`` (at most ``max_cuts`` of them).

    Returns ``{found, entry, chain, per_line, cuts, seconds, ...}``; ``found`` is False when no recorded value at
    a date <= cut changed (nothing leaks at these cuts). Raises ``sandbox.tracer.TraceError`` on policy/sandbox
    errors.
    """
    from sandbox.tracer import run_trace

    t0 = time.time()
    src = code.splitlines()
    inst, meta = instrument(code)
    variants = _variants(data, cut_positions, seed, n_cuts, max_cuts)
    if not variants:
        return {"found": False, "reason": "no usable cut dates", "seconds": round(time.time() - t0, 3)}
    base = {k: data[k] for k in data}
    res = run_trace(code, inst, base, [v for _, v in variants], [c for c, _ in variants], timeout=timeout,
                    func_name=func_name)
    per_cut, analyses = [], []
    for job in res["jobs"]:
        a = analyse_job(job["records"], meta, func_name)
        a["cut_date"] = job["cut_date"]
        analyses.append(a)
        pc = {"cut_date": job["cut_date"], "output_changed": a["found"]}
        if a["found"]:
            pc["entry_line"] = a["entry"]["meta"]["line"]
        per_cut.append(pc)
    found = [a for a in analyses if a["found"]]
    out = {"found": bool(found), "method": "dynamic taint by future perturbation", "cuts": per_cut,
           "n_statements_instrumented": len(meta), "n_values_recorded": len(res.get("base_order", [])),
           "recorder_errors": res.get("recorder_errors", 0), "sandbox_seconds": res.get("duration_s"),
           "seconds": round(time.time() - t0, 3)}
    if not found:
        fd = next((a["first_divergent"] for a in analyses if a["first_divergent"]), None)
        if fd is not None:  # something diverged but did not reach the returned weights
            out["note"] = "values diverged at dates <= cut but the returned weights did not"
            out["first_divergent_unused"] = _describe(fd, src, analyses[0]["cut_date"])
        return out
    # consensus entry line across cuts (ties -> first cut)
    votes = Counter(a["entry"]["meta"]["line"] for a in found)
    best_line = max(votes, key=lambda ln: (votes[ln], -[a["entry"]["meta"]["line"] for a in found].index(ln)))
    a = next(x for x in found if x["entry"]["meta"]["line"] == best_line)
    entry = _describe(a["entry"], src, a["cut_date"])
    chain, seen = [], set()
    for x in a["chain"]:
        k = (x["meta"]["line"], x["meta"]["var"])
        if k in seen:
            continue
        seen.add(k)
        chain.append(_describe(x, src))
    per_line, seen = [], {}
    for x in a["rows"]:
        k = (x["meta"]["line"], x["meta"]["var"], x["meta"]["func"])
        if k in seen:
            if x["diverged"] and not per_line[seen[k]]["diverged"]:
                per_line[seen[k]] = _describe(x, src) | {"diverged": True, "origin": x["origin"]}
            continue
        seen[k] = len(per_line)
        per_line.append(_describe(x, src) | {"diverged": x["diverged"], "origin": x["origin"]})
    for p in per_line:
        if not p["diverged"]:
            for f in ("kind", "first_changed_date", "detail", "rows_changed", "max_abs_change"):
                p.pop(f, None)
    out.update({"entry": entry, "chain": chain, "per_line": per_line,
                "agreement": f"{votes[best_line]}/{len(found)} leaking cut(s)"})
    return out


def localized_message(loc: dict | None) -> str:
    """One line for humans and LLM prompts, or '' when nothing was localized."""
    if not loc or not loc.get("found") or not loc.get("entry"):
        return ""
    e = loc["entry"]
    when = f", first changed date {e['first_changed_date']}" if e.get("first_changed_date") else ""
    return (f"Leak enters at line {e['line']}: `{e['source_line']}` (variable `{e['var']}`; {e['kind']}{when}; "
            f"found by dynamic perturbation trace).")

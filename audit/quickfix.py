"""Quick fix: deterministic, LLM-free repair of a look-ahead / survivorship leak.

After the audit fails and the dynamic localizer (``audit/localize.py``) has pinned the line where future data
enters, this module generates a small set of candidate patches for that line (and for the lines the static AST scan
flags), applies each one, and re-runs the *full* deterministic audit (future perturbation, survivorship, execution
delay, static scan ...) in the sandbox. The first candidate that passes every blocking test **without going to
cash** (``non_trivial`` must pass and mean gross exposure must stay >= ``MIN_GROSS``, the same over-correction rule
as the repair benchmark) is accepted and returned as a unified diff.

Every rewrite is a general pandas rule, keyed on the AST pattern, never on a specific file:

==========================  ===================================================================================
rule                        rewrite(s), in the order they are tried
==========================  ===================================================================================
negative_period             ``x.shift(-k)`` -> ``x.shift(k)`` (also ``pct_change``/``diff``, ``-var``); then drop the
                            negative shift
centered_window             ``rolling(n, center=True)`` -> ``rolling(n)``
backward_fill               ``bfill()``/``backfill()`` -> ``ffill()``; ``method='bfill'|'nearest'`` -> ``'ffill'``;
                            ``interpolate(...)`` -> ``ffill()``
time_axis_rank              ``rank()`` (axis 0 by default) -> ``rank(axis=1)``; explicit ``rank(axis=0)`` ->
                            ``expanding(min_periods=..).rank()`` first (keeps the time-series intent)
full_sample_statistic       ``X.mean()/std()/median()/var()/quantile(q)/min()/max()`` over time -> ``X.expanding()...``
time_axis_demean            ``X.sub(X.mean(), axis=1)`` / ``X - X.mean()`` -> cross-sectional
                            ``X.sub(X.mean(axis=1), axis=0)`` (tried first when the code wrote ``axis=1``)
resample_start_label        ``resample('MS').last()`` -> ``resample('ME').last()``; then ``.shift(1)`` after it; then
                            ``label='right', closed='right'``
period_transform            ``groupby(<period>).transform('last')`` -> ``transform('first')``
reversed_time_window        drop ``iloc[::-1]`` / ``sort_index(ascending=False)`` around order-sensitive windows
pooled_quantile             ``pd.qcut(stacked, ..)`` -> per date ``stacked.groupby(level=0).transform(qcut)``;
                            ``X.stack().groupby(level=1).quantile(q)`` -> ``X.expanding().quantile(q)``
survivor_universe           ``X.dropna(axis=1)`` -> ``X``
future_bar_index            backtrader ``self.data.close[1]`` (next bar) -> ``[0]``; ``.get(ago=k)`` -> ``ago=0``
lag_statement (fallback)    ``v = <expr>`` -> ``v = (<expr>).shift(1)`` on the localized line
==========================  ===================================================================================

Nothing here calls a model, so a Quick fix costs $0. When no candidate passes, the caller falls back to the
LLM Coder (``agents/loop.py``) or reports that no deterministic fix was found.
"""
from __future__ import annotations

import ast
import copy
import difflib
import time
from dataclasses import dataclass, field

from audit.ast_scan import (COL_AXIS, LOOKAHEAD_RULE_IDS, ROW_AXIS, START_ANCHORED, _axis_of, _chain_names,
                            _const, _is_reversal, _kw, _neg_const, _pooled_panel, _reversed_parity, scan_code)

MIN_GROSS = 0.05            # same over-correction threshold as bench/run_repair.py (TRIVIAL_GROSS)
MAX_CANDIDATES = 12
REDUCERS = {"mean", "std", "median", "var", "min", "max", "quantile", "sem", "skew", "kurt"}
WINDOWED = {"rolling", "expanding", "ewm", "groupby", "resample"}
SURVIVORSHIP_RULES = {"survivor_universe"}
END_ANCHOR = {"MS": "ME", "BMS": "BME", "SMS": "SME", "QS": "QE", "BQS": "BQE", "AS": "YE", "YS": "YE",
              "BAS": "BYE", "BYS": "BYE"}


@dataclass
class Candidate:
    rule: str
    description: str
    code: str
    lines: list[int] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Source editing: replace node segments in the original text (comments and formatting elsewhere are kept)
# ----------------------------------------------------------------------------
def _offsets(code: str) -> list[int]:
    """Byte offset of the start of every line (ast col offsets are UTF-8 byte offsets)."""
    out, pos = [0], 0
    for ln in code.encode("utf-8").splitlines(keepends=True):
        pos += len(ln)
        out.append(pos)
    return out


def _apply(code: str, edits: list[tuple[ast.AST, str]]) -> str:
    """Replace each node's source span with text. Overlapping edits: the outermost wins (inner ones dropped)."""
    b = code.encode("utf-8")
    offs = _offsets(code)
    spans = []
    for node, text in edits:
        s = offs[node.lineno - 1] + node.col_offset
        e = offs[node.end_lineno - 1] + node.end_col_offset
        spans.append((s, e, text))
    spans.sort(key=lambda x: (x[0], -x[1]))
    kept, last_end = [], -1
    for s, e, t in spans:
        if s >= last_end:
            kept.append((s, e, t))
            last_end = e
    for s, e, t in reversed(kept):
        b = b[:s] + t.encode("utf-8") + b[e:]
    return b.decode("utf-8")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def _lagged(node: ast.AST, k: str = "1") -> str:
    """Source of ``node.shift(k)``, parenthesised only when needed."""
    t = _src(node)
    simple = isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript))
    return f"{t if simple else '(' + t + ')'}.shift({k})"


def _call(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)


def _set_kw(call: ast.Call, name: str, value: ast.AST | None, positional: int | None = None) -> ast.Call:
    """Copy of call with keyword ``name`` set (or removed when value is None); a positional slot is dropped."""
    c = copy.deepcopy(call)
    if positional is not None and len(c.args) > positional:
        c.args = c.args[:positional] + c.args[positional + 1:]
    c.keywords = [k for k in c.keywords if k.arg != name]
    if value is not None:
        c.keywords.insert(0, ast.keyword(arg=name, value=value))
    return c


def _with_attr(call: ast.Call, attr: str, *, keep_args: bool = True) -> ast.Call:
    c = copy.deepcopy(call)
    c.func.attr = attr
    if not keep_args:
        c.args, c.keywords = [], []
    return c


# ----------------------------------------------------------------------------
# Locating the code to patch
# ----------------------------------------------------------------------------
def _statements(tree: ast.Module) -> list[ast.stmt]:
    simple = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Return)
    return [n for n in ast.walk(tree) if isinstance(n, simple)]


def _stmt_at(tree: ast.Module, line: int) -> ast.stmt | None:
    best = None
    for s in _statements(tree):
        if s.lineno <= line <= (s.end_lineno or s.lineno):
            if best is None or (s.end_lineno - s.lineno) < (best.end_lineno - best.lineno):
                best = s
    return best


def _nodes_in(stmt: ast.stmt) -> list[ast.AST]:
    return [n for n in ast.walk(stmt) if hasattr(n, "lineno")]


def _assignments(tree: ast.Module) -> dict[str, ast.Assign]:
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out.setdefault(n.targets[0].id, n)
    return out


# ----------------------------------------------------------------------------
# Candidate generators: (tree, stmt, ctx) -> list[(rule, description, edits)]
# ----------------------------------------------------------------------------
def _gen_negative_period(tree, stmt, ctx):
    flip, drop = [], []
    for n in _nodes_in(stmt):
        if not (_call(n) and n.func.attr in ("shift", "pct_change", "diff")):
            continue
        arg, kw = (n.args[0], None) if n.args else (_kw(n, "periods"), "periods")
        if arg is None:
            continue
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            pos = arg.operand
        elif _neg_const(arg):
            pos = ast.Constant(-arg.value)
        else:
            continue
        flip.append((arg, _src(pos)))
        if n.func.attr == "shift":
            drop.append((n, _src(n.func.value)))
    out = []
    if flip:
        out.append(("negative_period", "negative period -> positive (look back instead of ahead)", flip))
    # a forward-looking target (fwd = ret.shift(-k)) fed into a trailing estimate (rolling(...).corr(fwd)):
    # keep the target, lag the estimate by the horizon k so its window ends at the last fully observed target
    if flip and isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        v = stmt.targets[0].id
        horizon = flip[0][1]
        cons = []
        for s2 in _statements(tree):
            if s2.lineno <= stmt.lineno or not isinstance(s2, ast.Assign) or s2.value is None:
                continue
            loads = {n.id for n in ast.walk(s2.value) if isinstance(n, ast.Name)}
            windows = {c.func.attr for c in ast.walk(s2.value) if _call(c)} & {"rolling", "expanding", "ewm"}
            if v in loads and windows:
                cons.append((s2.value, _lagged(s2.value, horizon)))
        if cons:
            out.append(("negative_period", f"forward target in a trailing estimate -> lag the estimate by "
                                           f"the horizon ({horizon})", cons))
    if drop:
        out.append(("negative_period", "remove the negative shift", drop))
    return out


def _gen_centered(tree, stmt, ctx):
    ed = [(n, _src(_set_kw(n, "center", None))) for n in _nodes_in(stmt)
          if _call(n) and n.func.attr == "rolling" and _const(_kw(n, "center")) is True]
    return [("centered_window", "centred rolling window -> trailing window", ed)] if ed else []


def _gen_backfill(tree, stmt, ctx):
    ed = []
    for n in _nodes_in(stmt):
        if not _call(n):
            continue
        a = n.func.attr
        if a in ("bfill", "backfill"):
            ed.append((n, _src(_with_attr(n, "ffill"))))
        elif a in ("fillna", "reindex", "asfreq") and _const(_kw(n, "method")) in ("bfill", "backfill", "nearest"):
            ed.append((n, _src(_set_kw(n, "method", ast.Constant("ffill")))))
        elif a == "interpolate":
            ed.append((n, _src(_with_attr(n, "ffill", keep_args=False))))
    return [("backward_fill", "backward fill / interpolation -> forward fill", ed)] if ed else []


def _gen_rank(tree, stmt, ctx):
    xs, ex = [], []
    explicit_time = False
    for n in _nodes_in(stmt):
        if not (_call(n) and n.func.attr == "rank"):
            continue
        recv = n.func.value
        if (isinstance(recv, ast.Attribute) and recv.attr == "T") or any(c in WINDOWED for c in _chain_names(recv)):
            continue
        if isinstance(recv, ast.Subscript) and isinstance(recv.value, ast.Attribute) \
                and recv.value.attr in ("iloc", "loc", "xs") and not isinstance(recv.slice, (ast.Slice, ast.Tuple)):
            continue  # a single row: rank across tickers already
        ax = _axis_of(n, positional=0)
        if ax != "absent" and ax not in ROW_AXIS:
            continue
        explicit_time = explicit_time or ax != "absent"
        xs.append((n, _src(_set_kw(n, "axis", ast.Constant(1), positional=0))))
        e = _set_kw(n, "axis", None, positional=0)
        e.func.value = ast.Call(ast.Attribute(copy.deepcopy(recv), "expanding", ast.Load()), [],
                                [ast.keyword("min_periods", ast.Constant(ctx["min_periods"]))])
        ex.append((n, _src(e)))
    if not xs:
        return []
    cs = ("time_axis_rank", "rank over the time axis -> rank across tickers on each date (axis=1)", xs)
    ts = ("time_axis_rank", "rank against the full history -> rank against the history so far (expanding)", ex)
    return [ts, cs] if explicit_time else [cs, ts]


def _is_time_reducer(n) -> bool:
    if not (_call(n) and n.func.attr in REDUCERS):
        return False
    if _axis_of(n) not in ("absent",) + ROW_AXIS:
        return False
    if n.func.attr == "quantile":
        if len(n.args) > 1:
            return False
    elif n.args:
        return False
    chain = _chain_names(n.func.value)
    if any(c in WINDOWED for c in chain) or any(c in ("stack", "melt", "unstack") for c in chain):
        return False
    base = n.func.value
    # x.iloc[-1] / x.iloc[-k:] are row selections; their reductions are not fixed by expanding()
    if isinstance(base, ast.Subscript):
        return False
    return True


def _gen_full_sample(tree, stmt, ctx):
    """X.mean() over time -> X.expanding().mean(); also the cross-sectional alternative for demeaning."""
    reds = [n for n in _nodes_in(stmt) if _is_time_reducer(n)]
    if not reds:
        return []
    exp_edits = []
    for n in reds:
        e = copy.deepcopy(n)
        e.keywords = [k for k in e.keywords if k.arg != "axis"]
        e.func.value = ast.Call(ast.Attribute(copy.deepcopy(n.func.value), "expanding", ast.Load()), [],
                                [ast.keyword("min_periods", ast.Constant(ctx["min_periods"]))])
        exp_edits.append((n, _src(e)))
    expanding = ("full_sample_statistic", "full-sample statistic -> expanding (point-in-time) statistic", exp_edits)

    # cross-sectional alternative: rewrite the statement value with a transformer
    cs_intent = False

    class CS(ast.NodeTransformer):
        changed = False

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, (ast.Sub, ast.Div)) and _is_time_reducer(node.right):
                r = copy.deepcopy(node.right)
                r.keywords = [k for k in r.keywords if k.arg != "axis"] + [ast.keyword("axis", ast.Constant(1))]
                meth = "sub" if isinstance(node.op, ast.Sub) else "div"
                CS.changed = True
                return ast.Call(ast.Attribute(node.left, meth, ast.Load()), [r],
                                [ast.keyword("axis", ast.Constant(0))])
            return node

        def visit_Call(self, node):
            nonlocal cs_intent
            self.generic_visit(node)
            if (_call(node) and node.func.attr in ("sub", "subtract", "div", "divide", "truediv")
                    and node.args and _is_time_reducer(node.args[0])):
                if _axis_of(node, positional=1) in COL_AXIS:
                    cs_intent = True
                r = copy.deepcopy(node.args[0])
                r.keywords = [k for k in r.keywords if k.arg != "axis"] + [ast.keyword("axis", ast.Constant(1))]
                c = _set_kw(node, "axis", ast.Constant(0), positional=1)
                c.args = [r] + c.args[1:]
                CS.changed = True
                return c
            return node

    value = getattr(stmt, "value", None)
    out = [expanding]
    if value is not None:
        new = CS().visit(copy.deepcopy(value))
        if CS.changed:
            cross = ("time_axis_demean", "per-ticker full-sample demeaning -> cross-sectional (per date)",
                     [(value, _src(new))])
            out = [cross, expanding] if cs_intent else [expanding, cross]
    return out


def _end_anchor(rule: str) -> str | None:
    import re

    import pandas as pd
    m = re.match(r"^(\d*)([A-Z]+)(-[A-Z]{3})?$", rule)
    if not m or m.group(2) not in END_ANCHOR:
        return None
    end = END_ANCHOR[m.group(2)]
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    if (major, minor) < (2, 2):  # old aliases before pandas 2.2
        end = {"ME": "M", "BME": "BM", "SME": "SM", "QE": "Q", "BQE": "BQ", "YE": "A", "BYE": "BA"}[end]
    return m.group(1) + end + (m.group(3) or "")


def _gen_resample(tree, stmt, ctx):
    rename, lag, label = [], [], []
    for n in _nodes_in(stmt):
        if not _call(n):
            continue
        inner = n.func.value
        if not (_call(inner) and inner.func.attr == "resample"):
            continue
        rule_arg = inner.args[0] if inner.args else _kw(inner, "rule")
        rs = _const(rule_arg)
        if not (isinstance(rs, str) and START_ANCHORED.match(rs)) or _const(_kw(inner, "label")) == "right":
            continue
        end = _end_anchor(rs)
        if end:
            rename.append((rule_arg, _src(ast.Constant(end))))
        lag.append((n, _lagged(n)))
        li = _set_kw(_set_kw(inner, "closed", ast.Constant("right")), "label", ast.Constant("right"))
        label.append((inner, _src(li)))
    out = []
    if rename:
        out.append(("resample_start_label", "start-labelled resample -> end-labelled bins", rename))
    if lag:
        out.append(("resample_start_label", "shift the resampled aggregate by one bin", lag))
    if label:
        out.append(("resample_start_label", "resample(label='right', closed='right')", label))
    return out


def _gen_period_transform(tree, stmt, ctx):
    ed = []
    for n in _nodes_in(stmt):
        if _call(n) and n.func.attr == "transform" and n.args and isinstance(_const(n.args[0]), str) \
                and _const(n.args[0]) != "first" and _call(n.func.value) and n.func.value.func.attr == "groupby":
            c = copy.deepcopy(n)
            c.args[0] = ast.Constant("first")
            ed.append((n, _src(c)))
    return [("period_transform", "per-period aggregate broadcast to every day -> value known on the period's "
                                 "first day (transform('first'))", ed)] if ed else []


class _Unreverse(ast.NodeTransformer):
    changed = False

    def _strip(self, node):
        if _is_reversal(node):
            _Unreverse.changed = True
            if isinstance(node, ast.Subscript):
                v = node.value
                return v.value if isinstance(v, ast.Attribute) and v.attr in ("iloc", "loc") else v
            return node.func.value
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        return self._strip(node)


def _gen_reversed(tree, stmt, ctx):
    edits = []
    value = getattr(stmt, "value", None)
    if value is None:
        return []
    _Unreverse.changed = False
    new = _Unreverse().visit(copy.deepcopy(value))
    if _Unreverse.changed:
        edits.append((value, _src(new)))
    # a reversed frame defined earlier (recent_first = close.sort_index(ascending=False)) and used here
    assigns = ctx["assigns"]
    for n in _nodes_in(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in assigns:
            a = assigns[n.id]
            if a is stmt or not _reversed_parity(a.value, set()):
                continue
            _Unreverse.changed = False
            nv = _Unreverse().visit(copy.deepcopy(a.value))
            if _Unreverse.changed:
                edits.append((a.value, _src(nv)))
    return [("reversed_time_window", "remove the time reversal around the window (trailing, oldest-first)",
             edits)] if edits else []


def _gen_pooled(tree, stmt, ctx):
    out = []
    assigns = ctx["assigns"]
    stacked = ctx["stacked"]
    per_date = []
    for n in _nodes_in(stmt):
        if isinstance(n, ast.Call) and (getattr(n.func, "attr", None) == "qcut" or getattr(n.func, "id", None) == "qcut") \
                and n.args and _pooled_panel(n.args[0], stacked):
            lam_call = copy.deepcopy(n)
            lam_call.args[0] = ast.Name("_s", ast.Load())
            new = f"{_src(n.args[0])}.groupby(level=0).transform(lambda _s: {_src(lam_call)})"
            per_date.append((n, new))
    if per_date:
        out.append(("pooled_quantile", "pooled quantile breakpoints -> breakpoints per date", per_date))
    trailing = []
    for n in _nodes_in(stmt):
        if not (_call(n) and n.func.attr in ("quantile", "median", "mean", "std")):
            continue
        g = n.func.value
        if not (_call(g) and g.func.attr == "groupby" and _const(_kw(g, "level")) not in (0, None)):
            continue
        base = g.func.value
        src_frame = None
        if isinstance(base, ast.Name) and base.id in assigns:
            v = assigns[base.id].value
            if _call(v) and v.func.attr == "stack":
                src_frame = v.func.value
        elif _call(base) and base.func.attr == "stack":
            src_frame = base.func.value
        if src_frame is None:
            continue
        e = copy.deepcopy(n)
        e.func.value = ast.Call(ast.Attribute(copy.deepcopy(src_frame), "expanding", ast.Load()), [],
                                [ast.keyword("min_periods", ast.Constant(ctx["min_periods"]))])
        trailing.append((n, _src(e)))
    if trailing:
        out.append(("pooled_quantile", "per-ticker full-sample threshold -> expanding per-ticker threshold",
                    trailing))
    return out


def _gen_survivor(tree, stmt, ctx):
    ed = []
    for n in _nodes_in(stmt):
        if _call(n) and n.func.attr == "dropna":
            ax = _kw(n, "axis") or (n.args[0] if n.args else None)
            if _const(ax) in (1, "columns"):
                ed.append((n, _src(n.func.value)))
    return [("survivor_universe", "drop dropna(axis=1): keep every ticker, point-in-time", ed)] if ed else []


def _gen_future_bar(tree, stmt, ctx):
    """backtrader (audit/adapters.py): a positive line index reads the next bar -> read the current bar."""
    from audit.ast_scan import _pos_int, _rooted_at_self
    edits = []
    for n in _nodes_in(stmt):
        if isinstance(n, ast.Subscript) and _rooted_at_self(n.value) and _pos_int(n.slice) > 0 and not (
                isinstance(n.value, ast.Attribute) and n.value.attr in ("datas", "dnames")):
            edits.append((n.slice, "0"))
        elif _call(n) and n.func.attr == "get" and _rooted_at_self(n.func):
            a = _kw(n, "ago") or (n.args[0] if n.args else None)
            if a is not None and _pos_int(a) > 0:
                edits.append((a, "0"))
    return [("future_bar_index", "backtrader: read the current bar [0] instead of a future bar", edits)] if edits \
        else []


def _gen_lag(tree, stmt, ctx):
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return [("lag_statement", "lag the localized value by one bar (.shift(1))",
                 [(stmt.value, _lagged(stmt.value))])]
    return []


GENERATORS = [_gen_negative_period, _gen_centered, _gen_backfill, _gen_reversed, _gen_period_transform,
              _gen_resample, _gen_pooled, _gen_rank, _gen_full_sample, _gen_survivor, _gen_future_bar]


def generate_candidates(code: str, lines: list[int], *, fallback_lines: list[int] | None = None,
                        min_periods: int = 20) -> list[Candidate]:
    """Candidate patches for the given source lines, best first, de-duplicated, each a full new source."""
    tree = ast.parse(code)
    assigns = _assignments(tree)
    stacked = {k for k, a in assigns.items() if _pooled_panel(a.value, set())}
    ctx = {"assigns": assigns, "stacked": stacked, "min_periods": min_periods}
    per_line: list[list[tuple]] = []
    seen_stmts = set()
    for ln in lines:
        st = _stmt_at(tree, ln)
        if st is None or id(st) in seen_stmts:
            continue
        seen_stmts.add(id(st))
        opts = []
        for g in GENERATORS:
            try:
                opts += [(r, d, e, st.lineno) for r, d, e in g(tree, st, ctx)]
            except Exception:  # noqa: BLE001 - a generator that cannot handle an odd node just yields nothing
                continue
        per_line.append(opts)
    out, seen = [], {code}

    def add(rule, desc, edits, lns):
        try:
            new = _apply(code, edits)
            ast.parse(new)
        except Exception:  # noqa: BLE001
            return
        if new not in seen:
            seen.add(new)
            out.append(Candidate(rule, desc, new, sorted({n.lineno for n, _ in edits})))

    # several flagged lines (independent leaks usually need all of them fixed): try the best option of each
    # together first, so a multi-leak draft costs one sandboxed audit instead of one per single-line patch
    firsts = [o[0] for o in per_line if o]
    if len(firsts) > 1:
        add("+".join(dict.fromkeys(o[0] for o in firsts)), "combined: " + "; ".join(o[1] for o in firsts),
            [x for o in firsts for x in o[2]], [o[3] for o in firsts])
    for opts in per_line:
        for r, d, e, ln in opts:
            add(r, d, e, [ln])
    # last resort: lag the localized statement
    for ln in (fallback_lines or []):
        st = _stmt_at(tree, ln)
        if st is not None:
            for r, d, e in _gen_lag(tree, st, ctx):
                add(r, d, e, [st.lineno])
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def _blocking(rep) -> list[str]:
    return [t.name for t in rep.tests if t.blocking]


def _over_corrected(rep) -> bool:
    nt = next((t for t in rep.tests if t.name == "non_trivial"), None)
    return bool((nt is not None and not nt.passed) or rep.metrics.get("avg_gross", 0.0) < MIN_GROSS)


def target_lines(report_tests: list[dict], code: str) -> tuple[list[int], list[int]]:
    """(lines to patch, best first; localized lines for the lag fallback) from audit test dicts."""
    lines, loc_lines = [], []
    fp = next((t for t in report_tests if t["name"] == "future_perturbation"), None)
    if fp and not fp["passed"]:
        lz = (fp.get("evidence") or {}).get("localized") or {}
        if lz.get("found") and lz.get("entry"):
            lines.append(int(lz["entry"]["line"]))
            loc_lines.append(int(lz["entry"]["line"]))
    sv = next((t for t in report_tests if t["name"] == "survivorship"), None)
    lookahead = bool(fp and not fp["passed"])
    survivor = bool(sv and not sv["passed"] and sv["severity"] in ("critical", "high"))
    for f in scan_code(code):
        if (lookahead and f.rule in LOOKAHEAD_RULE_IDS) or (survivor and f.rule in SURVIVORSHIP_RULES) \
                or f.severity in ("critical", "high"):
            if f.line not in lines:
                lines.append(f.line)
    return lines, loc_lines


def unified_diff(old: str, new: str, name: str = "strategy.py") -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=f"a/{name}", tofile=f"b/{name}"))


def _summary(rep) -> dict:
    return {"clean": rep.clean, "blocking": _blocking(rep),
            "sharpe": round(float(rep.metrics.get("sharpe", 0.0)), 4),
            "cagr": round(float(rep.metrics.get("cagr", 0.0)), 4),
            "avg_gross": round(float(rep.metrics.get("avg_gross", 0.0)), 4)}


def quick_fix(code: str, data: dict, *, report=None, seed: int = 7, n_cuts: int = 6, cost_bps: float = 10.0,
              timeout: float = 90.0, max_candidates: int = MAX_CANDIDATES, name: str = "strategy.py",
              executor_factory=None) -> dict:
    """Try deterministic patches until one passes the full audit without going to cash.

    ``report``: an ``AuditReport`` (or the ``to_dict()`` of one) of ``code`` from ``run_audit`` with localization,
    to avoid re-auditing the original. ``executor_factory(code) -> executor`` defaults to the sandbox.

    Returns ``{status, rule, description, lines, diff, code, before, after, attempts, n_candidates, seconds,
    cost_usd}`` with status ``fixed`` | ``not_needed`` | ``no_candidate`` | ``failed``; when fixed, also
    ``report``: the accepted candidate's ``AuditReport`` (not JSON-serialisable; use ``public(result)``).
    """
    from audit.leak_tests import run_audit
    from sandbox.runner import SandboxError, SandboxExecutor, check_policy

    t0 = time.time()
    make = executor_factory or (lambda c: SandboxExecutor(c, timeout))
    out = {"status": "failed", "rule": None, "description": "", "lines": [], "diff": "", "code": None,
           "before": None, "after": None, "attempts": [], "n_candidates": 0, "cost_usd": 0.0, "method": "quick_fix"}
    if report is None:
        report = run_audit(data, make(code), code, seed=seed, n_cuts=n_cuts, cost_bps=cost_bps)
    rd = report if isinstance(report, dict) else report.to_dict()
    tests = rd["tests"]
    blocking = [t["name"] for t in tests if t.get("blocking")]
    m = rd.get("metrics") or {}
    out["before"] = {"clean": not blocking, "blocking": blocking, "sharpe": round(float(m.get("sharpe", 0.0)), 4),
                     "cagr": round(float(m.get("cagr", 0.0)), 4),
                     "avg_gross": round(float(m.get("avg_gross", 0.0)), 4)}
    if not blocking:
        out.update(status="not_needed", seconds=round(time.time() - t0, 2))
        return out
    try:
        lines, loc_lines = target_lines(tests, code)
        cands = generate_candidates(code, lines, fallback_lines=loc_lines)[:max_candidates]
    except SyntaxError:
        cands = []
    out["n_candidates"] = len(cands)
    if not cands:
        out.update(status="no_candidate", seconds=round(time.time() - t0, 2))
        return out
    for c in cands:
        att = {"rule": c.rule, "description": c.description, "lines": c.lines}
        if check_policy(c.code):
            att["result"] = "policy_violation"
            out["attempts"].append(att)
            continue
        try:
            rep = run_audit(data, make(c.code), c.code, seed=seed, n_cuts=n_cuts, cost_bps=cost_bps, localize=False)
        except (SandboxError, RuntimeError, ValueError) as e:
            att["result"] = "error"
            att["error"] = f"{type(e).__name__}: {e}"[:300]
            out["attempts"].append(att)
            continue
        s = _summary(rep)
        att |= {"blocking": s["blocking"], "sharpe": s["sharpe"], "avg_gross": s["avg_gross"]}
        if not rep.clean:
            att["result"] = "still_failing"
        elif _over_corrected(rep):
            att["result"] = "over_corrected"
        else:
            att["result"] = "accepted"
            out["attempts"].append(att)
            out.update(status="fixed", rule=c.rule, description=c.description, lines=c.lines, code=c.code,
                       diff=unified_diff(code, c.code, name), after=s, report=rep)
            break
        out["attempts"].append(att)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def public(qf: dict | None) -> dict:
    """The JSON-safe part of a quick_fix result (drops the AuditReport object)."""
    return {k: v for k, v in (qf or {}).items() if k != "report"}


def quickfix_message(qf: dict | None) -> str:
    """One line for humans / traces."""
    if not qf:
        return ""
    st = qf.get("status")
    if st == "fixed":
        b, a = qf["before"], qf["after"]
        return (f"Quick fix ({qf['rule']}, line {', '.join(map(str, qf['lines']))}): {qf['description']}. "
                f"Sharpe {b['sharpe']:.2f} -> {a['sharpe']:.2f}; all blocking tests pass; no LLM, $0.")
    if st == "not_needed":
        return "Quick fix: nothing to fix (no blocking test fails)."
    if st == "no_candidate":
        return "Quick fix: no deterministic rewrite applies to the flagged lines."
    return f"Quick fix: none of {qf.get('n_candidates', 0)} deterministic candidate(s) passed the audit."

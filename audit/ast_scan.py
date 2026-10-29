"""Static AST scan for common look-ahead / leakage red flags.

Static findings are *hints*; the dynamic tests are the proof. Severity is set
so that patterns that are almost always bugs (negative shifts, centred windows,
backward fills) block the loop, while softer smells only warn.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass
class Finding:
    rule: str
    line: int
    snippet: str
    severity: str
    message: str
    fix: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _neg_const(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return isinstance(node.operand.value, (int, float)) and node.operand.value > 0
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    return False


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _chain_names(node: ast.AST) -> list[str]:
    """Method names along an attribute/call chain, e.g. x.rolling(5).mean() -> [mean, rolling]."""
    names = []
    cur = node
    while True:
        if isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, ast.Attribute):
            names.append(cur.attr)
            cur = cur.value
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        else:
            break
    return names


WINDOW_METHODS = {"rolling", "expanding", "ewm", "groupby", "resample", "cummax", "cummin", "cumsum"}
FULL_SAMPLE_STATS = {"mean", "std", "var", "median", "min", "max", "quantile", "sum", "skew", "kurt"}


def scan_code(code: str) -> list[Finding]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Finding("syntax_error", e.lineno or 0, str(e.text or "").strip(), "critical",
                        f"Code does not parse: {e.msg}", "Return valid Python.")]
    lines = code.splitlines()

    def snip(n: ast.AST) -> str:
        ln = getattr(n, "lineno", 0)
        return lines[ln - 1].strip() if 0 < ln <= len(lines) else ""

    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        first = node.args[0] if node.args else None

        if attr in ("shift", "pct_change", "diff"):
            periods = first if first is not None else _kw(node, "periods")
            if _neg_const(periods):
                out.append(Finding(
                    f"negative_{attr}", node.lineno, snip(node), "critical",
                    f".{attr}() with a negative period pulls FUTURE rows into row t.",
                    f"Use a non-negative period (e.g. .{attr}(1)); the engine already trades at t+1."))

        if attr == "rolling" and _is_true(_kw(node, "center")):
            out.append(Finding(
                "centered_window", node.lineno, snip(node), "high",
                "rolling(center=True) averages over future observations.",
                "Use a trailing window: rolling(n) without center=True."))

        if attr in ("bfill", "backfill"):
            out.append(Finding(
                "backward_fill", node.lineno, snip(node), "high",
                f".{attr}() copies future values backwards in time.",
                "Use .ffill() (or leave NaN) so row t only sees data up to t."))
        if attr in ("fillna", "reindex", "interpolate"):
            m = _kw(node, "method")
            if isinstance(m, ast.Constant) and str(m.value) in ("bfill", "backfill", "nearest"):
                out.append(Finding(
                    "backward_fill", node.lineno, snip(node), "high",
                    f"{attr}(method='{m.value}') fills from the future.",
                    "Use method='ffill' or no filling."))
            if attr == "interpolate":
                out.append(Finding(
                    "interpolation", node.lineno, snip(node), "medium",
                    "interpolate() uses the next valid value to fill gaps.",
                    "Use ffill() instead of interpolation."))

        if attr == "dropna":
            axis = _kw(node, "axis")
            if axis is None and first is not None:
                axis = first
            if isinstance(axis, ast.Constant) and axis.value in (1, "columns"):
                out.append(Finding(
                    "survivor_universe", node.lineno, snip(node), "high",
                    "dropna(axis=1) keeps only tickers with a complete history, i.e. names that "
                    "survive to the end of the sample (survivorship bias).",
                    "Build a point-in-time universe: eligible = close.notna() on each date."))

        if attr in FULL_SAMPLE_STATS:
            axis = _kw(node, "axis")
            cross_sectional = isinstance(axis, ast.Constant) and axis.value in (1, "columns")
            chain = _chain_names(node.func.value)
            windowed = any(c in WINDOW_METHODS for c in chain)
            if not cross_sectional and not windowed and node.args == [] and attr != "sum":
                out.append(Finding(
                    "full_sample_statistic", node.lineno, snip(node), "medium",
                    f".{attr}() over the time axis uses the whole sample, including the future.",
                    f"Use a trailing estimate (.rolling(n).{attr}() or .expanding().{attr}()) "
                    f"or a cross-sectional one (axis=1)."))
    out += _pandas_trap_rules(tree, snip)
    out += future_bar_rules(tree, snip)
    # The generic full_sample_statistic finding is kept next to the more specific rules on purpose:
    # downstream tools (audit_cli suspect-line mapping) key on it.
    # de-duplicate (same rule on the same line)
    seen, uniq = set(), []
    for f in sorted(out, key=lambda f: (f.line, f.rule)):
        if (f.rule, f.line) not in seen:
            seen.add((f.rule, f.line))
            uniq.append(f)
    return uniq


# ----------------------------------------------------------------------------
# pandas default-axis / labelling traps (all "medium": hints for the Auditor, never blocking on their own)
# ----------------------------------------------------------------------------
# Rule ids that can explain a future_perturbation failure (for pointing at source lines).
LOOKAHEAD_RULE_IDS = {"negative_shift", "negative_pct_change", "negative_diff", "centered_window", "backward_fill",
                      "interpolation", "full_sample_statistic", "survivor_universe", "time_axis_rank",
                      "time_axis_demean", "full_sample_imputation", "resample_start_label", "period_transform",
                      "reversed_time_window", "pooled_quantile", "future_bar_index"}

ROW_AXIS = (0, "index", "rows")
COL_AXIS = (1, "columns")
REDUCERS = {"mean", "std", "median", "var"}
TIME_WINDOWS = {"rolling", "expanding", "ewm", "groupby", "resample"}
START_ANCHORED = re.compile(r"^\d*(MS|BMS|SMS|(B?(QS|AS|YS))(-[A-Z]{3})?)$")
RESAMPLE_AGGS = {"last", "max", "min", "mean", "median", "sum", "std", "var", "ohlc", "nunique", "count", "prod",
                 "quantile"}
PERIOD_AGGS = {"last", "max", "min", "mean", "median", "sum", "std", "var", "nunique", "count", "prod", "idxmax",
               "idxmin", "quantile", "size", "nth"}
PERIOD_ATTRS = {"to_period", "month", "year", "quarter", "week", "weekofyear", "isocalendar", "dayofyear", "is_month_end"}
ORDER_SENSITIVE = {"rolling", "expanding", "ewm", "cumsum", "cummax", "cummin", "cumprod", "ffill", "pad", "shift",
                   "pct_change", "diff"}


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _axis_of(call: ast.Call, positional: int | None = None):
    """'absent', or the constant axis value (None when it is not a constant)."""
    a = _kw(call, "axis")
    if a is None and positional is not None and len(call.args) > positional:
        a = call.args[positional]
    if a is None:
        return "absent"
    return _const(a)


def _chain_nodes(node: ast.AST) -> list[ast.AST]:
    """Nodes along a call/attribute/subscript chain from outermost to the base expression (inclusive)."""
    out = [node]
    cur = node
    while True:
        if isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, ast.Attribute):
            cur = cur.value
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        else:
            break
        out.append(cur)
    return out


def _base_name(node: ast.AST) -> str | None:
    b = _chain_nodes(node)[-1]
    return b.id if isinstance(b, ast.Name) else None


def _time_reducer(node: ast.AST):
    """X.mean()/std()/median()/var() reduced over the time axis (no axis / axis=0), not windowed. Returns X."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in REDUCERS):
        return None
    if node.args or _axis_of(node) not in ("absent",) + ROW_AXIS:
        return None
    if any(c in TIME_WINDOWS for c in _chain_names(node.func.value)):
        return None
    return node.func.value


def _is_reversal(n: ast.AST) -> bool:
    if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
        st = n.slice.step
        return (n.slice.lower is None and n.slice.upper is None and st is not None and _neg_const(st)
                and ((isinstance(st, ast.UnaryOp) and st.operand.value == 1) or _const(st) == -1))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sort_index":
        return _const(_kw(n, "ascending")) is False
    return False


def _is_order_reset(n: ast.AST) -> bool:
    return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sort_index"
            and _const(_kw(n, "ascending")) is not False)


def _reversed_parity(expr: ast.AST, reversed_names: set[str]) -> bool:
    """True when expr is a time-reversed frame: odd number of reversals since the last re-sort."""
    flips = 0
    for n in _chain_nodes(expr):
        if _is_order_reset(n):
            return flips % 2 == 1
        if _is_reversal(n):
            flips += 1
    b = _chain_nodes(expr)[-1]
    if isinstance(b, ast.Name) and b.id in reversed_names:
        flips += 1
    return flips % 2 == 1


def _period_key(expr: ast.AST, assigned: dict) -> bool:
    if isinstance(expr, ast.Name) and expr.id in assigned:
        expr = assigned[expr.id]
    for n in ast.walk(expr):
        if isinstance(n, ast.Attribute) and n.attr in PERIOD_ATTRS:
            return True
        if (isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "Grouper"
                and _kw(n, "freq") is not None):
            return True
    return False


def _pandas_trap_rules(tree: ast.AST, snip) -> list[Finding]:
    out: list[Finding] = []
    # one-level name resolution, in source order (good enough for strategy functions)
    assigned: dict[str, ast.AST] = {}
    reversed_names: set[str] = set()
    stacked_names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            assigned[n.targets[0].id] = n.value
    for name, val in assigned.items():
        if _reversed_parity(val, set()):
            reversed_names.add(name)
        if _pooled_panel(val, set()):
            stacked_names.add(name)

    def add(rule, node, message, fix):
        out.append(Finding(rule, node.lineno, snip(node), "medium", message, fix))

    for node in ast.walk(tree):
        # X - X.mean(), (X - X.mean()) / X.std()
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Div)):
            recv = _time_reducer(node.right)
            if recv is not None:
                key = ast.dump(recv)
                # X itself must appear on the left, not only inside another reduction (book.mean() / book.std())
                parents = {id(c): p for p in ast.walk(node.left) for c in ast.iter_child_nodes(p)}
                if any(ast.dump(s) == key and not (isinstance(parents.get(id(s)), ast.Attribute)
                                                    and parents[id(s)].attr in FULL_SAMPLE_STATS)
                       for s in ast.walk(node.left)):
                    add("time_axis_demean", node,
                        f"`{ast.unparse(node.right)}` is a per-column statistic over the WHOLE sample (pandas reduces "
                        "over the time axis by default), so every row is normalised with future data.",
                        "For a cross-sectional normalisation use X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1), axis=0); "
                        "for a per-ticker one use trailing stats: X.rolling(n).mean() / X.expanding(min_periods=n).std().")
        if not (isinstance(node, ast.Call)):
            continue
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        first = node.args[0] if node.args else None

        # DataFrame.rank() over the time axis
        if fname == "rank" and isinstance(node.func, ast.Attribute):
            recv = node.func.value
            ax = _axis_of(node, positional=0)
            transposed = (isinstance(recv, ast.Attribute) and recv.attr == "T") or (
                isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr == "transpose")
            windowed = any(c in {"rolling", "expanding", "groupby", "resample"} for c in _chain_names(recv))
            row_select = (isinstance(recv, ast.Subscript) and isinstance(recv.value, ast.Attribute)
                          and recv.value.attr in ("iloc", "loc", "xs") and not isinstance(recv.slice, (ast.Slice, ast.Tuple)))
            if not transposed and not windowed and not row_select and (ax == "absent" or ax in ROW_AXIS):
                add("time_axis_rank", node,
                    ("DataFrame.rank() without axis ranks DOWN each column (axis=0): every value is ranked against "
                     "the ticker's whole history, including the future." if ax == "absent" else
                     "rank(axis=0) ranks each value against the ticker's whole history, including the future."),
                    "Rank across tickers on each date with .rank(axis=1, pct=True); for a time-series percentile use "
                    "a trailing window, e.g. X.rolling(252, min_periods=60).rank(pct=True).")

        # X.sub(X.mean(), axis=1) / X.div(X.std()) : aligns a per-column (time) statistic onto every row
        if fname in ("sub", "subtract", "div", "divide", "truediv") and first is not None:
            recv = _time_reducer(first)
            ax = _axis_of(node, positional=1)
            if recv is not None and (ax == "absent" or ax in COL_AXIS):
                add("time_axis_demean", node,
                    f"`{ast.unparse(first)}` reduces over the time axis (one number per ticker from the WHOLE sample); "
                    f".{fname}(..., axis=1) then applies it to every date, so past rows use future data.",
                    f"Cross-sectional: X.{fname}(X.{first.func.attr}(axis=1), axis=0). Per ticker: use a trailing "
                    f"estimate, X.{fname}(X.expanding(min_periods=n).{first.func.attr}()).")

        # X.fillna(X.mean()) : imputation with a full-sample statistic
        if fname == "fillna" and first is not None and _time_reducer(first) is not None:
            add("full_sample_imputation", node,
                f"fillna({ast.unparse(first)}) imputes every gap with a per-ticker statistic of the WHOLE sample, "
                "including values observed after the gap.",
                "Leave the value NaN (rank/where will skip it), impute cross-sectionally with "
                "X.T.fillna(X.mean(axis=1)).T, or use a trailing estimate (X.fillna(X.expanding().mean().shift(1))).")

        # resample('MS').last(): start-labelled bins carry end-of-period data
        if isinstance(node.func, ast.Attribute):
            inner = node.func.value
            agg_ok = fname in RESAMPLE_AGGS or (fname in ("agg", "aggregate", "apply") and first is not None
                                                and _const(first) in RESAMPLE_AGGS)
            if (agg_ok and isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "resample"):
                rule_arg = inner.args[0] if inner.args else _kw(inner, "rule")
                rs = _const(rule_arg)
                label = _const(_kw(inner, "label"))
                if isinstance(rs, str) and START_ANCHORED.match(rs) and label != "right":
                    add("resample_start_label", node,
                        f"resample('{rs}') labels each bin with the period START, but .{fname}() summarises the whole "
                        "period; after reindex/ffill the end-of-period value is visible from the first day.",
                        "Use an end-labelled rule (resample('ME'/'BME'/'QE'/'YE'/'W-FRI')), or shift the result one "
                        "bin (.shift(1)) before mapping it back to daily dates, or pass label='right', closed='right'.")

        # groupby(<period key>).transform('last'/'max'/'mean'...)
        if fname == "transform" and isinstance(node.func, ast.Attribute) and first is not None \
                and _const(first) in PERIOD_AGGS:
            g = node.func.value
            if isinstance(g, ast.Call) and isinstance(g.func, ast.Attribute) and g.func.attr == "groupby":
                key = g.args[0] if g.args else _kw(g, "by")
                if key is not None and _period_key(key, assigned):
                    add("period_transform", node,
                        f"groupby(<period>).transform('{_const(first)}') gives EVERY day of the period a value that "
                        "depends on the whole period, including days after it.",
                        "Use transform('first') (the value known on the period's first day), or a causal "
                        "cumulative version (groupby(...).cummax()/cumsum(), expanding()), or shift the period "
                        "aggregate by one period before broadcasting it.")

        # sort_index(ascending=False) / .iloc[::-1] followed by an order-sensitive window
        if fname in ORDER_SENSITIVE and isinstance(node.func, ast.Attribute):
            if _reversed_parity(node.func.value, reversed_names):
                add("reversed_time_window", node,
                    f".{fname}() runs on a newest-first (time-reversed) frame, so each 'trailing' window or "
                    "cumulative value is computed from FUTURE rows.",
                    f"Keep the index ascending (sort_index()) before .{fname}(); trailing windows must end at row t.")

        # pooled quantiles on a stacked panel
        if fname in ("qcut",) and first is not None and _pooled_panel(first, stacked_names):
            add("pooled_quantile", node,
                "pd.qcut on a stacked (date x ticker) panel computes breakpoints from all dates at once, "
                "including the future.",
                "Bucket each date separately: X.rank(axis=1, pct=True), or "
                "stacked.groupby(level=0).transform(lambda s: pd.qcut(s, 5, labels=False)).")
        if fname in ("quantile", "median") and isinstance(node.func, ast.Attribute):
            recv = node.func.value
            if _pooled_panel(recv, stacked_names):
                add("pooled_quantile", node,
                    f".{fname}() on a stacked panel pools observations from every date (and, after "
                    "groupby(ticker), the ticker's full history), so the threshold uses future data.",
                    "Use per-date (cross-sectional) quantiles: X.quantile(q, axis=1) or "
                    "stacked.groupby(level=0).quantile(q); for a per-ticker threshold use "
                    "X.expanding(min_periods=n).quantile(q).shift(1).")
    return out


def _pooled_panel(expr: ast.AST, stacked_names: set[str]) -> bool:
    """expr is a stacked / flattened panel not regrouped by date (groupby(level=0))."""
    nodes = _chain_nodes(expr)
    for n in nodes:
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "groupby":
            lvl = _const(_kw(n, "level"))
            if lvl == 0 or (isinstance(lvl, str) and "date" in lvl.lower()):
                return False
    names = [n.func.attr for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    if any(a in ("stack", "melt", "ravel", "flatten") for a in names):
        return True
    b = nodes[-1]
    return isinstance(b, ast.Name) and b.id in stacked_names


# ----------------------------------------------------------------------------
# backtrader line indexing (audit/adapters.py): line[1] is the NEXT bar
# ----------------------------------------------------------------------------
def _pos_int(node) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd) and isinstance(node.operand, ast.Constant):
        return node.operand.value if isinstance(node.operand.value, int) else 0
    return 0


def _rooted_at_self(node) -> bool:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return isinstance(node, ast.Name) and node.id == "self"


def future_bar_rules(tree: ast.Module, snip) -> list[Finding]:
    """In a module importing backtrader: ``self.<line>[k]`` with k > 0, ``.get(ago=k>0)``, ``.date(k>0)``."""
    uses_bt = any((isinstance(n, ast.Import) and any(a.name.split(".")[0] == "backtrader" for a in n.names)) or
                  (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "backtrader")
                  for n in ast.walk(tree))
    if not uses_bt:
        return []
    out = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in ast.walk(cls):
            k, what = 0, ""
            if isinstance(node, ast.Subscript) and _rooted_at_self(node.value):
                inner = node.value
                if isinstance(inner, ast.Attribute) and inner.attr in ("datas", "dnames"):
                    continue          # self.datas[1] is the second feed, not the future
                k, what = _pos_int(node.slice), f"[{_pos_int(node.slice)}]"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _rooted_at_self(node.func):
                if node.func.attr == "get":
                    k = _pos_int(_kw(node, "ago") or (node.args[0] if node.args else None))
                    what = f".get(ago={k})"
                elif node.func.attr in ("date", "datetime", "time"):
                    k = _pos_int(node.args[0] if node.args else _kw(node, "ago"))
                    what = f".{node.func.attr}({k})"
            if k > 0:
                out.append(Finding(
                    "future_bar_index", node.lineno, snip(node), "critical",
                    f"backtrader line index {what} reads {k} bar(s) AHEAD ([0] is the current bar, [-1] the "
                    f"previous one); with preloaded data backtrader returns the future value.",
                    "Use [0] (current bar) or negative indices ([-1], [-2]) only."))
    return out

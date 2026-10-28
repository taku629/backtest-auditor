"""Static AST scan for common look-ahead / leakage red flags.

Static findings are *hints*; the dynamic tests are the proof. Severity is set
so that patterns that are almost always bugs (negative shifts, centred windows,
backward fills) block the loop, while softer smells only warn.
"""
from __future__ import annotations

import ast
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
    # de-duplicate (same rule on the same line)
    seen, uniq = set(), []
    for f in sorted(out, key=lambda f: (f.line, f.rule)):
        if (f.rule, f.line) not in seen:
            seen.add((f.rule, f.line))
            uniq.append(f)
    return uniq

"""Strategy format adapters: audit real-world strategy code without rewriting it as ``generate_signals``.

``adapt(code)`` detects the format and returns the code with a small ``generate_signals(data)`` wrapper appended
(the user's lines keep their line numbers, so findings, the leak localizer and Quick fix still point at them).
The wrapper runs inside the sandbox and calls the trusted stubs in ``sandbox/adapter_stubs.py``
(``import bta_adapter`` / ``vectorbt`` / ``backtrader`` resolve to those stubs there; neither library is needed).

Formats
-------
``signals``     ``def generate_signals(data) -> weights``: unchanged (the native contract).
``function``    another entry point: ``strategy(df)``, ``signals(close)``, ``compute_weights(prices)``,
                ``get_positions(data)`` ... Arguments are bound by name: ``data`` -> the dict of OHLCV panels;
                ``close/prices/price/px/df/...`` -> the close panel (dates x tickers); ``open/high/low/volume`` ->
                that panel. The return value may be weights, a boolean or +1/0/-1 signal frame (converted to equal
                weight per date: ``w = s / sum|s|``), a tuple ``(entries, exits)`` or a ``vbt.Portfolio``.
``vectorbt``    a function as above returning ``(entries, exits)`` / a Portfolio, or module-level script code
                that ends in ``vbt.Portfolio.from_signals(close, entries, exits)``. Entries/exits become a long
                position state machine (short_entries/short_exits and ``direction`` are honoured), then equal
                weights over the names in position.
``backtrader``  a ``bt.Strategy`` subclass: replayed bar by bar once per ticker with backtrader's line indexing
                (``[0]`` current bar, ``[-1]`` previous, ``[1]`` the NEXT bar - the look-ahead the audit
                catches). Orders set the ticker's target exposure; weights = exposure / max(1, sum |exposure|).
"""
from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field

FORMATS = ("auto", "signals", "function", "vectorbt", "backtrader")
WRAPPER_MARK = "# --- Backtest Auditor adapter (auto-generated; not part of your strategy) ---"

PREFERRED_NAMES = ("strategy", "signals", "signal", "compute_weights", "generate_weights", "get_weights",
                   "weights", "compute_signals", "get_signals", "generate_positions", "get_positions", "positions",
                   "compute_positions", "run_strategy", "backtest_signals", "entries_exits", "get_entries_exits",
                   "make_signals", "build_signals", "target_weights", "alpha")
CLOSE_PARAMS = {"close", "closes", "prices", "price", "px", "df", "close_df", "prices_df", "adj_close", "adjclose",
                "c", "close_prices", "panel"}
FIELD_PARAMS = {"open": "open", "open_": "open", "opens": "open", "high": "high", "highs": "high", "low": "low",
                "lows": "low", "volume": "volume", "vol": "volume", "volumes": "volume"}
DATA_PARAMS = {"data", "ohlcv", "bars", "market_data", "dataset"}
MODULE_PARAMS = ("data", "close", "open_", "high", "low", "volume", "price", "prices", "df")   # not "open": a blocked name

CAPTIONS = {
    "signals": "Detected: generate_signals(data) (native format)",
    "function": "Detected: {entry}() → adapted",
    "vectorbt": "Detected: vectorbt {entry} → adapted",
    "backtrader": "Detected: backtrader Strategy `{entry}` → adapted",
}


class AdapterError(ValueError):
    """The code is in no format we can adapt (message explains the accepted formats)."""


@dataclass
class Adapted:
    code: str                    # what runs in the sandbox (original + wrapper)
    format: str                  # signals | function | vectorbt | backtrader
    entry: str = ""              # function / class / "module script" that was wrapped
    caption: str = ""
    notes: list[str] = field(default_factory=list)
    original: str = ""
    line_offset: int = 0         # adapted line = original line + offset (only for module-level vectorbt scripts)
    localizable: bool = True     # leak localizer / Quick fix can point at the user's lines
    pbo: bool = True             # PBO grid is affordable / meaningful

    @property
    def adapted(self) -> bool:
        return self.format != "signals"

    def strip(self, code: str) -> str:
        """Remove the appended wrapper again (e.g. from a Quick fix of the adapted code)."""
        if self.line_offset:
            return code
        i = code.find("\n\n" + WRAPPER_MARK)
        return code[:i] + "\n" if i >= 0 else code

    def to_dict(self) -> dict:
        return {"format": self.format, "entry": self.entry, "caption": self.caption, "notes": self.notes,
                "line_offset": self.line_offset, "localizable": self.localizable}


# ---------------------------------------------------------------------------------------------- detection
def _imports(tree: ast.Module) -> dict[str, str]:
    """alias -> root module for vectorbt/backtrader imports anywhere in the module."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                root = a.name.split(".")[0]
                if root in ("vectorbt", "backtrader"):
                    out[(a.asname or a.name).split(".")[0]] = root
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in ("vectorbt", "backtrader"):
            for a in n.names:
                out[a.asname or a.name] = n.module.split(".")[0] + ":" + a.name
    return out


def _strategy_classes(tree: ast.Module, imports: dict[str, str]) -> list[ast.ClassDef]:
    bt_aliases = {k for k, v in imports.items() if v == "backtrader"}
    direct = {k for k, v in imports.items() if v == "backtrader:Strategy"}
    found, names = [], set()
    for n in tree.body:
        if not isinstance(n, ast.ClassDef):
            continue
        for b in n.bases:
            if (isinstance(b, ast.Attribute) and b.attr == "Strategy" and isinstance(b.value, ast.Name)
                    and b.value.id in bt_aliases) or (isinstance(b, ast.Name) and (b.id in direct or b.id in names)):
                found.append(n)
                names.add(n.name)
                break
    return found


def _addstrategy_target(tree: ast.Module) -> str | None:
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "addstrategy"
                and n.args and isinstance(n.args[0], ast.Name)):
            return n.args[0].id
    return None


def _calls_from_signals(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "from_signals"
               for n in ast.walk(node))


def _bind_params(fn: ast.FunctionDef) -> list[str] | None:
    """Argument expressions for fn's required parameters, or None when a required parameter is unknown."""
    a = fn.args
    pos = a.posonlyargs + a.args
    n_req = len(pos) - len(a.defaults)
    exprs = []
    for i, p in enumerate(pos):
        nm = p.arg
        if nm in DATA_PARAMS:
            e = "data"
        elif nm.lower() in CLOSE_PARAMS:
            e = 'data["close"]'
        elif nm.lower() in FIELD_PARAMS:
            e = f'data["{FIELD_PARAMS[nm.lower()]}"]'
        elif i >= n_req:
            break                     # optional parameter we do not know: keep its default
        else:
            return None
        exprs.append(e)
    for k, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is None:
            return None
    return exprs


def _candidate_functions(tree: ast.Module) -> list[tuple[ast.FunctionDef, list[str]]]:
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            b = _bind_params(n)
            if b is not None and b:
                out.append((n, b))
    return out


def _pick_function(tree: ast.Module) -> tuple[ast.FunctionDef, list[str]] | None:
    cands = _candidate_functions(tree)
    if not cands:
        return None
    by = {fn.name: (fn, b) for fn, b in cands}
    for name in PREFERRED_NAMES:
        if name in by:
            return by[name]
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    roots = [(fn, b) for fn, b in cands if fn.name not in called]   # helpers are called by the entry point
    return (roots or cands)[-1]


def detect(code: str) -> dict:
    """-> {"format", "entry", "reason"}; format "unknown" when nothing fits (and "signals" on a syntax error, so
    the sandbox reports it)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"format": "signals", "entry": "generate_signals", "reason": "syntax error"}
    if any(isinstance(n, ast.FunctionDef) and n.name == "generate_signals" for n in tree.body):
        return {"format": "signals", "entry": "generate_signals", "reason": "defines generate_signals"}
    imps = _imports(tree)
    roots = set(v.split(":")[0] for v in imps.values())
    if "backtrader" in roots:
        classes = _strategy_classes(tree, imps)
        if classes:
            target = _addstrategy_target(tree)
            cls = next((c for c in classes if c.name == target), classes[-1])
            return {"format": "backtrader", "entry": cls.name,
                    "reason": f"bt.Strategy subclass{'es' if len(classes) > 1 else ''}: "
                              + ", ".join(c.name for c in classes)}
    fn = _pick_function(tree)
    if "vectorbt" in roots:
        if fn is not None:
            return {"format": "vectorbt", "entry": fn[0].name, "reason": "function with vectorbt imported"}
        if _module_vbt(tree):
            return {"format": "vectorbt", "entry": "<module>", "reason": "module-level entries/exits"}
    if fn is not None:
        return {"format": "function", "entry": fn[0].name, "reason": "strategy-like function"}
    if _module_vbt(tree):
        return {"format": "vectorbt", "entry": "<module>", "reason": "module-level entries/exits"}
    return {"format": "unknown", "entry": "", "reason": "no entry point found"}


def _module_vbt(tree: ast.Module) -> bool:
    top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    assigned = {t.id for n in top if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    return any(_calls_from_signals(n) for n in top) or {"entries", "exits"} <= assigned


# ---------------------------------------------------------------------------------------------- wrapping
def _wrapper(body_lines: list[str]) -> str:
    return "\n\n" + WRAPPER_MARK + "\nimport bta_adapter\n\n\ndef generate_signals(data):\n" + \
        "".join(f"    {ln}\n" for ln in ["bta_adapter.begin(data)", *body_lines])


def _module_wrap(code: str, tree: ast.Module) -> tuple[str, int]:
    """Turn module-level script code into a function body. Imports, defs and classes before the first other
    statement stay at module level; the rest is indented into ``bta_module_strategy(data, close, ...)``.
    Returns (new code, line offset)."""
    lines = code.splitlines()
    first = None
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) \
                and n is tree.body[0]:
            continue   # module docstring
        first = n
        break
    if first is None:
        raise AdapterError("no module-level statements to wrap")
    start = first.lineno - 1      # 0-based index of the first wrapped line
    # lines inside multi-line strings must not be re-indented
    in_string = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                in_string.update(range(tok.start[0], tok.end[0]))   # 1-based rows after the first
    except (tokenize.TokenError, IndentationError):
        pass
    header = "def bta_module_strategy(" + ", ".join(f"{p}=None" for p in MODULE_PARAMS) + "):"
    body = []
    for i in range(start, len(lines)):
        ln = lines[i]
        body.append(ln if (i in in_string or not ln.strip()) else "    " + ln)
    assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Name)}
    ret = "return None"   # the stub captured the from_signals call
    fs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
          and n.func.attr == "from_signals"]
    if fs:   # hand the entries/exits back explicitly too, so the leak localizer can follow them
        call = max(fs, key=lambda n: n.lineno)
        args = {k: v for k, v in zip(("close", "entries", "exits"), call.args)}
        args.update({k.arg: k.value for k in call.keywords if k.arg})
        parts = [ast.get_source_segment(code, args[k]) for k in ("entries", "exits") if k in args]
        extra = [f"{k}={ast.get_source_segment(code, args[k])}" for k in ("short_entries", "short_exits", "direction")
                 if k in args]
        if len(parts) == 2 and all(p and "\n" not in p for p in parts + extra):
            ret = "return bta_adapter.vbt_module_result(" + ", ".join(parts + extra) + ")"
    elif {"entries", "exits"} <= assigned:
        ret = "return bta_adapter.vbt_module_result(entries, exits)"
    body.append("    " + ret)
    offset = 0
    prev = lines[start - 1].strip() if start > 0 else None
    if prev is not None and (prev == "" or prev.startswith("#")) and start - 1 >= 0 and \
            (start - 1) not in in_string and (start) not in in_string:
        head = lines[:start - 1] + [header + ("  " + prev if prev else "")]
    else:
        head = lines[:start] + [header]
        offset = 1
    return "\n".join(head + body) + "\n", offset


def adapt(code: str, fmt: str = "auto") -> Adapted:
    """Detect (or force, with ``fmt``) the strategy format and return code with a ``generate_signals`` wrapper."""
    if fmt not in FORMATS:
        raise AdapterError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
    det = detect(code)
    if fmt == "auto":
        fmt = det["format"]
    if fmt == "signals":
        return Adapted(code, "signals", "generate_signals", CAPTIONS["signals"], original=code)
    if det["format"] == "unknown" or (fmt != det["format"] and not (fmt == "vectorbt" and det["format"] == "function")
                                      and not (fmt == "function" and det["format"] == "vectorbt")):
        raise AdapterError(_no_format_message(fmt, det))
    tree = ast.parse(code)
    base = code if code.endswith("\n") else code + "\n"
    if fmt == "backtrader":
        cls = det["entry"]
        out = Adapted(base.rstrip("\n") + "\n" + _wrapper([f"return bta_adapter.run_backtrader({cls}, data)"]),
                      "backtrader", cls, CAPTIONS["backtrader"].format(entry=cls), original=code, pbo=False)
        out.notes = ["Replayed bar by bar once per ticker (each on its own trading days); line[1] reads the next "
                     "bar, exactly like backtrader with preloaded data.",
                     "Orders set a target exposure per ticker (buy/sell = ±size units, default 1; "
                     "order_target_percent/size = target); weights = exposure / max(1, Σ|exposure|) per date, "
                     "executed by the auditor's engine at the next bar.",
                     "Not modelled (fail with a clear error): broker cash/value, sizers, analyzers, Limit/Stop "
                     "orders, brackets, multiple data feeds per strategy, custom Indicator classes. "
                     "Cerebro setup is ignored; addstrategy(**params) is honoured.",
                     "PBO is skipped for backtrader strategies (each variant is a full Python bar loop)."]
        return out
    fn = _pick_function(tree) if det["entry"] != "<module>" else None
    if fn is not None:
        f, args = fn
        call = f"{f.name}({', '.join(args)})"
        label = "function" if fmt == "function" else "vectorbt"
        entry = f.name if label == "function" else f"{f.name}()"
        out = Adapted(base.rstrip("\n") + "\n" + _wrapper([f"return bta_adapter.to_weights({call}, data)"]),
                      label, f.name, CAPTIONS[label].format(entry=entry), original=code)
        out.notes = [f"Called {call}: parameters bound by name (data = OHLCV dict; close/prices/df = close panel).",
                     "Return value converted: weights as-is; bool or +1/0/-1 signals -> equal weight per date "
                     "(w = s / Σ|s|); (entries, exits) or vbt.Portfolio -> long position state machine -> equal "
                     "weight."]
        return out
    if fmt == "function":
        raise AdapterError(_no_format_message(fmt, det))
    wrapped, offset = _module_wrap(base, tree)
    call = "bta_module_strategy(" + ", ".join(
        "data" if p == "data" else f'data["{ "close" if p in ("price", "prices", "df") else p.rstrip("_")}"]'
        for p in MODULE_PARAMS) + ")"
    out = Adapted(wrapped.rstrip("\n") + "\n" + _wrapper([f"return bta_adapter.to_weights({call}, data)"]),
                  "vectorbt", "<module>", CAPTIONS["vectorbt"].format(entry="script (module-level entries/exits)"),
                  original=code, line_offset=offset, localizable=offset == 0)
    out.notes = ["Module-level script wrapped into a function; `close`, `prices`, `df`, `open_`, `high`, `low`, `volume`, `data` are the audit panel, "
                 "and vbt.YFData.download(...).get('Close') returns it too (no network).",
                 "Entries/exits from the last vbt.Portfolio.from_signals call (or the `entries`/`exits` variables) "
                 "-> long position state machine -> equal weight across names in position."]
    if offset:
        out.notes.append("Line numbers in findings are shifted by one; leak localization and Quick fix are off "
                         "for this script.")
    return out


def _no_format_message(fmt: str, det: dict) -> str:
    want = "" if fmt in ("auto", det["format"]) else f" as '{fmt}'"
    return (f"Could not adapt this code{want} (detected: {det['format']}, {det['reason']}). Accepted formats: "
            "generate_signals(data) -> weights; a function such as strategy(close) / signals(df) / "
            "compute_weights(prices) returning weights, bool or ±1 signals, or (entries, exits); vectorbt code "
            "calling vbt.Portfolio.from_signals(close, entries, exits); a backtrader bt.Strategy subclass.")


def adapt_or_none(code: str, fmt: str = "auto") -> Adapted | None:
    try:
        return adapt(code, fmt)
    except AdapterError:
        return None

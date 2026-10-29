"""Extract an auditable strategy module from a Jupyter notebook (or a .py research script).

Real quant research lives in notebooks: data loading, plots, exploration and the actual strategy
are mixed across cells. The auditor needs one self-contained module that defines
``generate_signals(data) -> weights``. :func:`extract_strategy` builds it:

1. Read the code cells (``.ipynb`` is parsed with :mod:`json`; no nbformat dependency). A ``.py`` file
   is split on ``# %%`` cell markers when it has them, otherwise treated as a single cell.
2. Neutralise IPython syntax: line magics / shell escapes (``%matplotlib``, ``!pip ...``,
   ``x = !ls``, ``obj?``) become ``pass`` so line numbers inside the cell stay intact; ``%%time``-style
   cell magics are dropped, other cell magics (``%%bash``, ``%%writefile``...) skip the whole cell.
   Cells that still do not parse are skipped. Every skip is recorded.
3. Walk the AST from ``generate_signals``: keep the top-level function/class definitions, constant
   assignments and imports it (transitively) depends on, in notebook order (imports first).
   Exploration cells, plots and guarded data loading are left out.
4. Keep a line map: extracted-module line -> (cell number, line in that cell), so a finding on line 12
   of the extracted code can be reported as "cell 7, line 3".

Cell numbers are 1-based positions among *all* cells of the notebook (markdown included), which is
what VS Code, JupyterLab's table of contents and nbviewer show when you count cells top to bottom.
For a plain ``.py`` without ``# %%`` markers the map holds ``(None, line in the file)``.
"""
from __future__ import annotations

import ast
import builtins
import json
import re
from dataclasses import dataclass

__all__ = ["extract_strategy", "locate", "where", "annotate_tests", "CONTRACT"]

ENTRY = "generate_signals"
CONTRACT = (
    "The auditor needs a function `generate_signals(data)` that takes a dict of wide DataFrames "
    "(data['close'], plus 'open'/'high'/'low'/'volume' when available; index = dates, columns = tickers) "
    "and returns a DataFrame of portfolio weights with the same dates x tickers shape. It may import only "
    "numpy, pandas and math, and must not read files or the network (the auditor supplies the data)."
)
# Cell magics whose body is still plain Python: drop the magic line, keep the cell.
PY_CELL_MAGICS = {"time", "timeit", "capture", "prun", "memit", "skip", "debug"}
_BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "display", "get_ipython"}
_DATA_PARAM_HINTS = {"data", "prices", "price", "close", "closes", "df", "px", "panel", "ohlcv", "bars"}
_ASSIGN_MAGIC = re.compile(r"^\s*[A-Za-z_][\w\s,]*=\s*[!%]")   # x = !ls / y = %who_ls


@dataclass
class _Cell:
    number: int          # 1-based position among all cells (None-like 0 for single-cell .py)
    source: str          # python source with magics neutralised
    lines: list[str]


@dataclass
class _Stmt:
    cell: _Cell
    node: ast.stmt
    order: tuple[int, int]
    defines: set[str]


# ----------------------------------------------------------------------------
# Reading cells
# ----------------------------------------------------------------------------
def _to_text(source: str | bytes) -> str:
    if isinstance(source, bytes):
        source = source.decode("utf-8-sig")
    return source.lstrip("\ufeff")


def _read_cells(text: str, filename: str) -> tuple[list[tuple[int | None, str]], int, bool]:
    """Return ([(cell number or None, raw source)], total cell count, is_notebook)."""
    if filename.lower().endswith(".ipynb"):
        try:
            nb = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{filename} is not valid notebook JSON: {e}") from e
        if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
            raise ValueError(f"{filename} has no 'cells' list; is it a Jupyter notebook (nbformat 4)?")
        if nb.get("nbformat", 4) < 4:
            raise ValueError(f"{filename} uses nbformat {nb.get('nbformat')}; re-save it with Jupyter (nbformat 4).")
        out = []
        for i, c in enumerate(nb["cells"], 1):
            if c.get("cell_type") != "code":
                continue
            src = c.get("source", "")
            out.append((i, "".join(src) if isinstance(src, list) else str(src)))
        return out, len(nb["cells"]), True
    # .py: optional percent-format cells ("# %%", "# %% [markdown]", "#%%")
    lines = text.splitlines(keepends=True)
    marks = [i for i, ln in enumerate(lines) if re.match(r"^#\s*%%", ln)]
    if not marks:
        return [(None, text)], 1, False
    out, bounds = [], ([0] + marks if marks[0] != 0 else marks) + [len(lines)]
    for n, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
        chunk = lines[a:b]
        if chunk and re.match(r"^#\s*%%.*\[markdown\]", chunk[0]):
            continue
        out.append((n, "".join(chunk)))
    return out, len(bounds) - 1, False


def _neutralise(raw: str, number: int | None, skipped: list[dict]) -> str | None:
    """Replace IPython-only lines with `pass` (same indentation). None = skip the whole cell."""
    lines = raw.splitlines()
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is not None and lines[first].lstrip().startswith("%%"):
        magic = lines[first].strip()[2:].split()[0] if lines[first].strip()[2:] else ""
        if magic not in PY_CELL_MAGICS:
            skipped.append({"cell": number, "line": first + 1, "kind": "cell_magic", "text": lines[first].strip(),
                            "reason": f"%%{magic} cell is not Python"})
            return None
        skipped.append({"cell": number, "line": first + 1, "kind": "magic", "text": lines[first].strip(),
                        "reason": "cell magic line removed"})
        lines[first] = ""
    cont = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if cont or s.startswith(("%", "!")) or _ASSIGN_MAGIC.match(ln) or (
                s.endswith("?") and not s.startswith("#") and re.match(r"^[\w.\[\]'\"]+\?{1,2}$", s)):
            indent = ln[: len(ln) - len(ln.lstrip())]
            if not cont:
                skipped.append({"cell": number, "line": i + 1, "kind": "magic", "text": s,
                                "reason": "IPython magic / shell escape removed"})
            lines[i] = indent + "pass" if not cont else ""
            cont = s.endswith("\\")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# AST helpers
# ----------------------------------------------------------------------------
def _target_names(t: ast.AST) -> set[str]:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(e) for e in t.elts)) if t.elts else set()
    if isinstance(t, ast.Starred):
        return _target_names(t.value)
    return set()


def _alias_name(a: ast.alias) -> str:
    return a.asname or a.name.split(".")[0]


def _defines(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {_alias_name(a) for a in node.names if a.name != "*"}
    if isinstance(node, ast.Assign):
        return set().union(*(_target_names(t) for t in node.targets))
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _stored_anywhere(node: ast.AST) -> set[str]:
    """Names bound by a non-extractable top-level statement (if/for/with/try blocks...)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {_alias_name(a) for a in n.names if a.name != "*"}
    return out


def _free_names(node: ast.AST) -> set[str]:
    """Names a statement reads from module scope (approximate: loads minus anything bound inside)."""
    loads, local = set(), set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + (
                [a.kwarg] if a.kwarg else []):
            local.add(arg.arg)
        # decorators and defaults are evaluated in module scope
        outer = list(getattr(node, "decorator_list", [])) + list(a.defaults) + [d for d in a.kw_defaults if d]
        for o in outer:
            loads |= _free_names(o)
        body = node.body if isinstance(node.body, list) else [node.body]
    else:
        body = [node]
    for b in body:
        for n in ast.walk(b):
            if isinstance(n, ast.Name):
                (loads if isinstance(n.ctx, ast.Load) else local).add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node:
                local.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                local |= {_alias_name(a) for a in n.names}
            elif isinstance(n, ast.arg):
                local.add(n.arg)
    return loads - local


def _returns_dataframe_like(fn: ast.FunctionDef) -> bool:
    return any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn))


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def extract_strategy(source: str | bytes, filename: str) -> dict:
    """Build a self-contained strategy module from a notebook or a Python file.

    Args:
        source: file contents (``.ipynb`` JSON or Python source), str or bytes.
        filename: used for the format (``.ipynb`` vs anything else = Python) and in messages.

    Returns a dict (never raises for bad input):
        ``ok`` (bool), ``error`` (str | None: why extraction failed, explaining the contract),
        ``code`` (str: extracted module, "" on error), ``line_map`` ({extracted line: (cell, line in cell)};
        cell is the 1-based notebook cell number, or None for a plain .py where the second item is the file line;
        line 1 is a header comment and is not mapped), ``cell_count`` (all cells in the notebook),
        ``code_cell_count``, ``entry_cell`` (cell holding generate_signals), ``kept`` (names kept, in order),
        ``skipped`` ([{cell, line, kind, text, reason}] for removed magics and unparsable / non-Python cells),
        ``warnings`` ([str]), ``is_notebook`` (bool).
    """
    res = {"ok": False, "error": None, "code": "", "line_map": {}, "cell_count": 0, "code_cell_count": 0,
           "entry_cell": None, "kept": [], "skipped": [], "warnings": [], "is_notebook": False}
    try:
        text = _to_text(source)
        raw_cells, res["cell_count"], res["is_notebook"] = _read_cells(text, filename)
    except (ValueError, UnicodeDecodeError) as e:
        res["error"] = str(e)
        return res
    res["code_cell_count"] = len(raw_cells)

    def where(cell: int | None, line: int) -> str:
        return f"cell {cell}, line {line}" if cell is not None else f"line {line}"

    stmts: list[_Stmt] = []
    opaque: dict[str, _Stmt] = {}        # name -> last non-extractable statement binding it
    for number, raw in raw_cells:
        src = _neutralise(raw, number, res["skipped"])
        if src is None:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            res["skipped"].append({"cell": number, "line": e.lineno or 1, "kind": "syntax_error",
                                   "text": (e.text or "").strip(), "reason": f"cell does not parse: {e.msg}"})
            if number is None:     # a single-cell .py that does not parse cannot be audited at all
                res["error"] = f"{filename} does not parse ({where(None, e.lineno or 1)}): {e.msg}"
                return res
            continue
        cell = _Cell(number or 0, src, src.splitlines())
        for node in tree.body:
            st = _Stmt(cell, node, (number or 0, node.lineno), _defines(node))
            if st.defines:
                stmts.append(st)
            else:
                for n in _stored_anywhere(node):
                    opaque[n] = st

    defs: dict[str, _Stmt] = {}
    multi: dict[str, list[_Stmt]] = {}
    for st in stmts:
        for n in st.defines:
            defs[n] = st                        # notebook semantics: the last execution wins
            multi.setdefault(n, []).append(st)

    if ENTRY not in defs or not isinstance(defs[ENTRY].node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        cands = []
        for st in stmts:
            nd = st.node
            if isinstance(nd, ast.FunctionDef) and nd.args.args and _returns_dataframe_like(nd) and \
                    nd.args.args[0].arg.lower() in _DATA_PARAM_HINTS:
                cands.append(f"`{nd.name}({nd.args.args[0].arg})` ({where(st.cell.number or None, nd.lineno)})")
        msg = f"No `{ENTRY}` function found in {filename}. {CONTRACT}"
        if cands:
            msg += (" Found " + ", ".join(cands) + ", which looks like a signal function: rename it, or add "
                    f"`def {ENTRY}(data):\n    return {cands[0].split('(')[0].strip('`')}(data[\"close\"])` "
                    "(adapt the argument to what it expects).")
        bad = [s for s in res["skipped"] if s["kind"] == "syntax_error"]
        if bad:
            msg += " Note: " + "; ".join(f"{where(s['cell'], s['line'])} was skipped ({s['reason']})" for s in bad) + "."
        res["error"] = msg
        return res

    # transitive dependency walk
    for n, lst in multi.items():
        if len(lst) > 1:
            cells = ", ".join(where(s.cell.number or None, s.node.lineno) for s in lst)
            res["warnings"].append(f"`{n}` is defined {len(lst)} times ({cells}); the last definition is used, "
                                   "as when the notebook runs top to bottom.")
    keep: dict[int, _Stmt] = {}
    needed_aliases: dict[int, set[str]] = {}
    queue, seen = [ENTRY], set()
    missing = []
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        st = defs.get(name)
        if st is None:
            if name in _BUILTINS:
                continue
            if name in opaque:
                o = opaque[name]
                res["warnings"].append(
                    f"`{name}` is only set inside a conditional/loop block ({where(o.cell.number or None, o.node.lineno)}) "
                    f"and was not extracted; {ENTRY} should compute it from `data`.")
            else:
                missing.append(name)
            continue
        keep[id(st)] = st
        if isinstance(st.node, (ast.Import, ast.ImportFrom)):
            needed_aliases.setdefault(id(st), set()).add(name)
            continue
        queue.extend(sorted(_free_names(st.node)))
    for name in sorted(set(missing)):
        res["warnings"].append(f"`{name}` is used by the strategy but never defined in {filename}; "
                               "the audit will fail with a NameError.")
    if res["skipped"] and any(s["kind"] == "syntax_error" for s in res["skipped"]):
        res["warnings"].append("Some cells did not parse and were skipped: " + "; ".join(
            where(s["cell"], s["line"]) for s in res["skipped"] if s["kind"] == "syntax_error") + ".")

    kept = sorted(keep.values(), key=lambda s: (not isinstance(s.node, (ast.Import, ast.ImportFrom)), s.order))
    origin = "notebook" if res["is_notebook"] else "file"
    out_lines = [f"# Strategy extracted from {origin} {filename} by Backtest Auditor (cells: see line map)"]
    line_map: dict[int, tuple[int | None, int]] = {}
    prev: _Stmt | None = None
    blocky = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    for st in kept:
        nd, cell = st.node, st.cell
        is_imp = isinstance(nd, (ast.Import, ast.ImportFrom))
        if not is_imp and prev is not None and (isinstance(nd, blocky) or isinstance(prev.node, blocky)
                                                 or prev.cell is not cell):
            out_lines += ["", ""]
        prev = st
        cnum = cell.number or None
        if is_imp:
            wanted = needed_aliases.get(id(st), set())
            aliases = [a for a in nd.names if _alias_name(a) in wanted]
            if len(aliases) == len(nd.names) and nd.lineno == nd.end_lineno:
                text_lines = [cell.lines[nd.lineno - 1].strip()]
            else:
                text_lines = [ast.unparse(type(nd)(names=aliases, **({"module": nd.module, "level": nd.level}
                                                                      if isinstance(nd, ast.ImportFrom) else {})))]
            for t in text_lines:
                out_lines.append(t)
                line_map[len(out_lines)] = (cnum, nd.lineno)
            continue
        start = min([nd.lineno] + [d.lineno for d in getattr(nd, "decorator_list", [])])
        # pull in comment lines directly above the definition (same cell)
        while start > 1 and cell.lines[start - 2].strip().startswith("#"):
            start -= 1
        for ln in range(start, nd.end_lineno + 1):
            out_lines.append(cell.lines[ln - 1].rstrip())
            line_map[len(out_lines)] = (cnum, ln)
    code = "\n".join(out_lines).rstrip() + "\n"
    try:
        ast.parse(code)
    except SyntaxError as e:   # should not happen; report rather than hand back broken code
        res["error"] = f"Internal error: extracted module does not parse (line {e.lineno}: {e.msg})."
        return res
    entry = defs[ENTRY]
    res.update(ok=True, code=code, line_map=line_map, entry_cell=entry.cell.number or None,
               kept=[n for st in kept for n in sorted(st.defines & seen)])
    return res


def locate(result: dict, line: int) -> tuple[int | None, int] | None:
    """(cell, line in cell) for a line of ``result['code']``, or None when unmapped (e.g. the header)."""
    m = result.get("line_map") or {}
    v = m.get(line, m.get(str(line)))
    return tuple(v) if v is not None else None


def where(result_or_map: dict, line: int) -> str:
    """'cell N, line M' for a line of the extracted module, '' when unmapped (or a plain .py without cells)."""
    m = result_or_map.get("line_map", result_or_map) if isinstance(result_or_map, dict) else {}
    v = (m or {}).get(int(line or 0), (m or {}).get(str(int(line or 0))))
    return f"cell {v[0]}, line {v[1]}" if v and v[0] is not None else ""


def annotate_tests(tests: list[dict], line_map: dict) -> list[dict]:
    """Add ``cell`` / ``cell_line`` / ``where`` to the line references in audit test dicts (in place).

    Covers the dynamic localizer's entry (future_perturbation evidence), its trace rows, and the static scan
    findings, so the UI and the downloadable report can say "cell 7, line 3" for an uploaded notebook.
    Line numbers themselves stay lines of the extracted module."""
    lm = {"line_map": line_map or {}}

    def tag(d: dict | None):
        if isinstance(d, dict) and d.get("line"):
            w = where(lm, d["line"])
            if w:
                c = locate(lm, int(d["line"]))
                d["cell"], d["cell_line"], d["where"] = c[0], c[1], w
    for t in tests or []:
        ev = t.get("evidence") or {}
        lz = ev.get("localized") or {}
        tag(lz.get("entry"))
        for row in (lz.get("chain") or []) + (lz.get("per_line") or []):
            tag(row)
        for f in ev.get("findings") or []:
            tag(f)
    return tests

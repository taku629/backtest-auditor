#!/usr/bin/env python3
"""Backtest Auditor for CI: deterministic look-ahead / survivorship audit of strategy files.

    python audit_cli.py strategies/*.py [--data synthetic|csv:PATH] [--format text|markdown|json|sarif]
                        [--fail-on critical|high|warning] [--out FILE] [--sarif-out FILE] [--fix [--write]]

Each file defines ``generate_signals(data) -> weights``, or is in another format that ``audit/adapters.py``
adapts automatically (``--strategy-format auto|signals|function|vectorbt|backtrader``): a function such as
``strategy(close)`` returning weights / bool / ±1 signals / (entries, exits), vectorbt code calling
``vbt.Portfolio.from_signals``, or a backtrader ``bt.Strategy`` subclass. Jupyter notebooks (``.ipynb``) are accepted:
``notebook.py`` extracts ``generate_signals`` plus the helpers, constants and imports it uses, and findings
are reported as "cell N, line M". The strategy is executed in the existing sandbox
(``sandbox/runner.py``) and checked by the existing deterministic battery (``audit/leak_tests.py``).
No LLM and no API key are needed for the audit. ``--fix`` first tries the deterministic Quick fix
(``audit/quickfix.py``: candidate patches of the localized leak line, each re-audited; no LLM, $0) and prints the
unified diff; ``--write`` saves it as ``<name>.fixed.py``. When no candidate passes, it falls back to the Nemotron agent
loop (``agents/loop.py``), which writes ``<name>.fixed.py``, if ``NEBIUS_API_KEY`` is set (or ``--mock-llm``).

Exit codes: 0 = no failing test at or above --fail-on; 1 = at least one; 2 = usage error (no files, bad data).
When ``GITHUB_ACTIONS=true`` it also prints ``::error file=..,line=..::`` workflow annotations and appends a
Markdown summary to ``$GITHUB_STEP_SUMMARY``.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

from audit.ast_scan import scan_code  # noqa: E402
from audit.leak_tests import SEVERITY_ORDER, run_audit  # noqa: E402
from sandbox.runner import SandboxError, SandboxExecutor, check_policy  # noqa: E402
from notebook import extract_strategy  # noqa: E402

TOOL_NAME = "Backtest Auditor"
TOOL_VERSION = "0.2.0"
INFO_URI = "https://github.com/"  # replaced by the repo URL when known (GITHUB_SERVER_URL/GITHUB_REPOSITORY)

# --fail-on threshold -> set of severities that make the run fail.
FAIL_ON = {
    "critical": {"critical"},
    "high": {"critical", "high"},
    "warning": {"critical", "high", "medium", "low"},
}
SARIF_LEVEL = {"critical": "error", "high": "error", "medium": "warning", "low": "note", "info": "note"}
GH_LEVEL = {"critical": "error", "high": "error", "medium": "warning", "low": "notice", "info": "notice"}

# Which static-scan rules are plausible root causes of each dynamic failure (used to point at source lines).
LOOKAHEAD_RULES = {"negative_shift", "negative_pct_change", "negative_diff", "centered_window", "backward_fill",
                   "interpolation", "full_sample_statistic", "survivor_universe"}
try:  # the newer pandas-trap hint rules (time_axis_rank, resample_start_label, ...)
    from audit.ast_scan import LOOKAHEAD_RULE_IDS as _EXTRA_RULES
    LOOKAHEAD_RULES = set(LOOKAHEAD_RULES) | set(_EXTRA_RULES)
except ImportError:
    pass
SUSPECT_RULES = {
    "future_perturbation": LOOKAHEAD_RULES,
    "survivorship": {"survivor_universe"},
    "execution_delay": {"negative_shift", "negative_pct_change", "negative_diff", "centered_window", "backward_fill"},
}

TEST_HELP = {
    "future_perturbation": "Scrambles all data after random cut dates; signals on or before each cut must not change.",
    "survivorship": "No weight before listing; past signals must not change when delisted names are revived.",
    "execution_delay": "Sharpe at t+1 vs t+2; a collapse points to same-bar or look-ahead execution.",
    "static_scan": "AST scan for negative shifts, centred windows, back-fills, survivor universes, full-sample stats.",
    "plausibility": "Sharpe above 3 after costs on daily equities is treated as a bug.",
    "deflated_sharpe": "Bailey & Lopez de Prado deflated Sharpe ratio (selection bias).",
    "train_test_split": "70/30 in-sample vs out-of-sample Sharpe degradation.",
    "non_trivial": "Strategy must take at least one position.",
    "pbo": "Probability of Backtest Overfitting (Bailey et al. 2015, CSCV) over a grid of the strategy's own parameters.",
    "sandbox": "Strategy must pass the sandbox policy and run without error.",
}


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class FileResult:
    path: str                      # as displayed / used in SARIF (relative, posix)
    abspath: str
    tests: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)
    def_line: int = 1
    error: str = ""
    seconds: float = 0.0
    fixed_path: str = ""
    fix_status: str = ""
    quick_fix: dict = field(default_factory=dict)   # audit/quickfix.py result (JSON-safe)
    raw: dict = field(default_factory=dict, repr=False)   # run_audit().to_dict() of the audited code
    code: str = ""                 # audited source (for .ipynb: the module extracted by notebook.py)
    notebook: dict = field(default_factory=dict)   # .ipynb only: cell_count, skipped, warnings, line_map
    equity: dict = field(default_factory=dict, repr=False)       # {"dates", "values"} for --report
    fix_equity: dict = field(default_factory=dict, repr=False)   # equity of the accepted Quick fix
    fix_metrics: dict = field(default_factory=dict, repr=False)  # full metrics of the accepted Quick fix
    adapter: dict = field(default_factory=dict)    # audit/adapters.py: format, entry, caption, notes
    adapted: object = field(default=None, repr=False)   # the Adapted object (to strip the wrapper from fixes)

    def cell_of(self, line: int) -> tuple[int, int] | None:
        """(cell, line in cell) for a line of the audited code when it came from a notebook."""
        v = (self.notebook.get("line_map") or {}).get(int(line or 0))
        return (v[0], v[1]) if v and v[0] is not None else None

    def where(self, line: int, sep: str = ", ") -> str:
        """'cell N, line M' for notebooks, '' for plain files."""
        c = self.cell_of(line)
        return f"cell {c[0]}{sep}line {c[1]}" if c else ""

    def failing(self) -> list[dict]:
        return [t for t in self.tests if not t["passed"]]

    def fails_threshold(self, fail_on: str) -> list[dict]:
        sev = FAIL_ON[fail_on]
        return [t for t in self.failing() if t["severity"] in sev]

    def to_dict(self, fail_on: str) -> dict:
        return {"path": self.path, "passed": not self.fails_threshold(fail_on), "error": self.error or None,
                "def_line": self.def_line, "metrics": self.metrics, "tests": self.tests,
                "static_findings": self.findings, "seconds": round(self.seconds, 3),
                **({"adapter": self.adapter} if self.adapter else {}),
                **({"fixed_path": self.fixed_path, "fix_status": self.fix_status} if self.fix_status else {}),
                **({"quick_fix": {k: v for k, v in self.quick_fix.items() if k != "code"}} if self.quick_fix else {}),
                **({"notebook": {**self.notebook, "extracted_code": self.code}} if self.notebook else {})}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def expand_paths(patterns: list[str]) -> tuple[list[Path], list[str]]:
    files, missing, seen = [], [], set()
    for pat in patterns:
        if any(c in pat for c in "*?["):   # glob: skip directories and our own *.fixed.py outputs
            hits = [h for h in sorted(glob.glob(pat, recursive=True))
                    if Path(h).is_file() and not h.endswith(".fixed.py")
                    and ".ipynb_checkpoints" not in Path(h).parts]
        else:
            hits = [pat]
        if not hits:
            missing.append(pat)
        for h in hits:
            p = Path(h)
            if not p.is_file():
                missing.append(h)
                continue
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                files.append(p)
    return files, missing


def display_path(p: Path) -> str:
    base = Path(os.environ.get("GITHUB_WORKSPACE") or os.getcwd()).resolve()
    rp = p.resolve()
    try:
        return rp.relative_to(base).as_posix()
    except ValueError:
        return rp.as_posix()


def load_data(spec: str) -> dict:
    if spec == "synthetic":
        from data import generate_panel
        return generate_panel()
    if spec.startswith("csv:"):
        from data import load_long_csv
        return load_long_csv(spec[4:])
    raise ValueError(f"--data must be 'synthetic' or 'csv:PATH', got {spec!r}")


def def_line_of(code: str) -> int:
    try:
        for n in ast.parse(code).body:
            if isinstance(n, ast.FunctionDef) and n.name == "generate_signals":
                return n.lineno
    except SyntaxError as e:
        return e.lineno or 1
    return 1


def suspect_lines(test: dict, findings: list[dict]) -> list[dict]:
    if test["name"] == "static_scan":
        return [f for f in findings]
    rules = SUSPECT_RULES.get(test["name"], set())
    return [f for f in findings if f["rule"] in rules]


def first_leak(test: dict) -> str | None:
    ev = test.get("evidence") or {}
    return ev.get("first_leaking_date") or ev.get("counterfactual_first_change")


def key_evidence(test: dict) -> list[str]:
    ev, n = test.get("evidence") or {}, test["name"]
    out = []
    if n == "future_perturbation":
        lz = ev.get("localized") or {}
        if lz.get("found") and lz.get("entry"):
            how = "static, adapted code" if str(lz.get("method", "")).startswith("static") else "dynamic"
            out.append(f"leak enters at {lz['entry'].get('where') or 'line ' + str(lz['entry']['line'])} "
                       f"({how}): `{lz['entry']['source_line']}`")
        out.append(f"leaking cut dates: {ev.get('n_leaking_cuts')}/{ev.get('n_cuts')}; "
                   f"max |weight change| {ev.get('max_abs_weight_change')}")
        for c in (ev.get("cuts") or [])[:3]:
            if c.get("leaked"):
                out.append(f"cut {c['cut_date']}: {c['rows_changed']} earlier rows changed "
                           f"({c.get('earliest_changed_date')} .. {c.get('latest_changed_date')}), "
                           f"e.g. {', '.join(c.get('example_tickers', [])[:3])}")
    elif n == "survivorship":
        for k in ("weight_before_listing", "counterfactual_revived", "counterfactual_rows_changed",
                  "counterfactual_first_change", "avg_weight_on_later_delisted", "data_warning"):
            if k in ev:
                out.append(f"{k}: {ev[k]}")
    elif n == "execution_delay":
        out.append(f"Sharpe t+1 {ev.get('sharpe_lag1')} vs t+2 {ev.get('sharpe_lag2')} "
                   f"(drop {ev.get('relative_drop')})")
    elif n in ("plausibility",):
        out.append(f"Sharpe {ev.get('sharpe')}, CAGR {ev.get('cagr')}")
    elif n == "deflated_sharpe":
        out.append(f"DSR {ev.get('dsr')} (threshold {ev.get('threshold')})")
    elif n == "pbo" and ev.get("status") == "ok":
        out.append(f"PBO {ev.get('pbo'):.2f} ({ev.get('risk')} risk, fails above {ev.get('threshold')}); "
                   f"{ev.get('n_variants')} variants x {ev.get('n_splits')} splits; degradation slope "
                   f"{ev.get('degradation_slope'):.2f}; P(OOS loss) {ev.get('prob_oos_loss'):.0%}")
    elif n == "train_test_split":
        out.append(f"Sharpe train {ev.get('sharpe_train')} vs test {ev.get('sharpe_test')}")
    elif n == "sandbox":
        out.append(str(ev.get("error", ""))[:400])
    return out


# ----------------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------------
def parse_param_grid(spec: str) -> dict[str, list[float]] | None:
    """'LOOKBACK=10,20,40;TOP=0.8,0.9' -> {'LOOKBACK': [10, 20, 40], 'TOP': [0.8, 0.9]}"""
    if not spec:
        return None
    out = {}
    for part in spec.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = [float(x) for x in v.split(",") if x.strip()]
    return out or None


def audit_file(path: Path, data: dict, *, seed: int = 7, n_cuts: int = 6, cost_bps: float = 10.0,
               timeout: float = 90.0, pbo: bool = True, param_grid: dict | None = None,
               strategy_format: str = "auto") -> FileResult:
    t0 = time.time()
    nb: dict = {}
    if path.suffix.lower() == ".ipynb":
        nb = extract_strategy(path.read_bytes(), display_path(path))
        if not nb["ok"]:
            return _notebook_error(path, nb, time.time() - t0)
        code = nb["code"]
    else:
        code = path.read_text(encoding="utf-8")
    from audit.adapters import AdapterError, adapt
    adapted = None
    try:
        adapted = adapt(code, strategy_format)
    except AdapterError as e:
        if strategy_format != "auto":
            res = FileResult(display_path(path), str(path.resolve()), code=code, error=str(e))
            res.tests = [{"name": "sandbox", "passed": False, "severity": "critical", "blocking": True,
                          "summary": "The strategy format could not be adapted, so the dynamic tests did not run.",
                          "evidence": {"error": str(e)}, "lines": [], "suggested_fix": str(e)}]
            res.seconds = time.time() - t0
            return res
    if adapted is not None and adapted.adapted:
        code = adapted.code
    res = FileResult(display_path(path), str(path.resolve()), def_line=def_line_of(code), code=code)
    if adapted is not None and adapted.adapted:
        res.adapter, res.adapted = adapted.to_dict(), adapted
        res.def_line = 1
        pbo = pbo and adapted.pbo
    if nb:
        res.notebook = {k: nb[k] for k in ("line_map", "cell_count", "code_cell_count", "entry_cell", "kept",
                                           "skipped", "warnings")}
    res.findings = [f.to_dict() for f in scan_code(code)]
    violations = check_policy(code)
    try:
        if violations:
            raise SandboxError("Sandbox policy violation:\n" + "\n".join(violations))
        rep = run_audit(data, SandboxExecutor(code, timeout), code, n_trials=1, cost_bps=cost_bps,
                        n_cuts=n_cuts, seed=seed, pbo=pbo, pbo_timeout=timeout, param_grid=param_grid,
                        localize=(adapted is None or adapted.localizable))
        rd = rep.to_dict()
        res.tests, res.metrics = rd["tests"], rd["metrics"]
        res.raw = rd
        res.equity = _equity_dict(rep.equity)
    except (SandboxError, RuntimeError, ValueError) as e:
        res.error = str(e)
        from audit.leak_tests import evaluate_static
        static = evaluate_static(code).to_dict()
        res.tests = [{"name": "sandbox", "passed": False, "severity": "critical", "blocking": True,
                      "summary": "Strategy could not be executed, so the dynamic tests did not run.",
                      "evidence": {"error": str(e)[:2000]},
                      "suggested_fix": "Define generate_signals(data) returning a dates x tickers DataFrame, "
                                       "importing only numpy/pandas/math, with no I/O."}, static]
    for t in res.tests:
        t["lines"] = [{"line": f["line"], "rule": f["rule"], "snippet": f["snippet"], "severity": f["severity"],
                       "message": f["message"], "fix": f["fix"]}
                      for f in suspect_lines(t, res.findings)] if not t["passed"] else []
        if t["name"] == "sandbox":
            t["lines"] = [{"line": int(v.split()[1].rstrip(":")), "rule": "sandbox_policy", "snippet": v}
                          for v in violations if v.startswith("line ")]
        lz = (t.get("evidence") or {}).get("localized") or {}
        if t["name"] == "future_perturbation" and not t["passed"] and lz.get("found") and lz.get("entry"):
            e = lz["entry"]  # dynamic localization (audit/localize.py) becomes the primary line
            same = [x["rule"] for x in t["lines"] if x["line"] == e["line"]]
            t["lines"] = [{"line": e["line"], "rule": ", ".join(["dynamic_leak_entry", *same]),
                           "snippet": e["source_line"], "severity": t["severity"],
                           "message": f"leak enters at line {e['line']} "
                                      f"({'static' if str(lz.get('method', '')).startswith('static') else 'dynamic'}): "
                                      f"{e['kind']}",
                           "fix": t.get("suggested_fix", "")}] + [x for x in t["lines"] if x["line"] != e["line"]]
    res.tests.sort(key=lambda t: (t["passed"], SEVERITY_ORDER.get(t["severity"], 9)))
    _annotate_cells(res)
    res.seconds = time.time() - t0
    return res


def _equity_dict(eq) -> dict:
    return {"dates": [d.strftime("%Y-%m-%d") for d in eq.index], "values": [round(float(x), 6) for x in eq.to_numpy()]}


def _annotate_cells(res: FileResult) -> None:
    """Notebooks: add cell / cell_line to every line reference (lines stay lines of the extracted code)."""
    if not res.notebook:
        return
    for item in res.findings + [ln for t in res.tests for ln in t.get("lines", [])]:
        c = res.cell_of(item.get("line", 0))
        if c:
            item["cell"], item["cell_line"] = c
            if str(item.get("message", "")).startswith(f"leak enters at line {item['line']} "):
                item["message"] = item["message"].replace(f"at line {item['line']} ", f"at {res.where(item['line'])} ", 1)
    for t in res.tests:   # dynamic localization entry (audit/localize.py) -> cell reference
        e = ((t.get("evidence") or {}).get("localized") or {}).get("entry")
        if isinstance(e, dict) and res.cell_of(e.get("line", 0)):
            e["cell"], e["cell_line"] = res.cell_of(e["line"])
            e["where"] = res.where(e["line"])


def _notebook_error(path: Path, nb: dict, seconds: float) -> FileResult:
    """A notebook we could not turn into a strategy module fails like a sandbox error, with the contract."""
    res = FileResult(display_path(path), str(path.resolve()), error=nb["error"], seconds=seconds,
                     notebook={k: nb.get(k) for k in ("cell_count", "code_cell_count", "skipped", "warnings")})
    res.notebook["line_map"] = {}
    res.tests = [{"name": "notebook", "passed": False, "severity": "critical", "blocking": True,
                  "summary": "No auditable strategy could be extracted from the notebook.",
                  "evidence": {"error": nb["error"], "skipped": nb.get("skipped", [])}, "lines": [],
                  "suggested_fix": "Define generate_signals(data) in a code cell; it may use helper functions, "
                                   "constants and numpy/pandas imports from other cells."}]
    return res


def strategy_hypothesis(code: str, path: Path) -> str:
    try:
        for n in ast.parse(code).body:
            if isinstance(n, ast.FunctionDef) and n.name == "generate_signals":
                doc = ast.get_docstring(n)
                if doc:
                    return doc.strip().splitlines()[0]
        doc = ast.get_docstring(ast.parse(code))
        if doc:
            return doc.strip().splitlines()[0]
    except SyntaxError:
        pass
    return f"Keep the intent of the strategy in {path.name}, but make it free of look-ahead and survivorship bias."


def propose_fix(path: Path, res: FileResult, data: dict, *, mock: bool, max_iters: int, cost_bps: float,
                quick: bool = True, write: bool = False, llm: bool = True, seed: int = 7, n_cuts: int = 6,
                timeout: float = 90.0) -> None:
    """Quick fix first (deterministic, no LLM); then, if allowed, agents/loop.py with Nemotron."""
    if not res.failing() or not any(t.get("blocking") for t in res.tests):
        res.fix_status = "not_needed"
        return
    if quick and res.adapted is not None and not res.adapted.localizable:
        quick = False   # module-level script wrapped with shifted lines: patches would not map back
    if quick and res.raw and (res.code or not res.notebook):
        from audit.quickfix import public, quick_fix
        code = res.code or path.read_text(encoding="utf-8")
        qf_full = quick_fix(code, data, report=res.raw, seed=seed, n_cuts=n_cuts, cost_bps=cost_bps,
                            timeout=timeout, name=path.name if not res.notebook else path.stem + ".py")
        if qf_full.get("report") is not None:
            res.fix_equity = _equity_dict(qf_full["report"].equity)
            res.fix_metrics = dict(qf_full["report"].metrics)
        qf = public(qf_full)
        res.quick_fix = qf
        if qf.get("status") == "fixed":
            res.fix_status = "quick_fix"
            if write:
                out = path.with_name(path.stem + ".fixed.py")
                fixed = res.adapted.strip(qf["code"]) if res.adapted is not None else qf["code"]
                out.write_text(fixed, encoding="utf-8")
                res.fixed_path = display_path(out)
            return
    if not llm:
        res.fix_status = "quick_fix_failed"
        return
    if not mock and not os.environ.get("NEBIUS_API_KEY"):
        res.fix_status = "skipped_no_api_key"
        return
    if res.notebook and not res.code:
        res.fix_status = "no_strategy_extracted"
        return
    from agents.llm import make_llm
    from agents.loop import run_audit_loop
    code = res.code or path.read_text(encoding="utf-8")   # notebooks: the extracted module -> <name>.fixed.py
    trace = run_audit_loop(strategy_hypothesis(code, path), data, make_llm(mock=mock), max_iters=max_iters,
                           cost_bps=cost_bps, initial_code=code, summarize=False, quick_fix=False)
    last = trace.iterations[-1] if trace.iterations else None
    if last is None or last.user_code:
        res.fix_status = "no_candidate"
        return
    out = path.with_name(path.stem + ".fixed.py")
    header = (f"# Proposed by Backtest Auditor --fix ({'mock LLM' if mock else 'Nemotron via Nebius Token Factory'}).\n"
              f"# Deterministic audit status of this version: {trace.final_status}. Review before merging.\n")
    out.write_text(header + last.code, encoding="utf-8")
    res.fixed_path, res.fix_status = display_path(out), trace.final_status


# ----------------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------------
def _status(t: dict) -> str:
    return "PASS" if t["passed"] else "FAIL"


def _ref(r: FileResult, line: int) -> str:
    """'path:LINE' for .py files, 'path (cell N, line M)' for notebooks."""
    w = r.where(line)
    return f"{r.path} ({w})" if w else f"{r.path}:{line}"


def _notebook_text(r: FileResult) -> list[str]:
    nb = r.notebook
    if not nb:
        return []
    out = []
    if r.code:
        out.append(f"  notebook: {nb.get('cell_count')} cells ({nb.get('code_cell_count')} code); "
                   f"generate_signals in cell {nb.get('entry_cell')}; audited "
                   f"{', '.join(nb.get('kept') or [])}")
    else:
        out.append(f"  notebook error: {r.error}")
    skipped = nb.get("skipped") or []
    if skipped:
        out.append("  skipped: " + "; ".join(f"cell {s['cell']} line {s['line']} {s['text'][:40]!r} ({s['kind']})"
                                            for s in skipped[:6]) + (" ..." if len(skipped) > 6 else ""))
    for w in nb.get("warnings") or []:
        out.append(f"  warning: {w}")
    return out


def _quick_fix_text(r: FileResult) -> list[str]:
    qf = r.quick_fix
    if not qf:
        return []
    if qf.get("status") == "fixed":
        b, a = qf["before"], qf["after"]
        out = [f"  ⚡ Quick fix (no LLM, $0): {qf['rule']} at {', '.join(_ref(r, ln) for ln in qf['lines'])}: "
               f"{qf['description']}",
               f"     Sharpe {b['sharpe']:.2f} -> {a['sharpe']:.2f}; patched code passes every blocking test "
               f"(gross exposure {a['avg_gross']:.2f}, not cash).", ""]
        out += ["     " + ln for ln in qf["diff"].rstrip("\n").splitlines()]
        return out
    return [f"  ⚡ Quick fix: {qf.get('status')} ({qf.get('n_candidates', 0)} candidate(s) tried, none passed "
            "without going to cash)."]


def render_text(results: list[FileResult], fail_on: str, data_desc: str) -> str:
    out = [f"{TOOL_NAME} {TOOL_VERSION}  |  data: {data_desc}  |  fail-on: {fail_on}", ""]
    for r in results:
        bad = r.fails_threshold(fail_on)
        out.append("=" * 78)
        out.append(f"{r.path}  ->  {'FAIL' if bad else 'PASS'}"
                   + (f"  ({len(bad)} test(s) at or above '{fail_on}')" if bad else ""))
        if r.metrics:
            m = r.metrics
            out.append(f"  backtest: Sharpe {m.get('sharpe', 0):.2f}, CAGR {m.get('cagr', 0):.1%}, "
                       f"max DD {m.get('max_drawdown', 0):.1%}, turnover {m.get('turnover_annual', 0):.1f}x/yr")
        out += _notebook_text(r)
        if r.adapter:
            out.append(f"  {r.adapter['caption']}")
            for n in r.adapter.get("notes", [])[:2]:
                out.append(f"    · {n}")
        if r.error and not (r.notebook and not r.code):
            out.append(f"  sandbox error: {'; '.join(r.error.splitlines())[:300]}")
        out.append("=" * 78)
        for t in r.tests:
            out.append(f"  [{_status(t)}] {t['name']:<20} {t['severity']:<8} {t['summary']}")
            if t["passed"]:
                continue
            fl = first_leak(t)
            if fl:
                out.append(f"         first leaking date: {fl}")
            for e in key_evidence(t):
                out.append(f"         evidence: {e}")
            for ln in t.get("lines", []):
                out.append(f"         {_ref(r, ln['line'])}  [{ln['rule']}]  {ln['snippet']}")
            if t.get("suggested_fix"):
                out.append(f"         fix: {t['suggested_fix']}")
        out += _quick_fix_text(r)
        if r.fix_status:
            out.append(f"  --fix: {r.fix_status}" + (f" -> {r.fixed_path}" if r.fixed_path else ""))
        out.append("")
    n_bad = sum(bool(r.fails_threshold(fail_on)) for r in results)
    out.append(f"Summary: {len(results) - n_bad}/{len(results)} file(s) passed (fail-on: {fail_on}).")
    return "\n".join(out) + "\n"


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_markdown(results: list[FileResult], fail_on: str, data_desc: str) -> str:
    out = [f"## {TOOL_NAME} report", "", f"Data: `{data_desc}` · fail-on: `{fail_on}`", ""]
    for r in results:
        bad = r.fails_threshold(fail_on)
        out.append(f"### {'❌' if bad else '✅'} `{r.path}`")
        if r.metrics:
            m = r.metrics
            out.append(f"Sharpe **{m.get('sharpe', 0):.2f}**, CAGR {m.get('cagr', 0):.1%}, "
                       f"max DD {m.get('max_drawdown', 0):.1%}")
        if r.adapter:
            out.append(f"\n> {_md_escape(r.adapter['caption'])}")
        if r.notebook:
            out += [""] + [f"> {_md_escape(x.strip())}  " for x in _notebook_text(r)]
        elif r.error:
            out.append(f"> Sandbox error: `{_md_escape('; '.join(r.error.splitlines())[:300])}`")
        if r.notebook and r.error and r.code:
            out.append(f"> Sandbox error: `{_md_escape('; '.join(r.error.splitlines())[:300])}`")
        out += ["", "| Test | Result | Severity | Summary | First leak | Lines |", "|---|---|---|---|---|---|"]
        for t in r.tests:
            lines = ", ".join(r.where(x["line"], " ") or f"L{x['line']}" for x in t.get("lines", []))
            out.append(f"| {t['name']} | {'✅ pass' if t['passed'] else '❌ fail'} | {t['severity']} | "
                       f"{_md_escape(t['summary'])} | {first_leak(t) or ''} | {lines} |")
        fails = r.failing()
        if fails:
            by_line: dict[int, dict] = {}
            for t in fails:
                for ln in t.get("lines", []):
                    e = by_line.setdefault(ln["line"], {"snippet": ln["snippet"], "rule": ln["rule"], "tests": []})
                    e["tests"].append(t["name"])
            if by_line:
                out += ["", "**Offending lines**", ""]
                for n, e in sorted(by_line.items()):
                    out.append(f"- `{_ref(r, n)}` `{e['snippet']}` — {e['rule']} ({', '.join(e['tests'])})")
            out += ["", "**Suggested fixes**", ""]
            for t in fails:
                if t.get("suggested_fix"):
                    out.append(f"- **{t['name']}**: {t['suggested_fix']}")
        if r.quick_fix.get("status") == "fixed":
            qf = r.quick_fix
            out += ["", f"**⚡ Quick fix (no LLM, $0)**: `{qf['rule']}` — {qf['description']}; Sharpe "
                        f"{qf['before']['sharpe']:.2f} → {qf['after']['sharpe']:.2f}", "", "```diff",
                    qf["diff"].rstrip(), "```"]
        if r.fix_status:
            out.append(f"\n`--fix`: {r.fix_status}" + (f" → `{r.fixed_path}`" if r.fixed_path else ""))
        out.append("")
    n_bad = sum(bool(r.fails_threshold(fail_on)) for r in results)
    out.append(f"**{len(results) - n_bad}/{len(results)} file(s) passed.**")
    return "\n".join(out) + "\n"


def render_json(results: list[FileResult], fail_on: str, data_desc: str) -> str:
    return json.dumps({"tool": TOOL_NAME, "version": TOOL_VERSION, "data": data_desc, "fail_on": fail_on,
                       "passed": not any(r.fails_threshold(fail_on) for r in results),
                       "files": [r.to_dict(fail_on) for r in results]}, indent=2, default=str) + "\n"


def _info_uri() -> str:
    if os.environ.get("GITHUB_REPOSITORY"):
        return f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
    return INFO_URI


def _cell_prefix(r: FileResult, line: int) -> str:
    """Notebooks: SARIF/annotation regions point at the extracted module, so name the cell in the message.
    (GitHub cannot reliably annotate lines inside .ipynb JSON; the file-level location still lands on the PR.)"""
    w = r.where(line, " ")
    return f"[{w}] " if w else ""


def _cell_props(r: FileResult, line: int) -> dict:
    c = r.cell_of(line)
    return {"notebookCell": c[0], "notebookCellLine": c[1], "lineRefersTo": "extracted module"} if c else {}


def build_sarif(results: list[FileResult], fail_on: str) -> dict:
    rules: dict[str, dict] = {}
    sarif_results = []

    def rule(rid: str, desc: str, sev: str, fix: str = "") -> int:
        if rid not in rules:
            rules[rid] = {"id": rid, "name": rid.replace("/", "_"),
                          "shortDescription": {"text": desc[:120] or rid},
                          "fullDescription": {"text": desc or rid},
                          "help": {"text": fix or desc or rid},
                          "defaultConfiguration": {"level": SARIF_LEVEL.get(sev, "warning")},
                          "properties": {"tags": ["backtest", "look-ahead-bias", "correctness"],
                                         "problem.severity": "error" if sev in ("critical", "high") else "warning"}}
        return list(rules).index(rid)

    def loc(path: str, line: int, snippet: str = "") -> dict:
        region = {"startLine": max(int(line or 1), 1)}
        if snippet:
            region["snippet"] = {"text": snippet}
        return {"physicalLocation": {"artifactLocation": {"uri": path, "uriBaseId": "%SRCROOT%"}, "region": region}}

    for r in results:
        for t in r.failing():
            if t["name"] == "static_scan" and t.get("lines"):
                for f in r.findings:
                    rid = f"static/{f['rule']}"
                    idx = rule(rid, f["message"], f["severity"], f["fix"])
                    sarif_results.append({
                        "ruleId": rid, "ruleIndex": idx, "level": SARIF_LEVEL.get(f["severity"], "warning"),
                        "message": {"text": _cell_prefix(r, f["line"]) + f"{f['message']} Fix: {f['fix']}"},
                        "locations": [loc(r.path, f["line"], f["snippet"])],
                        "properties": {"severity": f["severity"], "test": "static_scan", **_cell_props(r, f["line"])}})
                continue
            rid = f"dynamic/{t['name']}"
            idx = rule(rid, TEST_HELP.get(t["name"], t["name"]), t["severity"], t.get("suggested_fix", ""))
            lines = t.get("lines") or []
            primary = lines[0] if lines else {"line": r.def_line, "snippet": ""}
            msg = _cell_prefix(r, primary["line"]) + t["summary"]
            if first_leak(t):
                msg += f" First leaking date: {first_leak(t)}."
            if lines:
                msg += " Suspect line(s): " + ", ".join(f"{r.where(x['line'], ' ') or x['line']} ({x['rule']})"
                                                        for x in lines) + "."
            if t.get("suggested_fix"):
                msg += f" Fix: {t['suggested_fix']}"
            item = {"ruleId": rid, "ruleIndex": idx, "level": SARIF_LEVEL.get(t["severity"], "warning"),
                    "message": {"text": msg}, "locations": [loc(r.path, primary["line"], primary.get("snippet", ""))],
                    "properties": {"severity": t["severity"], "test": t["name"],
                                   "evidence": t.get("evidence", {}), **_cell_props(r, primary["line"])}}
            if len(lines) > 1:
                item["relatedLocations"] = [dict(loc(r.path, x["line"], x["snippet"]), id=i,
                                                 message={"text": x["rule"]}) for i, x in enumerate(lines[1:], 1)]
            sarif_results.append(item)
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": TOOL_NAME, "version": TOOL_VERSION, "semanticVersion": TOOL_VERSION,
                                "informationUri": _info_uri(), "rules": list(rules.values())}},
            "automationDetails": {"id": "backtest-auditor/"},
            "results": sarif_results,
            "invocations": [{"executionSuccessful": True,
                             "properties": {"fail_on": fail_on,
                                            "files": [r.path for r in results]}}],
        }],
    }


def render_sarif(results: list[FileResult], fail_on: str, data_desc: str) -> str:
    return json.dumps(build_sarif(results, fail_on), indent=2, default=str) + "\n"


def write_report(results: list[FileResult], path: str, data_desc: str) -> str:
    """Self-contained audit report (audit/report.py): HTML unless the path ends in .md/.markdown."""
    from audit.report import render_html, render_markdown, section_from_file_result
    secs = [section_from_file_result(r) for r in results]
    meta = {"data": data_desc, "version": TOOL_VERSION}
    md = path.lower().endswith((".md", ".markdown"))
    text = render_markdown(secs, meta) if md else render_html(secs, meta)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


RENDERERS = {"text": render_text, "markdown": render_markdown, "json": render_json, "sarif": render_sarif}


def _gh_escape(s: str, prop: bool = False) -> str:
    s = str(s).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        s = s.replace(":", "%3A").replace(",", "%2C")
    return s


def github_annotations(results: list[FileResult]) -> list[str]:
    """GitHub Actions workflow commands, one per offending line (or per failing test)."""
    out = []
    for r in results:
        for t in r.failing():
            level = GH_LEVEL.get(t["severity"], "warning")
            lines = t.get("lines") or [{"line": r.def_line, "rule": t["name"], "snippet": ""}]
            base_level = level
            for ln in lines:
                level = base_level
                title = f"{TOOL_NAME}: {t['name']} ({t['severity']})"
                msg = t["summary"]
                if first_leak(t):
                    msg += f" First leaking date: {first_leak(t)}."
                fix = t.get("suggested_fix", "")
                if t["name"] == "static_scan" and ln.get("message"):
                    msg, fix = ln["message"], ln.get("fix", fix)
                    level = GH_LEVEL.get(ln.get("severity", t["severity"]), level)
                if ln.get("snippet"):
                    msg += f" Line: {ln['snippet']} [{ln['rule']}]."
                if fix:
                    msg += f" Fix: {fix}"
                out.append(f"::{level} file={_gh_escape(r.path, True)},line={ln['line']},"
                           f"title={_gh_escape(title, True)}::{_gh_escape(_cell_prefix(r, ln['line']) + msg)}")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audit_cli.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="strategy .py / .ipynb files or globs (quoted globs are expanded, "
                                             "** allowed)")
    ap.add_argument("--data", default="synthetic", help="'synthetic' (default, seeded panel) or 'csv:PATH'")
    ap.add_argument("--format", choices=list(RENDERERS), default="text", help="report format")
    ap.add_argument("--strategy-format", "--input-format", dest="strategy_format", default="auto",
                    choices=["auto", "signals", "function", "vectorbt", "backtrader"],
                    help="strategy code format (audit/adapters.py); auto-detected by default")
    ap.add_argument("--fail-on", choices=list(FAIL_ON), default="high",
                    help="exit 1 if any test of this severity or worse fails (warning = medium/low too)")
    ap.add_argument("--out", default="", help="write the report here instead of stdout")
    ap.add_argument("--sarif-out", default="", help="also write a SARIF 2.1.0 file (for code scanning)")
    ap.add_argument("--annotations", choices=["auto", "always", "never"], default="auto",
                    help="GitHub ::error annotations (auto = when GITHUB_ACTIONS=true)")
    ap.add_argument("--seed", type=int, default=7, help="seed for perturbation cut dates")
    ap.add_argument("--n-cuts", type=int, default=6)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=90.0, help="sandbox wall-clock seconds per file")
    ap.add_argument("--no-pbo", action="store_true",
                    help="skip the Probability of Backtest Overfitting test (parameter grid, CSCV)")
    ap.add_argument("--param-grid", default="",
                    help="explicit PBO grid for UPPERCASE constants / generate_signals defaults, "
                         "e.g. 'LOOKBACK=10,20,40;TOP=0.8,0.9' (default: automatic 0.5x-2x grid)")
    ap.add_argument("--report", default="",
                    help="also write a self-contained audit report: .html (printable, inline SVG) or .md")
    ap.add_argument("--fix", action="store_true",
                    help="repair failing files: deterministic Quick fix first (prints a diff, no LLM), then the "
                         "Nemotron agent loop (needs NEBIUS_API_KEY) if no candidate passes")
    ap.add_argument("--write", action="store_true", help="with --fix: write the Quick fix to <name>.fixed.py")
    ap.add_argument("--no-quickfix", action="store_true", help="with --fix: skip the Quick fix, LLM only")
    ap.add_argument("--no-llm", action="store_true", help="with --fix: Quick fix only, never call an LLM")
    ap.add_argument("--mock-llm", action="store_true", help="with --fix: use the offline mock LLM (demo/tests)")
    ap.add_argument("--max-iters", type=int, default=4, help="with --fix: agent loop iterations")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    files, missing = expand_paths(a.paths)
    for m in missing:
        print(f"audit_cli: no such file or no match: {m}", file=sys.stderr)
    if not files:
        print("audit_cli: nothing to audit", file=sys.stderr)
        return 2
    try:
        data = load_data(a.data)
    except Exception as e:  # noqa: BLE001
        print(f"audit_cli: cannot load data: {e}", file=sys.stderr)
        return 2
    close = data["close"]
    data_desc = (f"{a.data} ({close.shape[1]} tickers, {len(close)} dates, "
                 f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d})")

    results = []
    for p in files:
        r = audit_file(p, data, seed=a.seed, n_cuts=a.n_cuts, cost_bps=a.cost_bps, timeout=a.timeout,
                       pbo=not a.no_pbo, param_grid=parse_param_grid(a.param_grid),
                       strategy_format=a.strategy_format)
        if a.fix:
            try:
                propose_fix(p, r, data, mock=a.mock_llm, max_iters=a.max_iters, cost_bps=a.cost_bps,
                            quick=not a.no_quickfix, write=a.write, llm=not a.no_llm, seed=a.seed,
                            n_cuts=a.n_cuts, timeout=a.timeout)
            except Exception as e:  # noqa: BLE001 - a failed fix must never hide the audit result
                r.fix_status = f"error: {e}"
            if r.fix_status == "skipped_no_api_key":
                print(f"audit_cli: --fix skipped for {r.path}: NEBIUS_API_KEY is not set", file=sys.stderr)
        results.append(r)

    report = RENDERERS[a.format](results, a.fail_on, data_desc)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    if a.sarif_out:
        Path(a.sarif_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.sarif_out).write_text(render_sarif(results, a.fail_on, data_desc), encoding="utf-8")

    if a.report:
        write_report(results, a.report, data_desc)

    in_gha = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    if a.annotations == "always" or (a.annotations == "auto" and in_gha):
        # keep machine-readable stdout clean: annotations go to stderr then (the runner reads both streams)
        stream = sys.stderr if (not a.out and a.format in ("json", "sarif")) else sys.stdout
        for line in github_annotations(results):
            print(line, file=stream)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if in_gha and summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(render_markdown(results, a.fail_on, data_desc))
        except OSError:
            pass
    return 1 if any(r.fails_threshold(a.fail_on) for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())

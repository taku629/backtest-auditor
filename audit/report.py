"""Self-contained, printable audit report (HTML and Markdown).

No external CSS, JS, fonts or images: the equity curve is an inline SVG, so the file can be e-mailed, attached
to a PR or printed to PDF from a browser.

Two entry points build the same "section" structure:

* ``section_from_trace(trace_dict)``: a run of the agent loop (UI, ``cli.py``);
* ``section_from_file_result(FileResult)``: one file audited by ``audit_cli.py``.

``render_html(sections, meta)`` / ``render_markdown(sections, meta)`` turn a list of sections into a document.
"""
from __future__ import annotations

import difflib
import hashlib
import html
import math
from datetime import datetime, timezone

TOOL = "Backtest Auditor"
SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
RED, GREEN, GREY, AMBER = "#b42318", "#067647", "#667085", "#b54708"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def code_hash(code: str | None) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()[:16] if code else ""


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _status(t: dict) -> str:
    if t.get("passed"):
        ev = t.get("evidence") or {}
        return "SKIP" if ev.get("status") == "skipped" else "PASS"
    return "FAIL" if t.get("blocking") else "WARN"


def _sorted_tests(tests: list[dict]) -> list[dict]:
    rank = {"FAIL": 0, "WARN": 1, "PASS": 2, "SKIP": 3}
    return sorted(tests or [], key=lambda t: (rank[_status(t)], SEV_ORDER.get(t.get("severity"), 9)))


def _f(x, fmt="{:.2f}", none="–"):
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return none
        return fmt.format(x)
    except (TypeError, ValueError):
        return str(x)


def evidence_lines(t: dict) -> list[str]:
    """The few evidence facts a reader needs, per test."""
    ev, n = t.get("evidence") or {}, t.get("name")
    out: list[str] = []
    if n == "future_perturbation":
        out.append(f"leaking cut dates: {ev.get('n_leaking_cuts')}/{ev.get('n_cuts')}; first leaking date "
                   f"{ev.get('first_leaking_date') or '–'}; max |weight change| {ev.get('max_abs_weight_change')}")
        lz = ev.get("localized") or {}
        if lz.get("found") and lz.get("entry"):
            e = lz["entry"]
            out.append(f"leak enters at {e.get('where') or 'line ' + str(e.get('line'))}: `{e.get('source_line', '').strip()}`"
                       f" ({e.get('kind', '')})")
    elif n == "survivorship":
        for k in ("weight_before_listing", "counterfactual_revived", "counterfactual_rows_changed",
                  "counterfactual_first_change", "data_warning"):
            if k in ev:
                out.append(f"{k.replace('_', ' ')}: {ev[k]}")
    elif n == "execution_delay":
        out.append(f"Sharpe t+1 {ev.get('sharpe_lag1')} vs t+2 {ev.get('sharpe_lag2')} (drop {_f(ev.get('relative_drop'), '{:.0%}')})")
    elif n == "static_scan":
        for f in (ev.get("findings") or [])[:8]:
            out.append(f"line {f.get('line')}: {f.get('rule')} ({f.get('severity')}) `{str(f.get('snippet', '')).strip()}`")
    elif n == "deflated_sharpe":
        out.append(f"DSR {_f(ev.get('dsr'))} (threshold {ev.get('threshold')}), PSR {_f(ev.get('psr'))}, "
                   f"{ev.get('n_trials')} trial(s), {ev.get('n_obs')} days")
    elif n == "train_test_split":
        out.append(f"Sharpe train {_f(ev.get('sharpe_train'))} vs test {_f(ev.get('sharpe_test'))}")
    elif n == "plausibility":
        out.append(f"Sharpe {_f(ev.get('sharpe'))}, CAGR {_f(ev.get('cagr'), '{:.1%}')} (max plausible {ev.get('max_sharpe')})")
    elif n == "pbo":
        if ev.get("status") == "ok":
            out.append(f"PBO {_f(ev.get('pbo'))} ({ev.get('risk')} risk; fails above {ev.get('threshold')}); "
                       f"{ev.get('n_variants')} variants × {ev.get('n_splits')} CSCV splits ({ev.get('n_blocks')} blocks)")
            out.append(f"P(OOS loss) {_f(ev.get('prob_oos_loss'), '{:.0%}')}; degradation slope "
                       f"{_f(ev.get('degradation_slope'))}")
            out.append("grid: " + "; ".join(f"{p['name']} ∈ {{{', '.join(f'{v:g}' for v in p['values'])}}}"
                                            for p in ev.get("params", [])))
        else:
            out.append(f"skipped: {ev.get('reason')}")
    elif n == "sandbox":
        out.append(str(ev.get("error", ""))[:400])
    return out


def _localized(tests: list[dict]) -> dict | None:
    for t in tests or []:
        if t.get("name") == "future_perturbation" and not t.get("passed"):
            lz = (t.get("evidence") or {}).get("localized") or {}
            if lz.get("found") and (lz.get("entry") or {}).get("line"):
                return lz["entry"]
    return None


def _flag_lines(tests: list[dict]) -> dict[int, str]:
    out: dict[int, str] = {}
    for t in tests or []:
        if t.get("name") == "static_scan":
            for f in (t.get("evidence") or {}).get("findings") or []:
                ln = int(f.get("line") or 0)
                out[ln] = (out.get(ln, "") + ("; " if ln in out else "") + f"{f.get('rule')} ({f.get('severity')})")
    return out


def _token_totals(calls: list[dict]) -> list[dict]:
    tot: dict[str, dict] = {}
    for c in calls or []:
        t = tot.setdefault(c.get("model", "?"), {"model": c.get("model", "?"), "role": c.get("role", ""), "calls": 0,
                                                "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        t["calls"] += 1
        t["prompt_tokens"] += int(c.get("prompt_tokens") or 0)
        t["completion_tokens"] += int(c.get("completion_tokens") or 0)
        t["cost_usd"] += float(c.get("cost_usd") or 0.0)
    return list(tot.values())


# ----------------------------------------------------------------------------
# section builders
# ----------------------------------------------------------------------------
def section_from_trace(trace: dict) -> dict:
    its = trace.get("iterations") or []
    scored = [i for i in its if i.get("metrics")]
    first = scored[0] if scored else (its[0] if its else {})
    last = scored[-1] if scored else (its[-1] if its else {})
    status = trace.get("final_status", "unknown")
    draft_code = (its[0].get("code") if its else "") or ""
    final_code = (its[-1].get("code") if its else "") or ""
    fix = None
    qf_it = next((i for i in its if i.get("source") == "quick_fix"), None)
    if qf_it is not None and (qf_it.get("quick_fix") or {}).get("diff"):
        q = qf_it["quick_fix"]
        fix = {"kind": "Quick fix (deterministic, no LLM, $0)",
               "description": f"rule `{q.get('rule')}` on line {', '.join(map(str, q.get('lines') or []))}: "
                              f"{q.get('description')}",
               "diff": q["diff"]}
        if len(its) >= 2 and its[-1] is not qf_it:
            fix["description"] += " (later iterations changed the code further; full diff below)"
    if len(its) >= 2 and draft_code != final_code and (fix is None or its[-1] is not qf_it):
        d = "\n".join(difflib.unified_diff(draft_code.splitlines(), final_code.splitlines(), "draft.py", "final.py",
                                           lineterm=""))
        coder = (trace.get("models") or {}).get("coder", "LLM")
        fix = {"kind": f"LLM repair ({coder.split('/')[-1]}{', mock' if trace.get('mock') else ''})",
               "description": f"{len(its)} iterations; draft vs final code", "diff": d}
    equity = []
    if first.get("equity"):
        equity.append({"label": f"First draft (Sharpe {_f((first.get('metrics') or {}).get('sharpe'))})",
                       "dates": first.get("dates") or [], "values": first["equity"], "color": RED})
    if last is not first and last.get("equity"):
        equity.append({"label": f"{'Audited final' if status == 'clean' else 'Last iteration'} "
                                f"(Sharpe {_f((last.get('metrics') or {}).get('sharpe'))})",
                       "dates": last.get("dates") or [], "values": last["equity"], "color": GREEN})
    iters = []
    for i in its:
        src = {"quick_fix": "Quick fix", "user": "user code"}.get(i.get("source"), "Coder")
        if i.get("user_code"):
            src = "user code"
        iters.append({"index": i.get("index", 0) + 1, "source": src, "status": i.get("status"),
                      "sharpe": (i.get("metrics") or {}).get("sharpe"),
                      "failing": [t["name"] for t in i.get("tests") or [] if not t.get("passed")],
                      "error": (i.get("error") or "")[:300]})
    return {
        "name": "User strategy" if (trace.get("settings") or {}).get("user_code") else "Strategy written by the Coder",
        "hypothesis": trace.get("hypothesis", ""),
        "verdict": status,
        "code": draft_code, "code_hash": code_hash(draft_code),
        "final_code": final_code if final_code != draft_code else "", "final_code_hash":
            code_hash(final_code) if final_code != draft_code else "",
        "tests": last.get("tests") or [], "tests_label": f"iteration {last.get('index', 0) + 1}",
        "draft_tests": first.get("tests") if first is not last else None,
        "localized": _localized(first.get("tests") or []),
        "flag_lines": _flag_lines(first.get("tests") or []),
        "fix": fix,
        "metrics_before": first.get("metrics") or {}, "metrics_after": last.get("metrics") if last is not first else None,
        "equity": equity,
        "iterations": iters,
        "summary": trace.get("summary", ""),
        "calls": _token_totals(trace.get("calls") or []),
        "models": trace.get("models") or {},
        "mock": bool(trace.get("mock")),
    }


def section_from_file_result(r) -> dict:
    """``audit_cli.FileResult`` -> section (deterministic audit, optional Quick fix)."""
    tests = r.tests
    clean = not any(t.get("blocking") for t in tests)
    fix, after = None, None
    qf = r.quick_fix or {}
    if qf.get("status") == "fixed":
        fix = {"kind": "Quick fix (deterministic, no LLM, $0)",
               "description": f"rule `{qf.get('rule')}` on line {', '.join(map(str, qf.get('lines') or []))}: "
                              f"{qf.get('description')}", "diff": qf.get("diff", "")}
        after = getattr(r, "fix_metrics", None) or qf.get("after")
    elif qf:
        fix = {"kind": "Quick fix", "description": f"status {qf.get('status')}: {qf.get('n_candidates', 0)} "
                                                   "candidate(s), none passed without going to cash", "diff": ""}
    equity = []
    if getattr(r, "equity", None):
        equity.append({"label": f"As written (Sharpe {_f(r.metrics.get('sharpe'))})", "dates": r.equity["dates"],
                       "values": r.equity["values"], "color": RED if not clean else GREEN})
    if getattr(r, "fix_equity", None):
        equity.append({"label": f"After Quick fix (Sharpe {_f((after or {}).get('sharpe'))})",
                       "dates": r.fix_equity["dates"], "values": r.fix_equity["values"], "color": GREEN})
    return {
        "name": r.path, "hypothesis": "",
        "verdict": "error" if r.error else ("clean" if clean else "not_clean"),
        "code": r.code, "code_hash": code_hash(r.code),
        "final_code": (qf.get("code") or "") if qf.get("status") == "fixed" else "",
        "final_code_hash": code_hash(qf.get("code")) if qf.get("status") == "fixed" else "",
        "tests": tests, "tests_label": "as written", "draft_tests": None,
        "localized": _localized(tests), "flag_lines": _flag_lines(tests),
        "fix": fix, "metrics_before": r.metrics or {}, "metrics_after": after,
        "equity": equity, "iterations": [], "summary": "", "calls": [], "models": {}, "mock": False,
        "error": r.error,
    }


# ----------------------------------------------------------------------------
# SVG equity chart
# ----------------------------------------------------------------------------
def equity_svg(series: list[dict], width: int = 760, height: int = 260) -> str:
    series = [s for s in series if s.get("values")]
    if not series:
        return ""
    pad_l, pad_r, pad_t, pad_b = 52, 12, 12, 26
    W, H = width - pad_l - pad_r, height - pad_t - pad_b
    allv = [v for s in series for v in s["values"] if v and v > 0 and math.isfinite(v)]
    if not allv:
        return ""
    lo, hi = math.log(min(allv)), math.log(max(allv))
    if hi - lo < 1e-9:
        lo, hi = lo - 0.1, hi + 0.1
    n_max = max(len(s["values"]) for s in series)

    def xy(i, v, n):
        x = pad_l + (i / max(n - 1, 1)) * W * (n / n_max if n_max else 1)
        y = pad_t + H - (math.log(max(v, 1e-12)) - lo) / (hi - lo) * H
        return f"{x:.1f},{y:.1f}"

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="Equity curves, growth of $1, log scale" class="eq">',
             f'<rect x="{pad_l}" y="{pad_t}" width="{W}" height="{H}" fill="#fff" stroke="#d0d5dd"/>']
    # y ticks at "nice" multiples
    cands = [m * 10.0 ** e for e in range(-3, 9) for m in (1, 2, 5)]
    ticks = [v for v in cands if lo <= math.log(v) <= hi]
    if len(ticks) > 7:
        ticks = [v for v in ticks if math.log10(v).is_integer()] or ticks[:: max(1, len(ticks) // 6)]
        if len(ticks) > 7:
            ticks = ticks[:: math.ceil(len(ticks) / 6)]
    for v in ticks:
        lv = math.log(v)
        if True:
            y = pad_t + H - (lv - lo) / (hi - lo) * H
            parts.append(f'<line x1="{pad_l}" x2="{pad_l + W}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eaecf0"/>'
                         f'<text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end" fill="#475467">'
                         f'${v:,.4g}</text>')
    for s in series:
        n = len(s["values"])
        step = max(1, n // 500)
        pts = " ".join(xy(i, v, n) for i, v in enumerate(s["values"]) if i % step == 0 or i == n - 1)
        parts.append(f'<polyline fill="none" stroke="{s.get("color", GREY)}" stroke-width="1.6" points="{pts}"/>')
    dates = max(series, key=lambda s: len(s["values"])).get("dates") or []
    if dates:
        parts.append(f'<text x="{pad_l}" y="{height - 8}" font-size="10" fill="#475467">{html.escape(str(dates[0])[:10])}</text>'
                     f'<text x="{pad_l + W}" y="{height - 8}" font-size="10" text-anchor="end" fill="#475467">'
                     f'{html.escape(str(dates[-1])[:10])}</text>')
    parts.append("</svg>")
    legend = " ".join(f'<span class="lg"><i style="background:{s.get("color", GREY)}"></i>{html.escape(s["label"])}</span>'
                      for s in series)
    return "".join(parts) + f'<div class="legend">{legend} <span class="muted">· growth of $1, log scale</span></div>'


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
CSS = """
*{box-sizing:border-box} body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
color:#101828;max-width:900px;margin:24px auto;padding:0 20px;background:#fff}
h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 8px;padding-top:10px;border-top:2px solid #101828}
h3{font-size:15px;margin:20px 0 6px} .muted{color:#667085} .meta{font-size:12px;color:#475467}
table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:13px} th,td{border:1px solid #d0d5dd;
padding:5px 7px;text-align:left;vertical-align:top} th{background:#f2f4f7;font-weight:600}
.verdict{border:2px solid;border-radius:6px;padding:10px 14px;margin:10px 0;font-size:16px;font-weight:700}
.v-clean{border-color:#067647;color:#067647} .v-bad{border-color:#b42318;color:#b42318}
.st{font-weight:700;font-size:12px} .FAIL{color:#b42318} .WARN{color:#b54708} .PASS{color:#067647} .SKIP{color:#667085}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
pre{background:#f9fafb;border:1px solid #d0d5dd;border-radius:4px;padding:8px;overflow-x:auto;white-space:pre-wrap;
word-break:break-all} .code{border:1px solid #d0d5dd;border-radius:4px;overflow:hidden;font-size:12px}
.cl{white-space:pre-wrap;word-break:break-all;padding:0 8px;font-family:ui-monospace,Menlo,Consolas,monospace}
.cl .ln{display:inline-block;width:3em;color:#98a2b3} .cl.leak{background:#fee4e2;font-weight:700;
border-left:4px solid #b42318} .cl.flag{background:#fffaeb} .note{color:#b42318;font-size:11px}
.diff .add{color:#067647;background:#ecfdf3} .diff .del{color:#b42318;background:#fef3f2} .diff .hdr{color:#475467}
.legend{font-size:12px;margin:2px 0 8px} .lg{margin-right:14px} .lg i{display:inline-block;width:14px;height:3px;
vertical-align:middle;margin-right:5px} ul{margin:4px 0 8px 18px;padding:0} li{margin:1px 0}
@media print{body{margin:0;max-width:none} h2{break-before:auto} .code,pre,table,svg{break-inside:avoid}}
"""


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _md_inline_code_html(s: str) -> str:
    """Escape, then turn `x` into <code>x</code>."""
    parts = _e(s).split("`")
    return "".join(f"<code>{p}</code>" if k % 2 else p for k, p in enumerate(parts))


def _code_html(code: str, leak_line: int | None, flags: dict[int, str], leak_note: str = "") -> str:
    lines = code.splitlines()
    if len(lines) > 90 and leak_line:
        lo, hi = max(1, leak_line - 12), min(len(lines), leak_line + 12)
    else:
        lo, hi = 1, len(lines)
    out = []
    if lo > 1:
        out.append(f'<div class="cl muted">… lines 1–{lo - 1} omitted</div>')
    for i in range(lo, hi + 1):
        text = lines[i - 1]
        cls, note = "", ""
        if i == leak_line:
            cls, note = " leak", f'<span class="note">  ◀ LEAK ENTERS HERE{": " + _e(leak_note) if leak_note else ""}</span>'
        elif i in flags:
            cls, note = " flag", f'<span class="note">  ◀ {_e(flags[i])}</span>'
        out.append(f'<div class="cl{cls}"><span class="ln">{i}</span>{_e(text) or " "}{note}</div>')
    if hi < len(lines):
        out.append(f'<div class="cl muted">… lines {hi + 1}–{len(lines)} omitted</div>')
    return '<div class="code">' + "".join(out) + "</div>"


def _diff_html(diff: str) -> str:
    rows = []
    for ln in diff.splitlines():
        cls = ("hdr" if ln.startswith(("+++", "---", "@@")) else "add" if ln.startswith("+") else
               "del" if ln.startswith("-") else "")
        rows.append(f'<span class="{cls}">{_e(ln)}</span>')
    return '<pre class="diff">' + "\n".join(rows) + "</pre>"


METRIC_ROWS = [("Sharpe", "sharpe", "{:.2f}"), ("CAGR", "cagr", "{:.1%}"), ("Max drawdown", "max_drawdown", "{:.1%}"),
               ("Turnover / yr", "turnover_annual", "{:.1f}x"), ("Hit rate", "hit_rate", "{:.1%}"),
               ("Avg gross exposure", "avg_gross", "{:.2f}")]


def _metrics_table_html(before: dict, after: dict | None, after_label: str) -> str:
    if not before:
        return ""
    head = "<tr><th>Metric</th><th>As first audited</th>" + (f"<th>{_e(after_label)}</th>" if after else "") + "</tr>"
    rows = []
    for lab, k, fmt in METRIC_ROWS:
        if k not in before and not (after and k in after):
            continue
        rows.append(f"<tr><td>{lab}</td><td>{_f(before.get(k), fmt)}</td>" +
                    (f"<td>{_f(after.get(k), fmt)}</td>" if after else "") + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def _tests_html(tests: list[dict]) -> str:
    rows = []
    for t in _sorted_tests(tests):
        s = _status(t)
        ev = "".join(f"<li>{_md_inline_code_html(x)}</li>" for x in evidence_lines(t))
        fix = (f'<div class="muted">Fix: {_e(t.get("suggested_fix"))}</div>' if t.get("suggested_fix") and s != "PASS"
               else "")
        rows.append(f'<tr><td class="st {s}">{s}</td><td><b>{_e(t.get("name"))}</b><br><span class="muted">'
                    f'{_e(t.get("severity"))}</span></td><td>{_e(t.get("summary"))}'
                    f'{"<ul>" + ev + "</ul>" if ev else ""}{fix}</td></tr>')
    return "<table><tr><th>Result</th><th>Test</th><th>Finding and evidence</th></tr>" + "".join(rows) + "</table>"


def _section_html(sec: dict, k: int, n: int) -> str:
    v = sec.get("verdict")
    clean = v == "clean"
    tests = sec.get("tests") or []
    n_block = sum(1 for t in tests if t.get("blocking"))
    n_warn = sum(1 for t in tests if not t.get("passed") and not t.get("blocking"))
    verdict = ("CLEAN: no critical or high-severity test fails" if clean else
               "ERROR: the strategy could not be executed" if v == "error" else
               f"NOT CLEAN: {n_block} blocking failure(s). Do not trust this backtest.")
    if clean and n_warn:
        verdict += f" · {n_warn} warning(s)"
    out = [f"<h2>{k}. {_e(sec.get('name'))}</h2>" if n > 1 else f"<h2>{_e(sec.get('name'))}</h2>"]
    if sec.get("hypothesis"):
        out.append(f"<p><b>Hypothesis:</b> {_e(sec['hypothesis'])}</p>")
    out.append(f'<div class="verdict {"v-clean" if clean else "v-bad"}">{_e(verdict)}</div>')
    out.append(f'<p class="meta">Audited code SHA-256: <code>{_e(sec.get("code_hash"))}</code>'
               + (f' · final code SHA-256: <code>{_e(sec["final_code_hash"])}</code>' if sec.get("final_code_hash") else "")
               + "</p>")
    if clean:
        out.append('<p class="muted">Clean means honest (no look-ahead or survivorship bias detected), not profitable.</p>')
    if sec.get("error"):
        out.append(f"<pre>{_e(sec['error'][:1500])}</pre>")
    if sec.get("summary"):
        out.append("<h3>Executive summary</h3>" + "".join(f"<p>{_md_inline_code_html(p)}</p>"
                                                        for p in sec["summary"].split("\n") if p.strip()))
    if sec.get("draft_tests"):
        out.append("<h3>Tests on the first draft</h3>" + _tests_html(sec["draft_tests"]))
    out.append(f"<h3>Deterministic tests ({_e(sec.get('tests_label'))})</h3>" + _tests_html(tests))
    loc = sec.get("localized")
    if sec.get("code"):
        title = "Audited code" + (" · leak line highlighted" if loc else "")
        out.append(f"<h3>{title}</h3>")
        if loc:
            out.append(f"<p>Leak enters at <b>{_e(loc.get('where') or 'line ' + str(loc.get('line')))}</b>: "
                       f"<code>{_e(str(loc.get('source_line', '')).strip())}</code>. Found by experiment: the first "
                       "statement whose value changes when only future data changes.</p>")
        out.append(_code_html(sec["code"], (loc or {}).get("line"), sec.get("flag_lines") or {},
                              (loc or {}).get("kind", "")))
    fx = sec.get("fix")
    if fx:
        out.append(f"<h3>{_e(fx['kind'])}</h3><p>{_md_inline_code_html(fx.get('description', ''))}</p>")
        if fx.get("diff"):
            out.append(_diff_html(fx["diff"]))
    if sec.get("metrics_before"):
        out.append("<h3>Before / after</h3>" + _metrics_table_html(sec["metrics_before"], sec.get("metrics_after"),
                                                                  "Audited final" if clean else "After fix"))
    svg = equity_svg(sec.get("equity") or [])
    if svg:
        out.append(svg)
    if sec.get("iterations"):
        rows = "".join(f"<tr><td>{i['index']}</td><td>{_e(i['source'])}</td><td>{_e(i['status'])}</td>"
                       f"<td>{_f(i.get('sharpe'))}</td><td>{_e(', '.join(i['failing']) or '–')}"
                       f"{'<br><span class=muted>' + _e(i['error']) + '</span>' if i.get('error') else ''}</td></tr>"
                       for i in sec["iterations"])
        out.append("<h3>Iterations</h3><table><tr><th>#</th><th>Code by</th><th>Status</th><th>Sharpe</th>"
                   f"<th>Failing tests</th></tr>{rows}</table>")
    calls = sec.get("calls") or []
    out.append("<h3>Models and token cost</h3>")
    if calls:
        tot = sum(c["cost_usd"] for c in calls)
        rows = "".join(f"<tr><td><code>{_e(c['model'])}</code></td><td>{c['calls']}</td><td>{c['prompt_tokens']:,}</td>"
                       f"<td>{c['completion_tokens']:,}</td><td>${c['cost_usd']:.4f}</td></tr>" for c in calls)
        out.append("<table><tr><th>Model</th><th>Calls</th><th>Prompt tokens</th><th>Completion tokens</th>"
                   f"<th>Est. cost</th></tr>{rows}<tr><td><b>Total</b></td><td></td><td></td><td></td>"
                   f"<td><b>${tot:.4f}</b></td></tr></table>")
        if sec.get("mock"):
            out.append('<p class="muted">Mock run: model IDs show the configured routing; token counts are '
                       "estimates; no API was called.</p>")
    else:
        out.append('<p class="muted">No LLM calls: deterministic audit only ($0).</p>')
    return "\n".join(out)


def render_html(sections: list[dict], meta: dict | None = None) -> str:
    meta = meta or {}
    title = meta.get("title") or f"{TOOL} audit report"
    clean_all = all(s.get("verdict") == "clean" for s in sections)
    head = (f"<h1>{_e(title)}</h1><p class='meta'>Generated {_e(meta.get('generated_at') or now_utc())} · "
            f"{_e(TOOL)} {_e(meta.get('version', ''))} · data: {_e(meta.get('data', '–'))}"
            + (f" · {len(sections)} strategies, {sum(s.get('verdict') == 'clean' for s in sections)} clean"
               if len(sections) > 1 else "") + "</p>")
    body = "\n".join(_section_html(s, k, len(sections)) for k, s in enumerate(sections, 1))
    foot = ("<p class='meta' style='margin-top:30px;border-top:1px solid #d0d5dd;padding-top:8px'>Every verdict above "
            "comes from deterministic tests (perturbation, survivorship, execution delay, static scan, DSR, PBO). "
            "LLMs write and explain code; they never decide what is clean. Thresholds are heuristics; synthetic data "
            "is a test bench, not a claim about markets.</p>")
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{_e(title)}</title><style>{CSS}</style></head>"
            f"<body data-verdict='{'clean' if clean_all else 'not_clean'}'>{head}{body}{foot}</body></html>\n")


# ----------------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------------
def _md(s) -> str:
    return str(s if s is not None else "").replace("|", "\\|").replace("\n", " ")


def _section_md(sec: dict, k: int, n: int) -> list[str]:
    v = sec.get("verdict")
    clean = v == "clean"
    tests = sec.get("tests") or []
    n_block = sum(1 for t in tests if t.get("blocking"))
    out = [f"## {f'{k}. ' if n > 1 else ''}{sec.get('name')}", ""]
    if sec.get("hypothesis"):
        out += [f"**Hypothesis:** {sec['hypothesis']}", ""]
    out += [("**Verdict: ✅ CLEAN** (no critical or high-severity test fails; honest, not necessarily profitable)"
             if clean else "**Verdict: ⚠️ ERROR** (strategy could not be executed)" if v == "error" else
             f"**Verdict: ❌ NOT CLEAN** ({n_block} blocking failure(s); do not trust this backtest)"), "",
            f"Audited code SHA-256: `{sec.get('code_hash')}`" +
            (f" · final code SHA-256: `{sec['final_code_hash']}`" if sec.get("final_code_hash") else ""), ""]
    if sec.get("summary"):
        out += ["### Executive summary", "", sec["summary"].strip(), ""]
    out += [f"### Deterministic tests ({sec.get('tests_label')})", "", "| Result | Test | Severity | Finding | Evidence |",
            "|---|---|---|---|---|"]
    for t in _sorted_tests(tests):
        out.append(f"| {_status(t)} | {t.get('name')} | {t.get('severity')} | {_md(t.get('summary'))} | "
                   f"{_md('; '.join(evidence_lines(t)))} |")
    out.append("")
    loc = sec.get("localized")
    if sec.get("code"):
        lines = sec["code"].splitlines()
        if loc:
            ln = int(loc["line"])
            out += [f"### Leak line", "", f"Leak enters at **{loc.get('where') or 'line ' + str(ln)}**: "
                                          f"`{str(loc.get('source_line', '')).strip()}` (found by perturbation trace).", ""]
            lo, hi = max(1, ln - 4), min(len(lines), ln + 4)
        else:
            lo, hi = 1, len(lines)
        out.append("```python")
        flags = sec.get("flag_lines") or {}
        for i in range(lo, hi + 1):
            mark = "  # <-- LEAK ENTERS HERE" if loc and i == int(loc["line"]) else (f"  # <-- {flags[i]}" if i in flags else "")
            out.append(f"{i:>3}  {lines[i - 1]}{mark}")
        out += ["```", ""]
    fx = sec.get("fix")
    if fx:
        out += [f"### {fx['kind']}", "", fx.get("description", ""), ""]
        if fx.get("diff"):
            out += ["```diff", fx["diff"].rstrip(), "```", ""]
    b, a = sec.get("metrics_before") or {}, sec.get("metrics_after")
    if b:
        out += ["### Before / after", "", "| Metric | As first audited |" + (" After |" if a else ""),
                "|---|---|" + ("---|" if a else "")]
        for lab, key, fmt in METRIC_ROWS:
            out.append(f"| {lab} | {_f(b.get(key), fmt)} |" + (f" {_f(a.get(key), fmt)} |" if a else ""))
        out.append("")
    if sec.get("iterations"):
        out += ["### Iterations", "", "| # | Code by | Status | Sharpe | Failing tests |", "|---|---|---|---|---|"]
        for i in sec["iterations"]:
            out.append(f"| {i['index']} | {i['source']} | {i['status']} | {_f(i.get('sharpe'))} | "
                       f"{_md(', '.join(i['failing']) or '–')} |")
        out.append("")
    out += ["### Models and token cost", ""]
    calls = sec.get("calls") or []
    if calls:
        out += ["| Model | Calls | Prompt tok | Completion tok | Est. cost |", "|---|---|---|---|---|"]
        for c in calls:
            out.append(f"| `{c['model']}` | {c['calls']} | {c['prompt_tokens']} | {c['completion_tokens']} | "
                       f"${c['cost_usd']:.4f} |")
        out.append(f"| **Total** | | | | **${sum(c['cost_usd'] for c in calls):.4f}** |")
        if sec.get("mock"):
            out.append("\n_Mock run: token counts are estimates; no API was called._")
    else:
        out.append("No LLM calls: deterministic audit only ($0).")
    out.append("")
    return out


def render_markdown(sections: list[dict], meta: dict | None = None) -> str:
    meta = meta or {}
    out = [f"# {meta.get('title') or TOOL + ' audit report'}", "",
           f"Generated {meta.get('generated_at') or now_utc()} · {TOOL} {meta.get('version', '')} · "
           f"data: {meta.get('data', '–')}", ""]
    for k, s in enumerate(sections, 1):
        out += _section_md(s, k, len(sections))
    out.append("_Verdicts come from deterministic tests; LLMs write and explain code but never decide what is clean._")
    return "\n".join(out) + "\n"


def report_from_trace(trace: dict, fmt: str = "html", data_desc: str = "", version: str = "0.2.0") -> str:
    meta = {"data": data_desc or (f"{(trace.get('settings') or {}).get('n_tickers', '?')} tickers × "
                                  f"{(trace.get('settings') or {}).get('n_dates', '?')} days"), "version": version}
    sec = [section_from_trace(trace)]
    return render_html(sec, meta) if fmt == "html" else render_markdown(sec, meta)

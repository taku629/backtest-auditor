"""Backtest Auditor - Streamlit UI.

Run offline:  USE_MOCK=1 streamlit run app.py
Run live:     NEBIUS_API_KEY=... streamlit run app.py
"""
from __future__ import annotations

import difflib
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our notebook.py wins over Jupyter's

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.config import load_config, use_mock
from agents.llm import make_llm
from agents.loop import RunTrace, run_audit_loop
from data import generate_panel, listing_table, load_long_csv
from engine import run_backtest

st.set_page_config(page_title="Backtest Auditor", page_icon="🧪", layout="wide")

PRESETS = {
    "Momentum (top decile, weekly)": "Buy stocks whose 20-day momentum is in the top decile, equal weight, rebalance weekly.",
    "Mean reversion (below moving average)": "Mean reversion: each day buy the 5 stocks trading furthest below their 10-day moving average.",
    "Low volatility (monthly)": "Low-volatility anomaly: hold the 10 least volatile stocks over the last 60 days, rebalance monthly.",
}
RED, GREEN, AMBER, GREY = "#d64545", "#1f9d55", "#c27c0e", "#6b7280"

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600;
       margin:2px 4px 2px 0; border:1px solid transparent;}
.pill-fail {background:#fde8e8; color:#9b1c1c; border-color:#f8b4b4;}
.pill-warn {background:#fdf6e3; color:#8a5a00; border-color:#f3d48a;}
.pill-pass {background:#e6f6ec; color:#14532d; border-color:#a7dcb9;}
.model-tag {font-family: ui-monospace, monospace; font-size:0.78rem; color:#374151; background:#f3f4f6;
            padding:1px 6px; border-radius:4px;}
.subtle {color:#6b7280; font-size:0.9rem;}
.hero-sub {color:#4b5563; font-size:1.05rem; margin-top:-0.6rem;}
.steps {display:flex; gap:10px; margin:6px 0 14px 0; flex-wrap:wrap;}
.step {flex:1 1 200px; border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px; background:#fafafa;}
.step b {display:block; font-size:0.92rem; margin-bottom:2px;}
.step span {color:#4b5563; font-size:0.82rem;}
.codebox {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:0.8rem; line-height:1.45;
          border:1px solid #e5e7eb; border-radius:8px; overflow-x:auto; background:#fbfbfc; padding:6px 0;}
.codeline {white-space:pre; padding:0 10px;}
.codeline .ln {display:inline-block; width:2.6em; color:#9ca3af; user-select:none;}
.codeline.flag-dynamic {background:#d64545; color:white; font-weight:600;}
.codeline.flag-dynamic .ln {color:#fde8e8;}
.codeline.flag-dynamic .flagnote {color:white;}
.codeline.flag-critical {background:#fde8e8;}
.codeline.flag-high {background:#fdf1dc;}
.codeline.flag-medium {background:#fdf8e6;}
.flagnote {color:#9b1c1c; font-size:0.74rem; padding-left:1.2em;}
.verdict {border-radius:12px; padding:14px 18px; margin:6px 0 12px 0; border:1px solid #e5e7eb;}
.verdict .big {font-size:1.6rem; font-weight:700;}
.verdict .lab {color:#6b7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def synthetic(seed: int, n_tickers: int, years: int):
    return generate_panel(seed=seed, n_tickers=n_tickers, years=years)


@st.cache_data(show_spinner=False)
def real_panel():
    from data.real import load_real_panel
    return load_real_panel()


@st.cache_data(show_spinner=False)
def leak_zoo() -> dict:
    """LeakBench strategies as {label: code without the answer-key header}."""
    try:
        from bench.run_bench import load_strategies
    except Exception:
        return {}
    out = {}
    for sdef in load_strategies():
        mark = "✗ leak" if sdef["label"] == "leaky" else "✓ honest"
        name = sdef.get("name") or sdef.get("id")
        bug = sdef.get("bug") or ""
        out[f"{mark} · {name}" + (f" ({bug})" if bug and bug != "none" else "")] = sdef["stripped"]
    return out


def short_model(m: str) -> str:
    return m.split("/")[-1]


def pill(t: dict) -> str:
    if t["passed"]:
        cls, mark = "pill-pass", "✓"
    elif t.get("blocking"):
        cls, mark = "pill-fail", "✗"
    else:
        cls, mark = "pill-warn", "!"
    return f'<span class="pill {cls}" title="{html.escape(t["summary"])}">{mark} {t["name"]}</span>'


def flagged_code_html(code: str, tests: list[dict]) -> str | None:
    """Source with the lines the static scan flagged highlighted (None when nothing is flagged)."""
    st_test = next((t for t in tests if t["name"] == "static_scan"), None)
    findings = list((st_test or {}).get("evidence", {}).get("findings", []) if st_test else [])
    loc = localized_entry(tests)
    if loc:
        findings.append({"line": loc["line"], "rule": "LEAK ENTERS HERE (proved by perturbation)", "severity": "dynamic"})
    if not findings:
        return None
    by_line: dict[int, list[dict]] = {}
    for f in findings:
        by_line.setdefault(int(f.get("line", 0)), []).append(f)
    order = {"dynamic": -1, "critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    out = []
    for i, line in enumerate(code.splitlines(), start=1):
        fs = by_line.get(i)
        cls = ""
        note = ""
        if fs:
            worst = min((f.get("severity", "medium") for f in fs), key=lambda x: order.get(x, 9))
            cls = f" flag-{worst}"
            note = '<span class="flagnote">◀ ' + html.escape("; ".join(f"{f.get('rule')} ({f.get('severity')})" for f in fs)) + "</span>"
        out.append(f'<div class="codeline{cls}"><span class="ln">{i}</span>{html.escape(line) or " "}{note}</div>')
    return '<div class="codebox">' + "".join(out) + "</div>"


def localized_entry(tests: list[dict]) -> dict | None:
    fp = next((t for t in tests if t["name"] == "future_perturbation" and not t["passed"]), None)
    loc = ((fp or {}).get("evidence") or {}).get("localized") or {}
    e = loc.get("entry") if loc.get("found") else None
    return e if e and e.get("line") else None


def leak_proof_chart(test: dict, data: dict) -> go.Figure | None:
    """Each cut date t: all data after t was scrambled. Red bars = past signals that changed anyway."""
    cuts = (test.get("evidence") or {}).get("cuts") or []
    if not cuts:
        return None
    fig = go.Figure()
    idx = data["close"].index
    for k, c in enumerate(cuts):
        cut = pd.Timestamp(c["cut_date"])
        y = k + 1
        fig.add_trace(go.Scatter(x=[cut, idx[-1]], y=[y, y], mode="lines", line=dict(color="#d1d5db", width=8),
                                 hoverinfo="skip", showlegend=(k == 0), name="Scrambled future (after cut)"))
        if c.get("leaked"):
            a, b = pd.Timestamp(c["earliest_changed_date"]), pd.Timestamp(c["latest_changed_date"])
            fig.add_trace(go.Scatter(x=[a, b], y=[y, y], mode="lines", line=dict(color=RED, width=8),
                                     name="Past signals that changed", showlegend=not any(
                                         t.name == "Past signals that changed" for t in fig.data),
                                     hovertext=f"{c['rows_changed']} past rows changed, max weight change {c['max_abs_weight_change']:.3f}",
                                     hoverinfo="text"))
        fig.add_trace(go.Scatter(x=[cut], y=[y], mode="markers", marker=dict(symbol="line-ns", size=18,
                                 line=dict(width=3, color=RED if c.get("leaked") else GREEN)),
                                 showlegend=False, hovertext=f"cut {cut:%Y-%m-%d}: {'LEAK' if c.get('leaked') else 'ok'}",
                                 hoverinfo="text"))
    fig.update_layout(height=60 + 34 * len(cuts), margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", legend=dict(orientation="h", y=1.15, x=0),
                      yaxis=dict(tickvals=list(range(1, len(cuts) + 1)), ticktext=[f"cut {k}" for k in range(1, len(cuts) + 1)],
                                 showgrid=False))
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f3", range=[idx[0], idx[-1]])
    return fig


# ----------------------------------------------------------------------------- sidebar
cfg = load_config()
with st.sidebar:
    st.header("Settings")
    has_key = bool(os.environ.get("NEBIUS_API_KEY"))
    default_mock = use_mock()
    mode = st.radio("LLM backend", ["Mock (offline, reproducible)", "Nebius Token Factory (live)"],
                    index=1 if (has_key and os.environ.get("DEFAULT_LIVE") == "1") else 0,
                    help="Live mode needs NEBIUS_API_KEY in the environment.")
    live = mode.startswith("Nebius")
    if live and not has_key:
        st.error("NEBIUS_API_KEY is not set - falling back to Mock.")
        live = False
    live_cap = int(os.environ.get("MAX_LIVE_RUNS_PER_SESSION", "5"))
    live_used = st.session_state.get("live_runs", 0)
    if live:
        st.caption(f"Live runs this session: {live_used}/{live_cap} (public demo budget)")
        if live_used >= live_cap:
            st.warning("Live budget for this session is used up - switch to Mock or run locally with your own key.")
            live = False
    st.subheader("Model routing")
    for role in ("coder", "auditor", "summarizer"):
        cfg.routes[role].model = st.text_input(f"{role.capitalize()} model", cfg.routes[role].model,
                                               help=cfg.routes[role].purpose)
    st.caption(f"Endpoint: `{cfg.base_url}`")
    st.subheader("Loop")
    max_iters = st.slider("Max iterations", 1, 6, 4)
    cost_bps = st.slider("Transaction cost (bps per unit turnover)", 0, 50, 10)
    pacing = st.slider("Demo pacing (mock latency, s)", 0.0, 2.0, 0.6, 0.1, disabled=live)
    st.subheader("Data")
    src = st.radio("Universe", ["Synthetic (seeded)", "Real US equities (40, 2014–2025)", "Upload CSV"], index=0,
                   help="Synthetic has known delistings and IPOs (ground truth for survivorship). "
                        "Real = adjusted daily prices of 40 US names incl. 7 IPOs; it has no delisted names, "
                        "so the dataset itself is survivor-biased and the audit warns about it.")
    if src.startswith("Real"):
        data = real_panel()
    elif src.startswith("Synthetic"):
        c1, c2 = st.columns(2)
        seed = c1.number_input("Seed", value=51, step=1)
        years = c2.number_input("Years", value=6, min_value=2, max_value=15, step=1)
        n_tickers = st.slider("Tickers", 20, 120, 50, 5)
        data = synthetic(int(seed), int(n_tickers), int(years))
    else:
        up = st.file_uploader("Long CSV: date,ticker,close[,open,high,low,volume]", type="csv")
        data = load_long_csv(up) if up else synthetic(51, 50, 6)
        if not up:
            st.caption("No file yet - using the synthetic universe.")
    lt = listing_table(data)
    st.caption(f"{data['close'].shape[1]} tickers · {len(data['close'])} days · "
               f"{int(lt['delisted'].sum())} delisted · {int(lt['ipo_after_start'].sum())} IPOs")

# ----------------------------------------------------------------------------- header
st.title("Backtest Auditor")
st.markdown('<p class="hero-sub">LLM agents write the strategy. <b>Deterministic tests decide if it is honest.</b> '
            'Nemotron on Nebius Token Factory codes and explains; perturbation, survivorship and '
            'overfitting tests prove look-ahead bias with numbers.</p>', unsafe_allow_html=True)
st.markdown("""<div class="steps">
<div class="step"><b>1 · Write</b><span>Nemotron 3 Super turns your idea into <code>generate_signals(data)</code>, or you paste your own.</span></div>
<div class="step"><b>2 · Attack</b><span>Sandboxed run, then the future is scrambled, dead stocks revived, execution delayed, code scanned.</span></div>
<div class="step"><b>3 · Diagnose</b><span>Nemotron 3 Ultra reads the failing evidence and pins each bug to a line.</span></div>
<div class="step"><b>4 · Repeat</b><span>Loop until no critical/high test fails. The LLM proposes; the numbers decide.</span></div>
</div>""", unsafe_allow_html=True)

DATA_FOR_CHARTS = data
tab_audit, tab_bench, tab_ci = st.tabs(["🧪 Audit a strategy", "📊 LeakBench: tests vs LLM judges", "⚙️ Use it in CI"])
with tab_bench:
    try:
        from bench.ui import render_leakbench
        render_leakbench(st)
    except Exception as exc:  # bench results not generated yet
        st.info(f"LeakBench results are not available in this build ({type(exc).__name__}).")
with tab_ci:
    try:
        st.markdown(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "ci.md"), encoding="utf-8").read())
    except OSError:
        st.info("CI docs are not available in this build.")
with tab_audit:
    if "hyp" not in st.session_state:
        st.session_state["hyp"] = PRESETS["Momentum (top decile, weekly)"]
    c_in, c_side = st.columns([3, 1.25])
    with c_side:
        preset = st.selectbox("Preset hypotheses", list(PRESETS))
        if st.button("Use preset", width="stretch"):
            st.session_state["hyp"] = PRESETS[preset]
    with c_in:
        hyp = st.text_area("Trading hypothesis (natural language)", key="hyp", height=80)
    origin = st.radio("First draft", ["Nemotron writes it from the hypothesis", "Audit my own strategy code"],
                      horizontal=True, help="Paste a generate_signals(data) function from your notebook. "
                                            "Iteration 1 runs it unchanged; Nemotron only repairs it.")
    user_code = None
    if origin.startswith("Audit"):
        from agents.mock_strategies import LEAKY
        if "user_code" not in st.session_state:
            st.session_state["user_code"] = LEAKY["momentum"]
        zoo = leak_zoo()
        z1, z2 = st.columns([3, 1])
        pick = z1.selectbox("Load an example from LeakBench (or paste your own below)", ["—"] + list(zoo),
                            help="Labelled strategies from the benchmark. ✗ = has a leak, ✓ = honest. "
                                 "The subtle ones (rank default axis, resample('MS').last()) fooled Nemotron as a judge.")
        if z2.button("Load", width="stretch", disabled=(pick == "—")):
            st.session_state["user_code"] = zoo[pick]
            st.rerun()
        up_nb = st.file_uploader("…or upload a research notebook / script (.ipynb or .py)", type=["ipynb", "py"],
                                 help="Code cells are concatenated (magics skipped) and generate_signals is extracted "
                                      "with the helpers, constants and imports it uses.")
        if up_nb is not None and st.session_state.get("nb_loaded") != (up_nb.name, up_nb.size):
            from notebook import extract_strategy
            r = extract_strategy(up_nb.getvalue(), up_nb.name)
            st.session_state["nb_loaded"] = (up_nb.name, up_nb.size)
            if r.get("ok"):
                st.session_state["user_code"] = r["code"]
                st.session_state["nb_info"] = (f"Extracted from **{up_nb.name}**: " +
                                               (f"{r['cell_count']} cells, `generate_signals` in cell {r['entry_cell']}; "
                                                if r.get("is_notebook") else "") +
                                               "kept " + ", ".join(f"`{k}`" for k in r.get("kept", [])) +
                                               (f"; skipped {len(r['skipped'])} magic/invalid line(s)" if r.get("skipped") else ""))
                st.rerun()
            else:
                st.error(r.get("error") or "Could not extract generate_signals from this file.")
        if st.session_state.get("nb_info"):
            st.caption(st.session_state["nb_info"])
        user_code = st.text_area("Strategy code: def generate_signals(data) -> weights DataFrame",
                                 key="user_code", height=260)
    run = st.button("Run audit loop", type="primary")
    if not live:
        st.caption("Mock mode replays canned Coder drafts (picked by keyword) and a rule-based Auditor; "
                   "every test result and metric is computed for real.")


# ----------------------------------------------------------------------------- rendering
def render_iteration(it: dict, models: dict, expanded: bool):
    tests = it.get("tests", [])
    blocking = [t for t in tests if t.get("blocking")]
    if it["status"] == "clean":
        badge = f'<span class="pill pill-pass">CLEAN · no blocking failures</span>'
    elif it["status"] == "error":
        badge = f'<span class="pill pill-fail">SANDBOX ERROR</span>'
    elif it["status"] == "pending":
        badge = f'<span class="pill pill-warn">running tests…</span>'
    else:
        badge = f'<span class="pill pill-fail">{len(blocking)} blocking failure(s)</span>'
    label = "Draft" if it["index"] == 0 else "Fix"
    author = "you (pasted)" if it.get("user_code") else short_model(models["coder"])
    with st.container(border=True):
        st.markdown(f"**Iteration {it['index'] + 1} · {label}** &nbsp; {badge} &nbsp; "
                    f'<span class="subtle">code by</span> <span class="model-tag">{author}</span>',
                    unsafe_allow_html=True)
        if tests:
            st.markdown(" ".join(pill(t) for t in tests), unsafe_allow_html=True)
        m = it.get("metrics") or {}
        if m:
            a, b, c, d = st.columns(4)
            a.metric("Sharpe", f"{m['sharpe']:.2f}")
            b.metric("CAGR", f"{m['cagr']:.1%}")
            c.metric("Max drawdown", f"{m['max_drawdown']:.1%}")
            d.metric("Turnover / yr", f"{m['turnover_annual']:.1f}x")
        if it.get("error"):
            st.error(it["error"])
        flagged = flagged_code_html(it["code"], tests) if blocking else None
        with st.expander("Strategy code" + (" · flagged lines highlighted" if flagged else ""), expanded=False):
            if flagged:
                st.markdown(flagged, unsafe_allow_html=True)
            else:
                st.code(it["code"], language="python")
        if blocking:
            with st.expander("Failing test evidence", expanded=expanded):
                loc = localized_entry(tests)
                if loc:
                    st.markdown(f"🎯 **Leak enters at line {loc['line']}** (found by experiment: the first value that changes "
                                f"when only the future changes): `{loc.get('source_line', '').strip()}`")
                for t in blocking:
                    st.markdown(f"**{t['name']}** ({t['severity']}): {t['summary']}")
                pert = next((t for t in blocking if t["name"] == "future_perturbation"), None)
                if pert is not None and DATA_FOR_CHARTS is not None:
                    fig = leak_proof_chart(pert, DATA_FOR_CHARTS)
                    if fig is not None:
                        st.caption("Leak proof: for each cut date, every price after it was scrambled. "
                                   "Red = signals *before* the cut changed anyway, so the strategy reads the future.")
                        st.plotly_chart(fig, width="stretch", key=f"lp_{id(it)}_{it['index']}_{np.random.randint(1e9)}")
        if it.get("auditor_feedback"):
            with st.expander(f"Auditor diagnosis · {short_model(models['auditor'])}", expanded=expanded):
                st.markdown(it["auditor_feedback"])


def equity_chart(trace: dict, data: dict) -> go.Figure:
    its = [i for i in trace["iterations"] if i.get("equity")]
    fig = go.Figure()
    close = data["close"]
    ew = close.notna().astype(float)
    ew = ew.div(ew.sum(axis=1), axis=0)
    bench = run_backtest(ew, data).equity
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="Equal-weight universe", line=dict(color=GREY, width=1.3, dash="dot")))
    if its:
        first, last = its[0], its[-1]
        fig.add_trace(go.Scatter(x=first["dates"], y=first["equity"], name=f"First draft (Sharpe {first['metrics']['sharpe']:.2f})",
                                 line=dict(color=RED, width=2.2)))
        if last is not first:
            name = "Audited final" if last["status"] == "clean" else "Last iteration"
            fig.add_trace(go.Scatter(x=last["dates"], y=last["equity"], name=f"{name} (Sharpe {last['metrics']['sharpe']:.2f})",
                                     line=dict(color=GREEN, width=2.2)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log",
                      yaxis_title="Growth of $1 (log)", legend=dict(orientation="h", y=1.08, x=0),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f3")
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f3")
    return fig


def tests_table(tests: list[dict]) -> pd.DataFrame:
    rows = [{"Result": "PASS" if t["passed"] else ("FAIL" if t["blocking"] else "WARN"),
             "Test": t["name"], "Severity": t["severity"], "Finding": t["summary"],
             "Suggested fix": t.get("suggested_fix", "")} for t in tests]
    return pd.DataFrame(rows)


def style_result(v):
    return {"PASS": f"color:{GREEN};font-weight:600", "FAIL": f"color:{RED};font-weight:600",
            "WARN": f"color:{AMBER};font-weight:600"}.get(v, "")


def render_results(trace: dict, data: dict):
    its = trace["iterations"]
    status = trace["final_status"]
    if status == "clean":
        st.success(f"Clean after {len(its)} iteration(s): no critical or high-severity test fails. "
                   "Clean means honest, not necessarily profitable.")
    else:
        st.error(f"Not clean after {len(its)} iteration(s) ({status}). Do not trust these results.")

    scored0 = [i for i in its if i.get("metrics")]
    if len(scored0) >= 2 or (scored0 and status != "clean"):
        f0, fl = scored0[0], scored0[-1]
        pert0 = next((t for t in f0.get("tests", []) if t["name"] == "future_perturbation" and not t["passed"]), None)
        leak_d = (pert0 or {}).get("evidence", {}).get("first_leaking_date")
        n_fail0 = sum(1 for t in f0.get("tests", []) if t.get("blocking"))
        cols = st.columns(4)
        cols[0].markdown(f'<div class="verdict"><div class="lab">First draft Sharpe</div><div class="big" style="color:{RED}">'
                         f'{f0["metrics"]["sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div class="verdict"><div class="lab">{"Audited" if status == "clean" else "Last"} Sharpe</div>'
                         f'<div class="big" style="color:{GREEN if status == "clean" else AMBER}">{fl["metrics"]["sharpe"]:.2f}</div></div>',
                         unsafe_allow_html=True)
        cols[2].markdown(f'<div class="verdict"><div class="lab">Blocking failures in draft</div><div class="big">{n_fail0}</div></div>',
                         unsafe_allow_html=True)
        cols[3].markdown(f'<div class="verdict"><div class="lab">First leaking date</div><div class="big">'
                         f'{str(leak_d)[:10] if leak_d else "–"}</div></div>', unsafe_allow_html=True)

    st.subheader("Before vs after the audit")
    st.plotly_chart(equity_chart(trace, data), width="stretch")
    scored = [i for i in its if i.get("metrics")]
    if len(scored) >= 1:
        rows = []
        for i in scored:
            m = i["metrics"]
            rows.append({"Iteration": i["index"] + 1, "Status": i["status"], "Sharpe": m["sharpe"],
                         "Sharpe @ t+2": i["metrics_lag2"].get("sharpe"), "CAGR": m["cagr"],
                         "Max DD": m["max_drawdown"], "Hit rate": m["hit_rate"], "Turnover/yr": m["turnover_annual"]})
        st.dataframe(pd.DataFrame(rows).style.format({"Sharpe": "{:.2f}", "Sharpe @ t+2": "{:.2f}", "CAGR": "{:.1%}",
                                                      "Max DD": "{:.1%}", "Hit rate": "{:.1%}", "Turnover/yr": "{:.1f}"}),
                     hide_index=True, width="stretch")

    tab_tl, tab_rep, tab_diff, tab_use = st.tabs(["Iteration timeline", "Audit report", "Code diff", "Models & tokens"])
    with tab_tl:
        for it in its:
            render_iteration(it, trace["models"], expanded=False)
    with tab_rep:
        if trace.get("summary"):
            st.markdown(f"**Executive summary** · <span class='model-tag'>{short_model(trace['models']['summarizer'])}</span>",
                        unsafe_allow_html=True)
            st.markdown(trace["summary"])
        final = scored[-1] if scored else None
        if final:
            st.markdown(f"**Deterministic tests on iteration {final['index'] + 1}**")
            st.dataframe(tests_table(final["tests"]).style.map(style_result, subset=["Result"]),
                         hide_index=True, width="stretch")
            with st.expander("Raw evidence (JSON)"):
                st.json(final["tests"], expanded=False)
        st.download_button("Download full trace (JSON)", RunTrace_json(trace), "backtest_auditor_trace.json",
                           "application/json")
    with tab_diff:
        if len(its) >= 2:
            diff = difflib.unified_diff(its[0]["code"].splitlines(), its[-1]["code"].splitlines(),
                                        "draft.py", "final.py", lineterm="")
            st.code("\n".join(diff), language="diff")
        else:
            st.info("Only one iteration - nothing to diff.")
    with tab_use:
        calls = pd.DataFrame(trace["calls"])
        if not calls.empty:
            if trace.get("mock"):
                st.caption("Mock run: model IDs show the configured routing; token counts are estimates (chars/4).")
            show = calls[["iteration", "step", "role", "model", "prompt_tokens", "completion_tokens", "latency_s", "cost_usd"]].copy()
            show["iteration"] = show["iteration"].map(lambda x: "-" if pd.isna(x) else str(int(x) + 1))
            st.dataframe(show.rename(columns={"prompt_tokens": "prompt tok", "completion_tokens": "completion tok",
                                              "latency_s": "latency (s)", "cost_usd": "est. cost ($)"}),
                         hide_index=True, width="stretch")
            tot = calls.groupby("model").agg(calls=("step", "count"), prompt=("prompt_tokens", "sum"),
                                             completion=("completion_tokens", "sum"), cost=("cost_usd", "sum"))
            st.markdown("**Totals by model**")
            st.dataframe(tot.style.format({"cost": "${:.4f}"}), width="stretch")


def RunTrace_json(trace: dict) -> str:  # noqa: N802
    import json
    return json.dumps(trace, default=str, indent=1)


# ----------------------------------------------------------------------------- run
with tab_audit:
    if run and hyp.strip():
        if live:
            st.session_state["live_runs"] = st.session_state.get("live_runs", 0) + 1
        llm = make_llm(mock=not live, config=cfg, mock_latency_s=pacing)
        st.subheader("Live iteration timeline")
        status = st.status("Coder is drafting the strategy…", expanded=True)
        live_box = st.empty()
        state: dict = {"iterations": []}

        def redraw():
            with live_box.container():
                for it in state["iterations"]:
                    render_iteration(it, {r: cfg.route(r).model for r in ("coder", "auditor", "summarizer")}, expanded=True)

        def on_event(e):
            t = e["type"]
            if t == "coder_start":
                status.update(label=f"Iteration {e['iteration'] + 1}: Coder ({short_model(e['model'])}) is writing code…")
            elif t == "coder_done":
                state["iterations"].append({"index": e["iteration"], "code": e["code"], "status": "pending", "tests": [],
                                            "user_code": bool(e.get("user_code"))})
                status.update(label=f"Iteration {e['iteration'] + 1}: running sandbox + deterministic audit…")
            elif t == "tests_done":
                it = state["iterations"][-1]
                it.update(status="clean" if e["clean"] else "failing", tests=e["tests"], metrics=e["metrics"])
            elif t == "error":
                state["iterations"][-1].update(status="error", error=e["error"])
            elif t == "auditor_start":
                status.update(label=f"Iteration {e['iteration'] + 1}: Auditor ({short_model(e['model'])}) is diagnosing…")
            elif t == "auditor_done":
                state["iterations"][-1]["auditor_feedback"] = e["feedback"]
            elif t == "summary_start":
                status.update(label=f"Summarizer ({short_model(e['model'])}) is writing the report…")
            redraw()

        try:
            trace: RunTrace = run_audit_loop(hyp.strip(), data, llm, max_iters=max_iters, cost_bps=float(cost_bps),
                                             on_event=on_event, initial_code=(user_code or None))
            status.update(label=f"Done: {trace.final_status.replace('_', ' ')}",
                          state="complete" if trace.final_status == "clean" else "error", expanded=False)
            st.session_state["trace"] = trace.to_dict()
        except Exception as exc:  # surface API/config errors in the UI
            status.update(label="Run failed", state="error")
            st.exception(exc)
        live_box.empty()

    if "trace" in st.session_state:
        render_results(st.session_state["trace"], data)
    else:
        st.info("Pick a preset or type a hypothesis, then press **Run audit loop**.")

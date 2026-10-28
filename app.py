"""Backtest Auditor - Streamlit UI.

Run offline:  USE_MOCK=1 streamlit run app.py
Run live:     NEBIUS_API_KEY=... streamlit run app.py
"""
from __future__ import annotations

import difflib
import html
import os

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
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def synthetic(seed: int, n_tickers: int, years: int):
    return generate_panel(seed=seed, n_tickers=n_tickers, years=years)


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


# ----------------------------------------------------------------------------- sidebar
cfg = load_config()
with st.sidebar:
    st.header("Settings")
    has_key = bool(os.environ.get("NEBIUS_API_KEY"))
    default_mock = use_mock()
    mode = st.radio("LLM backend", ["Mock (offline, reproducible)", "Nebius Token Factory (live)"],
                    index=0 if default_mock or not has_key else 1,
                    help="Live mode needs NEBIUS_API_KEY in the environment.")
    live = mode.startswith("Nebius")
    if live and not has_key:
        st.error("NEBIUS_API_KEY is not set - falling back to Mock.")
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
    src = st.radio("Universe", ["Synthetic (seeded)", "Upload CSV"], index=0)
    if src.startswith("Synthetic"):
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

if "hyp" not in st.session_state:
    st.session_state["hyp"] = PRESETS["Momentum (top decile, weekly)"]
c_in, c_side = st.columns([3, 1.25])
with c_side:
    preset = st.selectbox("Preset hypotheses", list(PRESETS))
    if st.button("Use preset", width="stretch"):
        st.session_state["hyp"] = PRESETS[preset]
with c_in:
    hyp = st.text_area("Trading hypothesis (natural language)", key="hyp", height=80)
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
    with st.container(border=True):
        st.markdown(f"**Iteration {it['index'] + 1} · {label}** &nbsp; {badge} &nbsp; "
                    f'<span class="subtle">code by</span> <span class="model-tag">{short_model(models["coder"])}</span>',
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
        with st.expander("Strategy code", expanded=False):
            st.code(it["code"], language="python")
        if blocking:
            with st.expander("Failing test evidence", expanded=expanded):
                for t in blocking:
                    st.markdown(f"**{t['name']}** ({t['severity']}): {t['summary']}")
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
if run and hyp.strip():
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
            state["iterations"].append({"index": e["iteration"], "code": e["code"], "status": "pending", "tests": []})
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
                                         on_event=on_event)
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

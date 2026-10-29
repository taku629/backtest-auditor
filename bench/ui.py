"""Streamlit view of LeakBench results (bench/results.json).

Usage inside a Streamlit app::

    import streamlit as st
    from bench.ui import render_leakbench
    render_leakbench(st)
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
SHORT = {"deterministic": "Deterministic audit"}


def load_results(path: str | Path | None = None) -> dict | None:
    p = Path(path) if path else RESULTS_JSON
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _short(res: dict, m: str) -> str:
    lab = res.get("method_labels", {}).get(m, m)
    return lab.replace("LLM-only: ", "LLM judge: ")


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def summary_frame(res: dict):
    import pandas as pd

    rows = []
    for m in res["methods"]:
        s = res["metrics"][m]
        rows.append({"Method": _short(res, m), "Recall": s["recall"], "False-positive rate": s["false_positive_rate"],
                     "Precision": s["precision"], "Accuracy": s["accuracy"],
                     "Missed leaks": s["fn"], "False alarms": s["fp"]})
    return pd.DataFrame(rows)


def strategy_frame(res: dict):
    import pandas as pd

    rows = []
    for r in res["strategies"]:
        row = {"Strategy": r["id"], "Label": r["label"], "Bug type": r["bug"]}
        for m in res["methods"]:
            p = r["pred"].get(m)
            if p is None:
                row[_short(res, m)] = "–"
            else:
                row[_short(res, m)] = ("✅ " if bool(p) == r["leaky"] else "❌ ") + ("flag" if p else "pass")
        row["Blocking tests"] = ", ".join(r["deterministic"].get("blocking", []))
        rows.append(row)
    return pd.DataFrame(rows)


def bar_chart(res: dict):
    import plotly.graph_objects as go

    names = [_short(res, m) for m in res["methods"]]
    rec = [100 * (res["metrics"][m]["recall"] or 0) for m in res["methods"]]
    fpr = [100 * (res["metrics"][m]["false_positive_rate"] or 0) for m in res["methods"]]
    fig = go.Figure()
    fig.add_bar(name="Recall (leaks caught)", x=names, y=rec, marker_color="#2e7d32",
                text=[f"{v:.0f}%" for v in rec], textposition="outside")
    fig.add_bar(name="False-positive rate", x=names, y=fpr, marker_color="#c62828",
                text=[f"{v:.0f}%" for v in fpr], textposition="outside")
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(title="%", range=[0, 115]), legend=dict(orientation="h", y=1.12),
                      plot_bgcolor="white")
    return fig


def render_leakbench(st, path: str | Path | None = None) -> None:
    res = load_results(path)
    st.subheader("LeakBench: deterministic tests vs an LLM-only judge")
    if res is None:
        st.info("No LeakBench results yet. Run `python -m bench.run_bench` (add `--no-llm` to stay offline).")
        return
    st.caption(f"{res['n_strategies']} labelled strategies ({res['n_leaky']} with look-ahead or survivorship bugs, "
               f"{res['n_clean']} clean, several of them deliberately suspicious-looking). "
               "The LLM judge reads the code and answers leaky / clean; the deterministic audit runs it. "
               f"Generated {res['generated_at']}.")

    methods = res["methods"]
    cols = st.columns(len(methods))
    det = res["metrics"]["deterministic"]
    for c, m in zip(cols, methods):
        s = res["metrics"][m]
        delta = None
        if m != "deterministic" and s["recall"] is not None and det["recall"] is not None:
            delta = f"{100 * (s['recall'] - det['recall']):+.0f} pts recall vs deterministic"
        c.metric(_short(res, m), f"{_pct(s['recall'])} caught", delta=delta, delta_color="normal")
        c.caption(f"False alarms: {s['fp']}/{s['fp'] + s['tn']} clean ({_pct(s['false_positive_rate'])}) · "
                  f"missed {s['fn']}/{s['tp'] + s['fn']} leaks")

    st.plotly_chart(bar_chart(res), width="stretch")

    summ = summary_frame(res)
    st.dataframe(summ.style.format({k: lambda v: _pct(v) for k in
                                    ("Recall", "False-positive rate", "Precision", "Accuracy")}),
                 width="stretch", hide_index=True)

    missed = {m: [r["id"] for r in res["strategies"] if r["leaky"] and not r["pred"].get(m)] for m in methods[1:]}
    if any(missed.values()):
        lines = [f"**{_short(res, m)}** missed: " + ", ".join(f"`{x}`" for x in ids)
                 for m, ids in missed.items() if ids]
        st.markdown("\n\n".join(lines))

    with st.expander("Per-strategy verdicts", expanded=False):
        st.dataframe(strategy_frame(res), width="stretch", hide_index=True)
    if res.get("llm_cost_usd") is not None:
        st.caption(f"LLM judge cost: ${res['llm_cost_usd']:.3f} for {len(methods) - 1} model(s) × "
                   f"{res['n_strategies']} strategies. Deterministic audit: $0, about 1 s per strategy.")
    for n in res.get("notes", [])[:2]:
        st.caption("• " + n)

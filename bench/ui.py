"""Streamlit view of LeakBench results (bench/results.json).

Usage inside a Streamlit app::

    import streamlit as st
    from bench.ui import render_leakbench
    render_leakbench(st)
"""
from __future__ import annotations

import json
from pathlib import Path

from audit.mdsafe import escape_dollars

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
        st.caption(f"LLM judge cost: \\${res['llm_cost_usd']:.3f} for {len(methods) - 1} model(s) × "
                   f"{res['n_strategies']} strategies. Deterministic audit: \\$0, about 1 s per strategy.")
    for n in res.get("notes", [])[:2]:
        st.caption("• " + escape_dollars(n))


# ----------------------------------------------------------------------------
# In-the-wild study (bench/run_wild.py): read the slim cached JSON only, never call an LLM
# ----------------------------------------------------------------------------
WILD_PUBLIC = Path(__file__).resolve().parent / "wild_public.json"
_OUTCOME_ICON = {"leaky": "🔴 leaky", "clean": "🟢 clean", "crashed": "⚪ crashed", "never trades": "⚪ never trades"}


def load_wild(path: str | Path | None = None) -> dict | None:
    p = Path(path) if path else WILD_PUBLIC
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d if d.get("summary") else None


def wild_frame(w: dict, only_leaky: bool = True):
    import pandas as pd

    rows = [r for r in w["rows"] if r["outcome"] == "leaky" or not only_leaky]
    return pd.DataFrame([{
        "Draft": r["key"], "Request": r["prompt"], "Outcome": _OUTCOME_ICON.get(r["outcome"], r["outcome"]),
        "Failing tests": ", ".join(r["blocking"]),
        "Leak line": (f"L{r['leak_line']}: {(r.get('leak_source') or '').strip()}" if r.get("leak_line") else ""),
        "Sharpe": r["sharpe"], "Quick fix": r.get("quickfix") or "",
        "Nemotron loop": r.get("loop") or "",
        "Sharpe after repair": r.get("sharpe_after") if r.get("quickfix") == "fixed" else r.get("sharpe_after_loop"),
    } for r in rows])


def _cond_name(k: str) -> str:
    m, s = k.split("/")
    return f"{'Super' if m == 'super' else 'Lightning'} · {'app prompt' if s == 'app' else 'chat prompt'}"


def render_wild(st, path: str | Path | None = None) -> None:
    w = load_wild(path)
    st.subheader("In the wild: do Nemotron's own backtests leak?")
    if w is None:
        st.info("No in-the-wild results in this build. Run `python -m bench.run_wild`.")
        return
    a = w["summary"]["all"]
    st.caption("30 ordinary strategy requests (momentum, mean reversion, pairs, vol targeting, ML signals, rotation…), "
               "written by us, none asking for a bug. Nemotron Super and Lightning each wrote a first draft under the "
               "app's Coder prompt and under a plain chat-style prompt; every draft went through the same deterministic "
               "audit, localizer and Quick fix as above. Cached results, no live calls. "
               f"Generated {str(w.get('generated_at', ''))[:10]}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runnable drafts that leak", f"{a['leaky']}/{a['ran']}",
              f"{a['leaky'] / a['ran']:.0%}" if a["ran"] else None, delta_color="off")
    c2.metric("Look-ahead drafts, leak line reported", f"{a['look_ahead_localized']}/{a['look_ahead']}")
    c3.metric("Repaired", f"{a.get('system_repaired', a['quickfix_fixed'])}/{a['leaky']}")
    c3.caption(f"Quick fix (no LLM, \\$0): {a['quickfix_fixed']} · Nemotron loop: {a.get('loop_repaired', 0)}")
    c4.metric("Median Sharpe of leaky drafts", f"{a.get('median_sharpe_leaky_all')}")
    c4.caption(f"clean drafts: {a['median_sharpe_clean']} — leaks did not look too good to be true")
    conds = {k: v for k, v in w["summary"].items() if "/" in k}
    if conds:
        import pandas as pd
        st.dataframe(pd.DataFrame([{
            "Condition": _cond_name(k), "Drafts": g["drafted"], "Ran": g["ran"], "Crashed": g["crashed"],
            "Leaky": f"{g['leaky']}/{g['ran']}", "Look-ahead": g["look_ahead"],
            "Survivor universe": g["by_class"].get("survivor_universe", 0),
            "Weight before listing": g["by_class"].get("pre_listing_weight", 0), "Quick fix": f"{g['quickfix_fixed']}/{g['leaky']}",
            "+ Nemotron loop": f"{g.get('loop_repaired', 0)}/{g.get('loop_tried', 0)}",
        } for k, g in conds.items()]), width="stretch", hide_index=True)
    st.dataframe(wild_frame(w), width="stretch", hide_index=True)
    nb, pl = w["summary"].get("style:notebook") or {}, w["summary"].get("style:plain") or {}
    style = (f": {nb['leaky']}/{nb['ran']} of those drafts leaked vs {pl['leaky']}/{pl['ran']} of plain ones"
             if nb.get("ran") and pl.get("ran") else "")
    st.caption(escape_dollars(
        f"Caveats: small sample (one draft per prompt and condition), prompts written by us, a third of them mention a "
        f"notebook habit (fill gaps, drop missing tickers, z-score, pick the best parameters){style}; verdicts are on "
        f"the bundled synthetic panel; Quick fix rules for two patterns (weights on unlisted tickers, a shift on a "
        f"ticker-indexed result) were changed after seeing these drafts. "
        f"Live spend for the study: ${w.get('spend_usd') or 0:.2f}. Details: bench/WILD.md."))

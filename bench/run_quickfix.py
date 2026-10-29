"""Quick fix benchmark: deterministic, LLM-free repair on LeakBench. Writes ``bench/QUICKFIX.md`` and
``bench/quickfix_results.json``.

Usage::

    python -m bench.run_quickfix                 # 24 leaky + 15 clean + held-out variants
    python -m bench.run_quickfix --only L19 C0   # subset (results are merged into the JSON)

For every strategy (LeakBench header stripped, as in the repair benchmark) the full audit runs with the dynamic
localizer, then ``audit.quickfix.quick_fix`` tries its candidate patches. **Fixed** = a candidate passes every
blocking deterministic test and is not over-corrected (``non_trivial`` passes, mean gross exposure >= 5%: the same
rule as ``bench/run_repair.py``). Clean strategies must get no proposed change. No LLM is called; cost is $0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(__file__).resolve().parent
OUT_MD = BENCH_DIR / "QUICKFIX.md"
OUT_JSON = BENCH_DIR / "quickfix_results.json"
REPAIR_JSON = BENCH_DIR / "repair_results.json"
HOLDOUT_DIR = BENCH_DIR / "quickfix_holdout"


def load_holdout() -> list[dict]:
    from bench.run_bench import load_strategies
    return [s | {"holdout": True} for s in load_strategies(HOLDOUT_DIR)] if HOLDOUT_DIR.exists() else []


def run_one(s: dict, data: dict) -> dict:
    from audit.leak_tests import run_audit
    from audit.quickfix import quick_fix
    from sandbox.runner import SandboxError, SandboxExecutor

    code = s["stripped"]
    t0 = time.time()
    row = {"id": s["id"], "label": s["label"], "bug": s.get("bug", ""), "holdout": bool(s.get("holdout"))}
    try:
        rep = run_audit(data, SandboxExecutor(code), code)
    except (SandboxError, RuntimeError, ValueError) as e:
        return row | {"status": "audit_error", "error": str(e)[:300], "seconds": round(time.time() - t0, 1)}
    fp = next(t for t in rep.tests if t.name == "future_perturbation")
    loc = (fp.evidence.get("localized") or {}) if not fp.passed else {}
    qf = quick_fix(code, data, report=rep, name=f"{s['id']}.py")
    return row | {
        "status": qf["status"], "fixed": qf["status"] == "fixed", "changed": qf["status"] == "fixed",
        "rule": qf["rule"], "description": qf["description"], "patched_lines": qf["lines"],
        "localized_line": (loc.get("entry") or {}).get("line"), "n_candidates": qf["n_candidates"],
        "n_tried": len(qf["attempts"]),
        "attempts": [{k: a.get(k) for k in ("rule", "description", "result", "blocking", "sharpe", "avg_gross",
                                            "error")} for a in qf["attempts"]],
        "before": qf["before"], "after": qf["after"], "diff": qf["diff"],
        "seconds": round(time.time() - t0, 1), "quickfix_seconds": qf.get("seconds"), "cost_usd": 0.0}


def llm_repair_by_strategy(path: Path = REPAIR_JSON) -> dict:
    """Condition A of the live LLM repair benchmark: {id: (repaired_non_trivial, n_runs, cost)}."""
    if not path.exists():
        return {}
    out: dict = {}
    for r in json.loads(path.read_text())["runs"]:
        if r.get("condition") != "A" or r.get("status") != "done":
            continue
        ok, n, c, s = out.get(r["id"], (0, 0, 0.0, []))
        out[r["id"]] = (ok + bool(r.get("repaired_non_trivial")), n + 1, c + (r.get("cost_usd") or 0.0),
                        s + [r.get("seconds")])
    return out


def write_md(res: dict, path: Path = OUT_MD) -> None:
    rows = res["rows"]
    leaky = [r for r in rows if r["label"] == "leaky" and not r["holdout"]]
    clean = [r for r in rows if r["label"] == "clean" and not r["holdout"]]
    hold = [r for r in rows if r["holdout"]]
    fixed = [r for r in leaky if r.get("fixed")]
    clean_changed = [r for r in clean if r.get("changed") or r.get("status") not in ("not_needed",)]
    llm = llm_repair_by_strategy()
    llm_ok = sum(v[0] for v in llm.values())
    llm_n = sum(v[1] for v in llm.values())
    llm_cost = sum(v[2] for v in llm.values())
    llm_secs = [x for v in llm.values() for x in v[3] if x is not None] or [0.0]
    secs = [r["seconds"] for r in leaky if r.get("seconds") is not None]
    both = [r for r in leaky if r.get("fixed")]
    L = ["# Quick fix: deterministic, LLM-free repair on LeakBench", "",
         f"Generated {res['generated_at']} · no LLM calls · data: synthetic panel "
         f"{res['data']['n_tickers']} tickers × {res['data']['n_dates']} days · reproduce with "
         "`python -m bench.run_quickfix` (offline, $0).", "",
         "After the audit fails, `audit/quickfix.py` takes the line the dynamic localizer pinned (plus the lines the "
         "static scan flags), generates candidate patches from a fixed list of general pandas rewrite rules "
         "(negative shift → positive, centred → trailing window, bfill → ffill, rank → rank(axis=1) or expanding "
         "rank, full-sample statistic → expanding, start-labelled resample → end-labelled, transform('last') → "
         "'first', drop a time reversal, pooled quantile → per date, dropna(axis=1) → keep all tickers, a universe "
         "chosen from end-of-sample statistics (`iloc[-1]`, `iloc[-k:]`/`tail(k)` reductions, full-sample `count()`) "
         "→ the same statistic as known on each date with its `nlargest`/`.loc[:, mask]`/`isin` consumers turned "
         "into date-wise masks, `np.polyfit(x, y, 1)` over the whole series → expanding least squares, and a "
         "last-resort `.shift(1)` of the localized value), and re-runs the full deterministic audit on each. "
         "**Fixed** = the first candidate that passes every blocking test *and* is not over-corrected "
         "(`non_trivial` passes and mean gross exposure ≥ 5%, the repair benchmark's rule).", "",
         "## Headline", "",
         "| Method | Repaired (leaky) | Changes on clean code | Cost per repair | Time per strategy (incl. audits) |",
         "|---|---|---|---|---|",
         f"| **Quick fix (deterministic, no LLM)** | **{len(fixed)}/{len(leaky)} strategies "
         f"({len(fixed) / max(1, len(leaky)):.0%})** | {len(clean_changed)}/{len(clean)} | **$0** | "
         f"{_median(secs):.0f} s median (max {max(secs) if secs else 0:.0f} s) |"]
    if llm:
        L.append(f"| LLM loop (Nemotron Super + Ultra, `bench/REPAIR.md`, condition A) | {llm_ok}/{llm_n} runs "
                 f"({llm_ok / max(1, llm_n):.0%}) | not measured | ${llm_cost / max(1, llm_ok):.4f} | "
                 f"{_median(llm_secs):.0f} s median per run (max {max(llm_secs):.0f} s) |")
    L += ["", "Deterministic: rerunning gives the same patch (no sampling), so one run per strategy is the whole "
          "result, while the LLM numbers are over 3 sampled runs per strategy. Sharpe values below are from this "
          f"run (pandas {res.get('pandas', '?')}); a few leaky drafts score differently from `bench/REPAIR.md`, "
          "which was generated in another environment.", ""]
    if llm and len(fixed) == len(leaky):
        L += [f"Combined (Quick fix first, LLM only when it fails): all {len(leaky)} leaky strategies are repaired by "
              "the Quick fix for $0, so on LeakBench no LLM call is needed for the repair; the LLM Coder stays as the "
              "fallback for leaks outside the rule list (and writes the strategy from a hypothesis in the first place).",
              "", "The last five rules to land (L06, L13, L14, L15, L16: point-in-time selection and expanding fit) were "
              "written on 2026-10-30 *after* seeing exactly these five fail, so the 24/24 is an in-sample number; "
              "see the held-out section and the notes.", ""]
    elif llm:
        rescued = [r for r in leaky if not r.get("fixed")]
        L += [f"Combined (Quick fix first, LLM only when it fails): {len(fixed)}/{len(leaky)} repaired for $0; the "
              f"remaining {len(rescued)} go to the LLM Coder, which repaired them in "
              f"{sum(llm.get(r['id'], (0, 0, 0, []))[0] for r in rescued)}/"
              f"{sum(llm.get(r['id'], (0, 0, 0, []))[1] for r in rescued)} live runs. Only {len(rescued)} of {len(leaky)} "
              "strategies would need any LLM call.", ""]
    L += ["## Per strategy", "",
          "| Strategy | Bug | Result | Rule | Patched line(s) | Localized line | Candidates tried | Sharpe before → after | LLM loop (A) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in leaky + clean:
        b, a = r.get("before") or {}, r.get("after") or {}
        res_s = {"fixed": "✅ fixed", "not_needed": "— nothing to fix", "no_candidate": "❌ no rule applies",
                 "failed": "❌ no candidate passed"}.get(r["status"], r["status"])
        sh = (f"{b.get('sharpe', 0):.2f} → {a['sharpe']:.2f}" if a else
              (f"{b.get('sharpe', 0):.2f}" if b else ""))
        lr = llm.get(r["id"])
        L.append(f"| `{r['id']}` | {r.get('bug') or '-'} | {res_s} | {r.get('rule') or ''} | "
                 f"{', '.join(map(str, r.get('patched_lines') or [])) or ''} | {r.get('localized_line') or ''} | "
                 f"{r.get('n_tried', 0)}/{r.get('n_candidates', 0)} | {sh} | "
                 f"{f'{lr[0]}/{lr[1]}' if lr else ''} |")
    misses = [r for r in leaky if not r.get("fixed")]
    if misses:
        L += ["", "## Not fixed (fall back to the LLM Coder)", ""]
        for r in misses:
            tried = "; ".join(f"{a['description']} → {a['result']}" + (f" ({', '.join(a['blocking'])})"
                                                                         if a.get('blocking') else "")
                              for a in r.get("attempts", [])) or "no candidate generated"
            L.append(f"- **{r['id']}** ({r.get('bug')}): {tried}.")
        L += [""]
    if hold:
        hf = [r for r in hold if r.get("fixed") and r["label"] == "leaky"]
        hl = [r for r in hold if r["label"] == "leaky"]
        L += ["## Held-out variants", "",
              f"`bench/quickfix_holdout/`: {len(hl)} extra leaky strategies written *after* the rules, using the "
              "same bug families in different shapes (keyword `periods=-k`, `interpolate()`, a `ewm` on a reversed "
              "frame, `X - X.median()`, `resample('QS').max()` ...), to check the rules are not tied to the LeakBench "
              f"files. Fixed: **{len(hf)}/{len(hl)}**.", "",
              "H01–H10 were written after the first rule set. H11–H13 and HC3 were written after the point-in-time "
              "selection and expanding-fit rules (2026-10-30), and two of them changed the rules: H11 was first "
              "accepted with a meaningless cross-sectional rewrite of a `tail(252)` statistic (the full-sample rule "
              "treated `tail(k)` like the whole history; it now skips `tail`/`head`, as it already skipped "
              "`iloc[-k:]`), and H13 failed until the fit rule also matched `np.polyfit(..., deg=1)`. So H11 and "
              "H13 are no longer strictly held out.", "",
              "| Strategy | Result | Rule | Sharpe before → after |", "|---|---|---|---|"]
        for r in hold:
            b, a = r.get("before") or {}, r.get("after") or {}
            after = f" → {a['sharpe']:.2f}" if a else ""
            L.append(f"| `{r['id']}` | {r['status']} | {r.get('rule') or ''} | {b.get('sharpe', 0):.2f}{after} |")
        L.append("")
    L += ["## Example diffs", ""]
    show = [r for r in fixed if r["id"][:3] in ("L06", "L14", "L17", "L21")] or fixed[:4]
    for r in show:
        L += [f"`{r['id']}` ({r['rule']}):", "", "```diff", r["diff"].rstrip(), "```", ""]
    L += ["## Honest notes", "",
          "- The rules are general pandas rewrites keyed on the AST pattern of the localized / flagged line; no "
          "rule names a LeakBench file, variable or constant. They were, however, written by the same author who "
          "wrote LeakBench and with LeakBench in view, so the in-sample rate flatters them; the held-out variants "
          "are a small sanity check, not an independent benchmark.",
          "- \"Fixed\" means the deterministic battery passes and the strategy still trades. It does not mean the "
          "patch is the economically best version: e.g. `rank(axis=0)` becomes an expanding time-series rank, "
          "`rank()` becomes a cross-sectional rank, a forward-return IC is lagged by its horizon. Review the diff.",
          "- The selection rewrites change what the strategy is: a 'best last year' universe becomes the names that "
          "were best over the trailing year *as of each date*, a 'last 60 days' liquidity screen becomes a rolling "
          "60-day screen, and a 'at least 1,200 observations' filter becomes 'at least 1,200 observations so far', "
          "so on a 1,512-day panel L15's fix only trades the last ~300 days (metrics start at the first trade). "
          "The polyfit rewrite fits each point on data up to it (expanding OLS), so early residuals are noisy. "
          "These are the honest point-in-time versions, not tuned ones.",
          "- Candidates are tried in a fixed order and the first passing one wins; a later candidate could give a "
          "different Sharpe.",
          "- The per-date `pd.qcut` rewrite is a Python-level groupby and the slowest repair here (L24).",
          "- Clean strategies get no change because the Quick fix only runs when a blocking test fails; the rules "
          "never fire on passing code."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def run(only: list[str] | None = None, log=print) -> dict:
    from bench.run_bench import load_strategies
    import pandas as pd

    from data.synthetic import generate_panel

    data = generate_panel(seed=51)   # same panel as bench/run_repair.py
    strategies = load_strategies() + load_holdout()
    if only:
        strategies = [s for s in strategies if any(s["id"].startswith(o) for o in only)]
    rows = []
    for s in strategies:
        r = run_one(s, data)
        rows.append(r)
        log(f"{s['id']}: {r['status']} {r.get('rule') or ''} lines={r.get('patched_lines')} "
            f"tried={r.get('n_tried')}/{r.get('n_candidates')} ({r['seconds']}s)")
    return {"benchmark": "quickfix", "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "data": {"n_tickers": int(data["close"].shape[1]), "n_dates": int(len(data["close"]))},
            "pandas": pd.__version__, "llm_calls": 0, "cost_usd": 0.0, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="id prefixes to (re)run; merged into the existing results")
    a = ap.parse_args(argv)
    res = run(a.only)
    if a.only and OUT_JSON.exists():
        old = json.loads(OUT_JSON.read_text())
        by = {r["id"]: r for r in old["rows"]}
        by.update({r["id"]: r for r in res["rows"]})
        res["rows"] = [by[k] for k in sorted(by, key=lambda k: (by[k]["holdout"], k[0] != "L", k))]
    OUT_JSON.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    write_md(res)
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

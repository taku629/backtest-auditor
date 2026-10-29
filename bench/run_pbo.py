"""PBO on the 15 clean LeakBench strategies (offline, no LLM): python -m bench.run_pbo

Writes bench/pbo_results.json and bench/PBO.md. Leaky strategies are not scored: their PBO would measure the
leak, and run_audit skips PBO while a blocking test fails.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def main() -> None:
    from audit.pbo import compute_pbo, risk_label
    from bench.run_bench import load_strategies
    from data import generate_panel

    data = generate_panel()
    rows = []
    for s in load_strategies():
        if s["label"] != "clean":
            continue
        t0 = time.time()
        r = compute_pbo(s["stripped"], data)
        rows.append({"id": s["id"], "status": r.get("status"), "reason": r.get("reason"),
                     "params": [p["name"] for p in r.get("params", [])], "pbo": r.get("pbo"),
                     "risk": risk_label(r["pbo"]) if r.get("status") == "ok" else None,
                     "n_variants": r.get("n_variants"), "n_splits": r.get("n_splits"),
                     "prob_oos_loss": r.get("prob_oos_loss"), "degradation_slope": r.get("degradation_slope"),
                     "base_sharpe": (r.get("variants") or [{}])[0].get("sharpe"),
                     "seconds": round(time.time() - t0, 2)})
        print(rows[-1]["id"], rows[-1]["pbo"])
    (HERE / "pbo_results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ok = [r for r in rows if r["status"] == "ok"]
    md = ["# PBO on the clean LeakBench strategies", "",
          "`python -m bench.run_pbo` (offline, default synthetic panel (seed 51), 10 bps costs). The grid is built by the auditor: "
          "each detected parameter is varied 0.5x-2x (at most 16 variants), CSCV with 10 blocks (252 splits).", "",
          "| Strategy | Parameters varied | Variants | PBO | Risk | P(OOS loss) | Sharpe as written (common window) | Time |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            md.append(f"| {r['id']} | skipped: {r['reason']} | | | | | | |")
            continue
        md.append(f"| {r['id']} | {', '.join(r['params'])} | {r['n_variants']} | {r['pbo']:.2f} | {r['risk']} | "
                  f"{r['prob_oos_loss']:.0%} | {r['base_sharpe']:.2f} | {r['seconds']:.1f}s |")
    n_hi = sum(r["risk"] == "high" for r in ok)
    md += ["", f"**{n_hi}/{len(ok)} clean strategies have PBO > 0.5** (the `pbo` test fails, as a non-blocking warning); "
           f"median PBO {sorted(r['pbo'] for r in ok)[len(ok) // 2]:.2f}; median runtime "
           f"{sorted(r['seconds'] for r in ok)[len(ok) // 2]:.1f}s.", "",
           "How to read this honestly:", "",
           "- These strategies are *leak-free*, not *skilful*. A high PBO here does not mean a bug; it means that "
           "choosing among nearby parameter values by backtest would not have picked a better variant out of sample. "
           "For slow, market-heavy rules (50/200 MA crossover, 12-1 momentum, low-vol) the variants are nearly the same "
           "portfolio, so their ranking is mostly noise and PBO sits around 0.5-0.75.",
           "- That is why the test is non-blocking (`medium`): the grid is the auditor's, not the researcher's real search.",
           "- The degradation slope (OOS Sharpe of the IS winner regressed on its IS Sharpe) is close to -1 for every "
           "strategy. With CSCV the IS and OOS halves are complements of one fixed sample, so a variant that is lucky in "
           "one half is mechanically unlucky in the other; the slope is reported but not used for the verdict.",
           "- Benchmark, strategies and auditor were written by the same author on synthetic data."]
    (HERE / "PBO.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

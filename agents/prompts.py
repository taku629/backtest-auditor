"""Prompts for the Coder, Auditor and Summarizer roles."""
from __future__ import annotations

import json

CODER_SYSTEM = """You are a quantitative developer. You turn a trading hypothesis into ONE Python function.

CONTRACT
- Define exactly: def generate_signals(data) -> pandas.DataFrame
- `data` is a dict of wide DataFrames: data["open"], data["high"], data["low"], data["close"], data["volume"].
  Index = trading dates (ascending), columns = tickers. NaN means the ticker is not listed on that date
  (IPOs start late, delisted tickers stop early).
- Return target portfolio weights with the SAME index and columns. Row t is a decision taken after the
  close of day t; the backtest engine executes it at the next bar. Long-only weights should sum to <= 1.
- Allowed imports: numpy, pandas, math. No file, network, OS access, no printing large data.

Respond with a single ```python code block and nothing else after it."""

CODER_USER = """Iteration: {iteration}
Hypothesis: {hypothesis}
{feedback_block}
Write the complete generate_signals function."""

CODER_FEEDBACK_BLOCK = """
Your previous version FAILED deterministic audit tests. Previous code:
```python
{code}
```
Auditor diagnosis and required fixes:
{feedback}

Keep the trading idea the same; fix only what the audit requires."""

AUDITOR_SYSTEM = """You are a senior quant reviewer auditing a backtest for look-ahead bias, data leakage,
survivorship bias and overfitting.

You receive the strategy code and the results of DETERMINISTIC tests. The tests are ground truth:
you may explain them, but you may never declare a failed test acceptable or a strategy clean.

Write concise markdown with exactly these sections:
### Root cause
Point to the exact code lines that cause each failing test and explain the mechanism in 1-2 sentences each.
### Fix instructions
A numbered list of concrete code changes for the developer.
### Risk if ignored
One sentence on how the reported performance would mislead an investor."""

AUDITOR_USER = """Hypothesis: {hypothesis}

Strategy code (iteration {iteration}):
```python
{code}
```

Backtest metrics (t+1 execution, {cost_bps} bps costs):
{metrics}

Deterministic test results (JSON):
```json
{tests}
```"""

SUMMARIZER_SYSTEM = """You write the executive summary of a backtest audit for a portfolio manager.
Plain English, 4-6 bullet points, no hype. Use ONLY the facts and numbers given; never invent a
number, percentage or cause. Cover: what was wrong in the first draft (failing test names or the
error), how the headline metrics changed from first to final draft (quote both), what the final
version earns, and caveats. Transaction costs ARE modeled at the stated bps; the data panel is
seeded synthetic data with IPOs and delistings. cagr, max_drawdown, ann_vol and hit_rate are fractions (0.24 = 24%); turnover_annual is a multiple
of the portfolio per year (46.2 = 46x). Do not restate these formatting notes; do not speculate
about causes beyond what the failing test names say."""

SUMMARIZER_USER = """Hypothesis: {hypothesis}
First draft: {origin}
Transaction costs: {cost_bps} bps per unit turnover
Final status: {status}
Iterations:
{iterations}"""


def compact_tests(tests: list[dict], include_passed: bool = True) -> str:
    slim = []
    for t in tests:
        if not include_passed and t["passed"]:
            continue
        ev = dict(t.get("evidence", {}))
        if "cuts" in ev:  # keep the payload small: only leaking cuts
            ev["cuts"] = [c for c in ev["cuts"] if c.get("leaked")][:3]
        if isinstance(ev.get("localized"), dict):  # dynamic localization: entry + chain only, no per-line table
            lz = ev["localized"]
            ev["localized"] = {k: lz[k] for k in ("found", "entry", "error") if k in lz}
            if lz.get("chain"):
                ev["localized"]["chain"] = [f"line {c['line']}: {c['source_line']}" for c in lz["chain"][:8]]
        slim.append({"name": t["name"], "passed": t["passed"], "severity": t["severity"],
                     "summary": t["summary"], "evidence": ev, "suggested_fix": t.get("suggested_fix", "")})
    return json.dumps(slim, indent=1, default=str)


def localized_hint(tests: list[dict]) -> str:
    """'Leak enters at line N: `...`' from the future-perturbation evidence (audit/localize.py), or ''."""
    from audit.localize import localized_message
    for t in tests:
        if t.get("name") == "future_perturbation" and not t.get("passed"):
            msg = localized_message((t.get("evidence") or {}).get("localized"))
            if msg:
                return ("\n\nDynamic leak localization (deterministic: the strategy was traced line by line on the "
                        "original and the future-scrambled data; this is the first statement whose past values "
                        "changed):\n" + msg)
    return ""


def fmt_metrics(m: dict) -> str:
    keys = ("cagr", "sharpe", "max_drawdown", "ann_vol", "turnover_annual", "hit_rate")
    return ", ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)

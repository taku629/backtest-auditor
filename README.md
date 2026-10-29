# Backtest Auditor

**LLM agents write the trading strategy. Deterministic tests decide whether it is honest.**

Type a trading idea in plain English ("buy stocks whose 20-day momentum is in the top decile, rebalance weekly").
A Nemotron **Coder** agent writes the strategy, it runs in a sandbox, and a battery of deterministic tests
hunts for look-ahead bias, survivorship bias and overfitting. When a test fails, a deterministic **Quick fix**
first tries rule-based patches of the exact leak line, with no LLM at all; only if none passes does a Nemotron
**Auditor** agent read the failing evidence and tell the Coder what to fix. The loop repeats until no blocking test fails.

The LLM never judges correctness on its own. Numbers do.

> Built for the Nebius x NVIDIA Global AI Hackathon, *Coding & Agentic Engineering* track.

---

## The problem

AI coding agents are now writing quantitative research code, and they make the same mistakes junior quants make,
only faster: a `.shift(-1)` that reads tomorrow's price, a centred rolling window, a `bfill`, a universe built from
stocks that survived to today. These bugs do not crash. They produce a beautiful equity curve, and the error is
found later, with real money. Asking another LLM "is this code correct?" is not a control, because it can be confidently wrong.

## How it works

```
 hypothesis ─▶ Coder (Nemotron Super) ─▶ code ─▶ Sandbox ─▶ signals on the real data + perturbed copies
                     ▲                                              │
                     │                                              ▼
                     │                        ⚡ Quick fix: patch the localized line, re-audit (no LLM, $0)
                     │                                              │ no candidate passes
           fix instructions ◀── Auditor (Nemotron Ultra) ◀── deterministic test JSON + metrics
                                                                    │
                                   stop when no critical/high test fails ─▶ Summarizer (Nemotron Lightning)
```

### Deterministic tests (`audit/`)

| Test | What it proves | Blocking? |
|---|---|---|
| **Future-perturbation** | For random cut dates *t*, all data after *t* is scrambled (returns shuffled, sign-flipped and noised; some tickers delisted, others get short NaN gaps, and names already delisted at *t* trade again, so a `bfill`/`interpolate` over a missing stretch pulls post-*t* values into the past). Signals at dates ≤ *t* must be bit-identical. Reports the first leaking date and the size of the weight change. | critical |
| **Survivorship** | (a) no weight on tickers before they list; (b) counterfactual where delisted names keep trading: past signals must not change; (c) warns when the dataset has no delisted names at all. | critical / high |
| **Execution delay** | Sharpe at t+1 vs t+2. A collapse of more than 25% from a Sharpe above 1 points to same-bar or look-ahead execution. | high |
| **Static AST scan** | `.shift(-n)`, `pct_change(-n)`, `diff(-n)`, `rolling(center=True)`, `bfill`/`backfill`, `dropna(axis=1)`, full-sample `mean()/std()` normalisation, `interpolate`; plus pandas default-axis traps as non-blocking hints: `rank()` without `axis`, `X.sub(X.mean(), axis=1)` / `X - X.mean()`, `fillna(X.mean())`, start-labelled `resample('MS').last()`, `groupby(month).transform('last')`, rolling/cum* on a newest-first index, `pd.qcut`/`quantile` on a stacked panel; for backtrader code, a positive line index such as `self.data.close[1]` (the next bar). | critical/high rules block, others warn |
| **Deflated Sharpe Ratio** | Bailey & López de Prado (2014): the probability that the Sharpe beats the expected maximum of *N* unskilled trials, where *N* = loop iterations so far; corrects for skew and kurtosis. | warning |
| **Probability of Backtest Overfitting** | Bailey, Borwein, López de Prado & Zhu (2015), CSCV over a grid of the strategy's own parameters (≤ 16 variants, 0.5×–2×). Fails above PBO 0.5. Runs only once no blocking test fails. See [below](#probability-of-backtest-overfitting-pbo). | warning |
| **Train/test split** | 70/30 Sharpe degradation. | warning |
| **Plausibility** | Sharpe above 3 after costs on daily equities is treated as a bug until shown otherwise. | warning |

Every test returns structured JSON: `name, passed, severity, blocking, summary, evidence, suggested_fix`.

### Backtest engine (`engine/backtester.py`)

The strategy contract is a panel function:

```python
def generate_signals(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # data["open"|"high"|"low"|"close"|"volume"]: dates x tickers, NaN = not listed
    # return target weights, same shape; row t = decision after the close of t
```

The engine executes row *t* at the **next bar's open** (or the next close) with a configurable lag of at least 1,
and charges linear costs per unit of turnover. Metrics: CAGR, annual volatility, Sharpe, max drawdown, turnover,
hit rate, skew and kurtosis. The engine is causal by construction. The audit catches strategies that *read* the future.

### Sandbox (`sandbox/`)

Generated code runs in a separate `python -I` process with:
- an AST policy: only `numpy`, `pandas`, `math` and `datetime` imports (`vectorbt` / `backtrader` resolve to the trusted adapter stubs, never the real libraries); no dunder or private attributes; no file, network or eval APIs;
- restricted builtins (no `open`, `exec`, `eval`, `getattr` …) and an import allow-list at runtime;
- an **empty environment**, so API keys never reach generated code; sockets disabled; a temp working directory;
- OS limits: CPU seconds, file size 0 (no writes), no fork, and a wall-clock timeout;
- results returned as JSON plus raw float64 arrays, never pickle.

The test suite checks the runtime layer with the static policy switched off. This is defence in depth for a demo,
not a multi-tenant security boundary: in production, add a container or microVM layer.

### Data (`data/`)

A seeded synthetic panel with 50 tickers over 6 years, market and sector factors, volatility regimes, fat tails,
**8 delisted names that crash before they disappear** and **6 IPOs**. A slowly varying latent drift gives a small,
real momentum effect, so an honest strategy earns a realistic Sharpe while a leaky one looks spectacular.
`data/loader.py` loads your own long-format CSV (`date,ticker,close[,open,high,low,volume]`). No network is needed.

## How Nemotron and Nebius Token Factory are used

All LLM calls go through the OpenAI-compatible **Nebius Token Factory** endpoint (`https://api.tokenfactory.nebius.com/v1/`),
with routing set in `agents/config.py` and overridable by environment variable:

| Role | Default model | Why |
|---|---|---|
| Coder | `nvidia/nemotron-3-super-120b-a12b` | Strong code generation at low cost; called once per iteration |
| Auditor | `nvidia/Nemotron-3-Ultra-550b-a55b` | Deepest reasoning, for root-causing failing evidence from code plus test JSON |
| Summarizer | `nvidia/Nemotron-3_5-Lightning` | Fast and cheap, for the final executive summary |

The UI shows which model handled each step, with token usage, latency and estimated cost.
`scripts/check_models.py` checks the IDs against the live catalogue (all three verified on 2026-10-28).
Coder and Auditor run with Nemotron reasoning on (the trace arrives in `reasoning_content` and is kept out
of the code); the Summarizer runs with `enable_thinking: false` for speed.

## Live results (Nemotron on Token Factory, 2026-10-28)

| Run | First draft | Audited final | Iterations | Cost |
|---|---|---|---|---|
| Audit my own code: leaky momentum pasted (`examples/leaky_momentum.py`) | Sharpe 6.63, CAGR 545%; fails future-perturbation, static scan, survivorship, plausibility | Sharpe −0.67, clean | 2 (Ultra diagnosed, Super fixed) | < $0.01 |
| From hypothesis only (top-decile momentum) | Super's draft crashed in the sandbox (shape mismatch) | Sharpe 0.94, CAGR 24%, clean | 2 | ≈ $0.003 |

Full traces, including every model call with tokens and latency: `examples/live_nemotron_*.json`.
Lesson: on this prompt Nemotron did not leak future data on its own, so the product also audits code you
paste from a notebook, where look-ahead bugs usually live.

## Results on the bundled demo (Mock mode, seed 51)

| Scenario | First draft (leaky) | Audited final, Quick fix on (default) | Audited final, LLM path only |
|---|---|---|---|
| Momentum, top decile, weekly | Sharpe 6.63, CAGR 545% | Sharpe 0.73, CAGR 18%, 2 iterations, 0 repair calls | Sharpe 0.97, CAGR 25%, 3 iterations |
| Mean reversion below MA | Sharpe 10.96 | Sharpe −0.92, 2 iterations, 0 repair calls | Sharpe −0.87, 2 iterations |
| Low volatility, monthly | Sharpe 0.65 | Sharpe 0.70, 2 iterations, 0 repair calls | Sharpe 0.70, 2 iterations |

The honest mean-reversion strategy loses money either way. Quick fix patches are minimal edits of the draft, so
they can differ from the Coder's rewrite (for momentum, the Quick fix turns `shift(-1)` into `shift(1)`, one extra day of lag, while the mock Coder's rewrite drops the shift and rebuilds the weekly rebalance).

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # 303 tests, fully offline
USE_MOCK=1 streamlit run app.py          # offline demo, reproducible
NEBIUS_API_KEY=... streamlit run app.py  # live Nemotron on Token Factory
python cli.py "Buy top-decile 20-day momentum, rebalance weekly" --out run.trace.json   # headless
python cli.py --live --code examples/leaky_momentum.py   # audit your own strategy file
python audit_cli.py examples/leaky_momentum.py --fix --no-llm --report audit.html   # printable audit report
```

Environment variables: `NEBIUS_API_KEY`, `USE_MOCK`, `NEBIUS_BASE_URL`, `CODER_MODEL`, `AUDITOR_MODEL`, `SUMMARIZER_MODEL`, `MAX_LIVE_RUNS_PER_SESSION` (public demo budget, default 5).

### Deploy on Render

`render.yaml` plus `Dockerfile` (Python 3.11-slim). Set `NEBIUS_API_KEY` in the dashboard. Without it the app runs in Mock mode.

## LeakBench: deterministic tests vs LLM-only review

Can another LLM just review the code instead? `bench/` holds 39 labelled strategies (24 leaky, 15 clean),
from textbook bugs to realistic pandas traps (`rank()` defaulting to the time axis, `sub(mean(), axis=1)`,
`resample("MS").last()`, newest-first rolling windows) plus clean code that *looks* scary (`shift(1)`, a helper
named `peek_next`, a month-end flag built from the trading calendar). Nemotron judges see the code with labels stripped.

| Method | Recall (leaks caught) | False-positive rate | Accuracy |
|---|---|---|---|
| **Backtest Auditor (deterministic tests)** | **100%** | **0%** | **100%** |
| LLM-only: Nemotron 3 Ultra | 83% | 13% | 85% |
| LLM-only: Nemotron 3 Super | 88% | 0% | 92% |

Both LLM judges were perfect on the textbook bugs and missed the same subtle pandas traps, which the perturbation test
catches by experiment. Full table and every wrong verdict with the model's reasoning: [`bench/RESULTS.md`](bench/RESULTS.md).
Rerun offline from the cache with `python -m bench.run_bench` (the live run cost $0.22).
Caveat: benchmark and auditor were written by the same author on synthetic data; treat it as a set of realistic bug patterns.

## Dynamic leak localization: which line reads the future?

When the perturbation test fails, Backtest Auditor re-runs the strategy with every assignment instrumented (inside the
same sandbox) on the original and the perturbed data, and reports the **first statement whose value changes although
nothing it reads changed**: the exact line where future information enters. This works even where code review fails:
on LeakBench it pins the right line for **24/24** leaky strategies (including `rank()` on the time axis,
`sub(mean(), axis=1)`, `resample("MS").last()` and newest-first rolling windows, which both Nemotron judges missed)
and reports no leak for 15/15 clean ones. The line is highlighted in the UI, annotated in CI, and handed to the
Nemotron Auditor ("Leak enters at line 13: `pctl = mom.rank(pct=True)`"). See `bench/LOCALIZE.md`.

## Repair benchmark

`bench/REPAIR.md`: live Nemotron loop on the 24 leaky strategies, 3 runs each, with and without the localization hint.
Repaired 69/72 (with hint) vs 70/72 (without); the hint cuts Auditor output tokens by 46% and wrong line references
from 5 to 1; no run passed by going to cash; all failures were Coder rewrites crashing in the sandbox. $1.08 total.

## Quick fix: deterministic repair, no LLM

Once the localizer has pinned the leak line, `audit/quickfix.py` generates a few candidate patches for it (and for the
lines the static scan flags) from general pandas rewrite rules: `shift(-k)` → `shift(k)` (or lag the trailing estimate
that consumes a forward target), centred → trailing window, `bfill`/`interpolate` → `ffill`, `rank()` →
`rank(axis=1)` or an expanding rank, full-sample `mean/std/quantile` → `expanding()`, `sub(X.mean(), axis=1)` →
cross-sectional demeaning, `resample('MS').last()` → `resample('ME')`, `transform('last')` → `'first'`, drop a
time reversal, pooled `qcut` → per date, `dropna(axis=1)` → keep every ticker, and a last-resort `.shift(1)`.
Each candidate is re-run through the **full** deterministic audit; the first one that passes every blocking test
without going to cash (`non_trivial` passes, gross exposure ≥ 5%) is accepted and shown as a unified diff with
Sharpe before/after. The agent loop tries it before calling the Coder and falls back to Nemotron when it fails.

| Repair on the 24 leaky LeakBench strategies | Repaired | Changes on 15 clean | Cost per repair |
|---|---|---|---|
| ⚡ Quick fix (deterministic) | **19/24** (79%) | 0/15 (it only runs when a blocking test fails) | **$0** |
| Nemotron loop (`bench/REPAIR.md`) | 69/72 runs (96%) | not measured | $0.0072 |

The 5 it declines are selection and regression leaks (universe chosen from the last row or last 60 days, a full-sample
count filter, a per-ticker `polyfit`) that need a redesign rather than a one-line rewrite; the LLM repaired those in
14/15 live runs, so Quick fix first, LLM second, would need an LLM call for only 5 of the 24. Held-out check: 10/10 extra leaky variants written after the rules (`bench/quickfix_holdout/`, same author).
Caveat: the rules are general, not file-specific, but they were written by the LeakBench author with LeakBench in
view, so treat 19/24 as an in-sample number. Details: [`bench/QUICKFIX.md`](bench/QUICKFIX.md)
(`python -m bench.run_quickfix`, offline).

## Probability of Backtest Overfitting (PBO)

The Deflated Sharpe corrects for how many *loop iterations* were tried. PBO asks a different question: if you had
tuned this strategy's parameters on the backtest, would the winner keep winning? `audit/pbo.py` implements
Combinatorially Symmetric Cross-Validation (Bailey, Borwein, López de Prado & Zhu, 2015):

1. **Grid.** Parameters are module-level UPPERCASE numeric constants and numeric defaults of `generate_signals`;
   without those, inline literals that act as parameters (lookback windows in `rolling/ewm/shift/pct_change/diff`,
   thresholds such as `rank >= 0.9` or `rank <= 5`). `shift(1)` and `min_periods` are never varied (a literal
   `min_periods` is scaled with its window). Each value is varied 0.5×–2× (for `0.9`-style quantile thresholds
   the distance to 1 is scaled: top 10% → top 5–20%), at most 16 variants. You can pass your own grid:
   `audit_cli.py s.py --param-grid "LOOKBACK=10,20,40;TOP=0.8,0.9"`.
2. **Run.** All variants run in **one** sandbox process (each as a closure of the rewritten source), then
   through the same backtester and costs: a T×N return matrix.
3. **CSCV.** T is cut into 10 blocks (8 for short samples); for all 252 ways of picking 5 blocks as in-sample,
   the in-sample best variant's out-of-sample rank ω gives a logit ln(ω/(1−ω)). **PBO = share of logits ≤ 0.**
   Also reported: the logit histogram, the IS-vs-OOS Sharpe of the winner, the degradation slope and P(OOS loss).

Verdict: PBO > 0.5 fails, 0.25–0.5 is "moderate", below is "low". The test is **non-blocking** (`medium`):
the grid is the auditor's, not your actual search history, and a high PBO can simply mean the variants are nearly
the same portfolio. It is skipped (with the reason) when there are no tunable parameters, fewer than 4 distinct
variants, or a blocking test already fails (PBO of a leaky strategy measures the leak). Typical runtime 1–4 s.

On the 15 clean LeakBench strategies (`bench/PBO.md`, `python -m bench.run_pbo`): median PBO 0.10;
**6/15 exceed 0.5** (50/200 MA crossover 0.74, 12-1 momentum 0.65, low-vol 0.58, …). Those are leak-free but
slow, market-heavy rules whose parameter variants are near-duplicates, so choosing among them by backtest is
a coin flip; that is what PBO reports. The unit tests check that a noise-mined grid scores high and variants of a
genuine signal score low. Caveat: the degradation slope comes out near −1 for every strategy (CSCV halves are
complements of one sample), so it is shown but not used for the verdict.

## Downloadable audit report

One self-contained file to attach to a PR, send to a reviewer or print to PDF: verdict, every test with its
evidence, the leak line highlighted in the code, the Quick fix or LLM diff, before/after metrics with the equity
curve (inline SVG, no CDN, no JavaScript), PBO and DSR, models used with tokens and cost, a UTC timestamp and the
SHA-256 of the audited (and final) code. In the app: **Download audit report (HTML / Markdown)** next to the JSON
trace in the *Audit report* tab. Headless:

```bash
python audit_cli.py strategy.py --fix --no-llm --report audit.html   # or audit.md; several files -> one report
python cli.py --code examples/leaky_momentum.py --report run.html     # full agent loop (mock unless --live)
```

## Strategy format adapters: vectorbt, backtrader, `strategy(close)`

Real research code is rarely written as `generate_signals(data)`. `audit/adapters.py` detects the format and appends
a small `generate_signals` wrapper (your lines keep their line numbers, so findings, the leak line and Quick fix
diffs still point at your code). The wrapper runs in the same sandbox and calls trusted stand-ins in
`sandbox/adapter_stubs.py`: `import vectorbt` / `import backtrader` resolve to those stubs there, so **neither
library is installed or needed**, and the AST policy and runtime limits are unchanged.

| Format | Detected from | How it becomes weights |
|---|---|---|
| **Function** | `strategy(df)`, `signals(close)`, `compute_weights(prices)`, `get_positions(data)`, … (or the one top-level function whose arguments are named `close/prices/df/open/high/low/volume/data`) | arguments bound by name (`data` = OHLCV dict, `close/prices/df` = close panel). Returned weights are used as-is; a **bool or +1/0/−1** frame becomes equal weight per date, `w = s / Σ|s|`; `(entries, exits)` or a Portfolio is handled as below |
| **vectorbt** | `import vectorbt as vbt` plus a function returning `(entries, exits)` / a Portfolio, or module-level script code ending in `vbt.Portfolio.from_signals(close, entries, exits)` | entries/exits → long position state machine (entry while long and exit while flat do nothing; both on one bar cancel; `short_entries/short_exits` and `direction` honoured) → equal weight across names in position. Stubbed: `vbt.MA.run`, `vbt.RSI.run` (`*_crossed_above/below`, `*_above/below`), `.vbt.crossed_above/below`, and `vbt.YFData.download(...).get("Close")`, which returns the audit panel (no network) |
| **backtrader** | a `bt.Strategy` subclass (the one passed to `cerebro.addstrategy` if there are several) | replayed bar by bar **once per ticker**, on that ticker's own trading days. Lines index like backtrader: `[0]` current bar, `[-1]` previous, and **`[1]` the next bar**, which is what backtrader returns with preloaded data and exactly the leak to catch. Indicators: SMA, EMA, RSI (Wilder), Highest, Lowest, StdDev, ROC, Momentum, CrossOver, plus line arithmetic and comparisons; `params`/`self.p`, `self.position`, `notify_order` (orders fill at the next bar's open). `buy/sell` add ±size units (default 1), `close()` goes flat, `order_target_percent/size` set the target; weights = exposure / max(1, Σ\|exposure\|) per date |

CLI: auto-detected, or forced with `audit_cli.py s.py --strategy-format vectorbt` (`auto|signals|function|vectorbt|backtrader`;
`--format` stays the report format). The app detects pasted or uploaded `.py` code and shows e.g.
*"Detected: backtrader Strategy `SmaTrend` → adapted"*; the four examples are in the LeakBench picker under "Adapters".

Examples (`examples/`, all in the test suite): `backtrader_leaky.py` reads `self.data.close[1]` → future-perturbation
fails at 6/6 cut dates (Sharpe 4.22), the static rule `future_bar_index` flags line 23, and Quick fix rewrites
`[1]` → `[0]` (Sharpe 0.27, clean); `backtrader_clean.py` (SMA crossover) passes. `vectorbt_leaky.py` (entries
use `close.shift(-1)`, Sharpe 6.55) fails; `vectorbt_clean.py` (RSI `get_signals(close)`) passes.
The dynamic localizer cannot see inside the stubs, so when its trace ends at the wrapper call the report points
at the static look-ahead finding in your lines instead and says so.

Cost: a backtrader replay is a pure-Python bar loop, about 0.25 s CPU per 50 tickers × 6 years dataset (a full
audit runs 8 datasets: ~2 s locally, plus the localizer and Quick fix when it leaks). Replays above 600,000
ticker-bars per dataset are refused with a message, and PBO is skipped for backtrader strategies.

## Jupyter notebooks

Upload an `.ipynb` in the app or pass it to the CLI: code cells are concatenated (magics skipped), `generate_signals`
is extracted with the helpers, constants and imports it uses, and findings point at "cell 10, line 4".
`examples/leaky_research.ipynb` is a realistic leaky research notebook.

## Use it in CI

```bash
python audit_cli.py "strategies/**/*.py"            # exit 1 on look-ahead / survivorship; points at the lines
python audit_cli.py strategy.py --format sarif --out audit.sarif   # GitHub code scanning annotations
python audit_cli.py strategy.py --fix               # Quick fix diff (no LLM); falls back to Nemotron with NEBIUS_API_KEY
python audit_cli.py strategy.py --fix --write --no-llm   # write the deterministic patch to strategy.fixed.py
python audit_cli.py strategy.py --report audit.html      # self-contained HTML (or .md) audit report
```

GitHub Action: `uses: taku629/backtest-auditor@main` with `paths:` (annotates the offending lines in PRs, uploads SARIF).
A pre-commit hook (`backtest-auditor`) is in `.pre-commit-hooks.yaml`. Details: [`docs/ci.md`](docs/ci.md).

## Real market data

Besides the synthetic bench, the app ships adjusted daily prices for 40 US stocks and ETFs (2014–2025, incl. 7 IPOs
inside the window). The leaky momentum example scores Sharpe 5.67 there vs 0.78 for the honest version, and the
perturbation test flags it at 6/6 cut dates. The free data has no delisted names, so the dataset itself is
survivor-biased and the audit says so. See `data/real/README.md`.

## Repository layout

```
app.py                 Streamlit UI
cli.py                 headless loop
agents/                config (model routing, pricing), LLM clients (Nebius + Mock), prompts, loop, canned drafts
audit/                 leak_tests.py (perturbation, survivorship, delay, DSR, report), ast_scan.py, overfitting.py, pbo.py,
                       localize.py (dynamic leak line), quickfix.py (deterministic repair), report.py (HTML/MD report)
engine/backtester.py   causal daily backtester and metrics
sandbox/               runner.py (policy + subprocess), _child.py (restricted executor), adapter_stubs.py
                       (trusted vectorbt / backtrader stand-ins)
audit/adapters.py      strategy format detection + generate_signals wrapper
data/                  synthetic.py (seeded panel), loader.py (CSV)
bench/                 LeakBench strategies, runner, results, Streamlit view
audit_cli.py           CI entry point (text/markdown/json/SARIF, GitHub annotations)
action.yml             composite GitHub Action
tests/                 pytest suite (303 tests)
```

## Limitations

- The synthetic data is a test bench, not a claim about real markets. Use the CSV loader for real data.
- Perturbation tests detect any dependence on future rows. They cannot tell a legitimate use of future data from a bug, but a backtest has no legitimate use for it.
- The thresholds for execution delay, DSR and plausibility are heuristics and can be configured in code.
- Adapters are small, honest stand-ins, not the libraries. Not modelled (each fails with a clear "not supported"
  error, never silently): backtrader broker cash/value, sizers, analyzers, observers, Limit/Stop/bracket orders,
  `cancel`, several data feeds in one strategy (each ticker is replayed alone, so pair or cross-sectional logic
  inside `next()` cannot be expressed), custom `bt.Indicator` classes, indicators beyond the list above; vectorbt
  stops (`sl_stop`, `tp_stop`), custom sizing, `from_orders` / `from_order_func`, parameter sweeps (lists of
  windows). `start()`, `stop()` and `notify_trade()` are not called. Cerebro setup and vectorbt fees/cash are
  ignored (the auditor applies its own next-bar execution and costs), and the equal-weight conversion of signals is
  a convention, so metrics differ from what the original library reports; the leak tests do not depend on it.
- Notebooks still need a `generate_signals` cell; the adapters apply to `.py` files and pasted code.
- A module-level vectorbt script whose first statement directly follows the imports (no blank line) is wrapped
  with a one-line shift; localization and Quick fix are then switched off for it.

## License

MIT

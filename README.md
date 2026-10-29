# Backtest Auditor

**LLM agents write the trading strategy. Deterministic tests decide whether it is honest.**

Type a trading idea in plain English ("buy stocks whose 20-day momentum is in the top decile, rebalance weekly").
A Nemotron **Coder** agent writes the strategy, it runs in a sandbox, and a battery of deterministic tests
hunts for look-ahead bias, survivorship bias and overfitting. A Nemotron **Auditor** agent reads the failing
evidence and tells the Coder exactly what to fix. The loop repeats until no blocking test fails.

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
| **Static AST scan** | `.shift(-n)`, `pct_change(-n)`, `diff(-n)`, `rolling(center=True)`, `bfill`/`backfill`, `dropna(axis=1)`, full-sample `mean()/std()` normalisation, `interpolate`; plus pandas default-axis traps as non-blocking hints: `rank()` without `axis`, `X.sub(X.mean(), axis=1)` / `X - X.mean()`, `fillna(X.mean())`, start-labelled `resample('MS').last()`, `groupby(month).transform('last')`, rolling/cum* on a newest-first index, `pd.qcut`/`quantile` on a stacked panel. | critical/high rules block, others warn |
| **Deflated Sharpe Ratio** | Bailey & López de Prado (2014): the probability that the Sharpe beats the expected maximum of *N* unskilled trials, where *N* = loop iterations so far; corrects for skew and kurtosis. | warning |
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
- an AST policy: only `numpy`, `pandas` and `math` imports; no dunder or private attributes; no file, network or eval APIs;
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

| Scenario | First draft (leaky) | Audited final | Iterations |
|---|---|---|---|
| Momentum, top decile, weekly | Sharpe 6.63, CAGR 545% | Sharpe 0.97, CAGR 25% | 3 |
| Mean reversion below MA | Sharpe 10.96 | Sharpe −0.87 (the honest version loses money) | 2 |
| Low volatility, monthly | Sharpe 0.65 | Sharpe 0.70 | 2 |

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # 46 tests, fully offline
USE_MOCK=1 streamlit run app.py          # offline demo, reproducible
NEBIUS_API_KEY=... streamlit run app.py  # live Nemotron on Token Factory
python cli.py "Buy top-decile 20-day momentum, rebalance weekly" --out run.trace.json   # headless
python cli.py --live --code examples/leaky_momentum.py   # audit your own strategy file
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

## Jupyter notebooks

Upload an `.ipynb` in the app or pass it to the CLI: code cells are concatenated (magics skipped), `generate_signals`
is extracted with the helpers, constants and imports it uses, and findings point at "cell 10, line 4".
`examples/leaky_research.ipynb` is a realistic leaky research notebook.

## Use it in CI

```bash
python audit_cli.py "strategies/**/*.py"            # exit 1 on look-ahead / survivorship; points at the lines
python audit_cli.py strategy.py --format sarif --out audit.sarif   # GitHub code scanning annotations
python audit_cli.py strategy.py --fix               # with NEBIUS_API_KEY: Nemotron writes strategy.fixed.py
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
audit/                 leak_tests.py (perturbation, survivorship, delay, DSR, report), ast_scan.py, overfitting.py
engine/backtester.py   causal daily backtester and metrics
sandbox/               runner.py (policy + subprocess), _child.py (restricted executor)
data/                  synthetic.py (seeded panel), loader.py (CSV)
bench/                 LeakBench strategies, runner, results, Streamlit view
audit_cli.py           CI entry point (text/markdown/json/SARIF, GitHub annotations)
action.yml             composite GitHub Action
tests/                 pytest suite (122 tests)
```

## Limitations

- The synthetic data is a test bench, not a claim about real markets. Use the CSV loader for real data.
- Perturbation tests detect any dependence on future rows. They cannot tell a legitimate use of future data from a bug, but a backtest has no legitimate use for it.
- The thresholds for execution delay, DSR and plausibility are heuristics and can be configured in code.

## License

MIT

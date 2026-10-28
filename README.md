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
| **Future-perturbation** | For random cut dates *t*, all data after *t* is scrambled (returns shuffled, sign-flipped and noised; some tickers delisted). Signals at dates ≤ *t* must be bit-identical. Reports the first leaking date and the size of the weight change. | critical |
| **Survivorship** | (a) no weight on tickers before they list; (b) counterfactual where delisted names keep trading: past signals must not change; (c) warns when the dataset has no delisted names at all. | critical / high |
| **Execution delay** | Sharpe at t+1 vs t+2. A collapse of more than 25% from a Sharpe above 1 points to same-bar or look-ahead execution. | high |
| **Static AST scan** | `.shift(-n)`, `pct_change(-n)`, `diff(-n)`, `rolling(center=True)`, `bfill`/`backfill`, `dropna(axis=1)`, full-sample `mean()/std()` normalisation, `interpolate`. | critical/high rules block, others warn |
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
`scripts/check_models.py` checks the IDs against the live catalogue.

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
pytest                                   # 45 tests, fully offline
USE_MOCK=1 streamlit run app.py          # offline demo, reproducible
NEBIUS_API_KEY=... streamlit run app.py  # live Nemotron on Token Factory
python cli.py "Buy top-decile 20-day momentum, rebalance weekly" --out run.trace.json   # headless
```

Environment variables: `NEBIUS_API_KEY`, `USE_MOCK`, `NEBIUS_BASE_URL`, `CODER_MODEL`, `AUDITOR_MODEL`, `SUMMARIZER_MODEL`.

### Deploy on Render

`render.yaml` plus `Dockerfile` (Python 3.11-slim). Set `NEBIUS_API_KEY` in the dashboard. Without it the app runs in Mock mode.

## Repository layout

```
app.py                 Streamlit UI
cli.py                 headless loop
agents/                config (model routing, pricing), LLM clients (Nebius + Mock), prompts, loop, canned drafts
audit/                 leak_tests.py (perturbation, survivorship, delay, DSR, report), ast_scan.py, overfitting.py
engine/backtester.py   causal daily backtester and metrics
sandbox/               runner.py (policy + subprocess), _child.py (restricted executor)
data/                  synthetic.py (seeded panel), loader.py (CSV)
tests/                 pytest suite
```

## Limitations

- The synthetic data is a test bench, not a claim about real markets. Use the CSV loader for real data.
- Perturbation tests detect any dependence on future rows. They cannot tell a legitimate use of future data from a bug, but a backtest has no legitimate use for it.
- The thresholds for execution delay, DSR and plausibility are heuristics and can be configured in code.

## License

MIT

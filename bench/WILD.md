# In the wild: do Nemotron-written backtests leak when nobody asks them to?

Generated 2026-10-30 01:44 UTC · live Nemotron on Nebius Token Factory · `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b` · synthetic panel, 50 tickers x 1512 days, seed 51 (8 delisted, 6 IPOs).

30 ordinary strategy requests (`bench/wild_prompts.py`) were each sent once to each model under two system prompts: the app's own Coder prompt (`app`, which says row t is a decision after the close of t, executed at the next bar) and a minimal request a user would type into a chat assistant (`chat`, same function contract, no timing guidance). Each first draft (no repair loop, no retries) was run through the deterministic audit, the leak localizer and the LLM-free Quick fix exactly as the app does. **Leaky** = at least one blocking (critical/high) test fails among drafts that run and trade.

## Headline

- **25/95 (26%) of runnable drafts leak** (95 of 120 drafts ran; 25 crashed in the sandbox, 13 never took a position).
- Look-ahead (future-perturbation test failed): 20 drafts; the localizer reported a line in 19/20 (see the manual review below for how many are the root-cause line).
- Quick fix (no LLM, $0) repaired 11/25 (44%) of leaky drafts with the current rules (4/25 with the rules at the start of the study; the added rules were written after seeing these drafts, so the higher number is in-sample).
- The full agentic loop (Quick fix → Nemotron Ultra Auditor → Nemotron Super Coder, at most 2 repair attempts) repaired 11/14 of the drafts Quick fix could not ($0.171 in total), so the system repaired 22/25 (88%) of leaky drafts.
- Leaks were mostly not spectacular: median Sharpe of the leaky drafts 0.23 vs -0.04 for clean drafts; after repair, the repaired ones moved from a median 0.28 to 0.37. The equity curve would not have given these bugs away.
- PBO above 0.5 (overfitting warning, non-blocking): 16 of 44 drafts where PBO could be measured.
- Live spend for the whole study: $0.3324 (first drafts and crash retries $0.1536, repair loop $0.1709, the rest pilot calls; logged call by call in `bench/wild_spend.jsonl`; cached reruns are free).

## By condition

| Condition | Drafts | Ran | Crashed | Leaky (of ran) | Look-ahead | Survivor universe | Weight before listing | Line localized | Quick fix repaired | + loop repaired | Median Sharpe leaky → fixed (Quick fix) | PBO > 0.5 | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **All drafts** | 120 | 95 | 25 | 25/95 (26%) | 20 | 14 | 11 | 19/20 | 11/25 | 11/14 | 0.369 → 0.448 | 16/44 | $0.1536 |
| Super, app Coder prompt | 30 | 28 | 2 | 3/28 (11%) | 1 | 1 | 2 | 1/1 | 2/3 | 0/1 | 0.582 → 0.582 | 6/17 | $0.0699 |
| Super, chat prompt | 30 | 27 | 3 | 7/27 (26%) | 5 | 3 | 2 | 5/5 | 3/7 | 3/4 | 0.439 → 0.305 | 5/15 | $0.0497 |
| Lightning, app Coder prompt | 30 | 19 | 11 | 4/19 (21%) | 4 | 4 | 2 | 4/4 | 3/4 | 1/1 | -0.477 → 0.84 | 2/6 | $0.0150 |
| Lightning, chat prompt | 30 | 21 | 9 | 11/21 (52%) | 10 | 6 | 5 | 9/10 | 3/11 | 7/8 | 0.369 → 0.855 | 3/6 | $0.0189 |
| All models, plain-style prompts | 76 | 58 | 18 | 6/58 (10%) | 3 | 2 | 5 | 3/3 | 4/6 | 1/2 | 0.349 → 0.409 | 13/35 | $0.0993 |
| All models, notebook-style prompts | 44 | 37 | 7 | 19/37 (51%) | 17 | 12 | 6 | 16/17 | 7/19 | 10/12 | 0.439 → 0.835 | 3/9 | $0.0543 |

Blocking failures by test, all drafts: look_ahead 20, survivor_universe 14, pre_listing_weight 11, static_red_flag 1, same_bar_execution 1 (a draft can fail several).

## Every leaky draft

| Draft | Request | Failing blocking tests | Leak line (localizer) | Sharpe | Quick fix | Sharpe after | Loop (if Quick fix failed) |
|---|---|---|---|---|---|---|---|
| `W01|lightning|app` | Cross-sectional momentum: each week, buy the top decile of stocks by 1… | future_perturbation, survivorship | L40: `rank = ret_net.rank(pct=True, method="first", axis=0)` | -0.516 | fixed (time_axis_rank) | 1.4556 | — |
| `W04|lightning|app` | Momentum like in my notebook: compute 60-day returns with pct_change, … | future_perturbation, survivorship | L30: `mean_60 = returns_60.mean()` | 0.582 | fixed (full_sample_statistic+mask_unlisted) | 0.8404 | — |
| `W04|lightning|chat` | Momentum like in my notebook: compute 60-day returns with pct_change, … | future_perturbation, survivorship | L16: `mean = returns.mean()` | 0.634 | fixed (full_sample_statistic+mask_unlisted) | 0.855 | — |
| `W04|super|app` | Momentum like in my notebook: compute 60-day returns with pct_change, … | survivorship | — | 0.835 | fixed (mask_unlisted) | 0.8348 | — |
| `W05|lightning|chat` | 52-week-high momentum: buy stocks trading within 5% of their 52-week h… | survivorship | — | 0.369 | fixed (mask_unlisted) | 0.3699 | — |
| `W08|lightning|chat` | RSI(14) mean reversion: buy names with RSI under 30, equal weight. Cle… | future_perturbation, survivorship | L54: `n_active = oversold.sum()  # count per row (may be 0)` | -0.41 | failed | None | ✅ clean, 2 iters, Sharpe -1.073, $0.0073 |
| `W10|super|chat` | Pairs trading: find the two most correlated stocks in the universe, tr… | future_perturbation, survivorship | L27: `corr = log_ret.corr()                     # Pearson correlation, pairw` | -2.529 | no_candidate | None | clean, but < 5% avg gross exposure (not counted), 2 iters, Sharpe 0.654, $0.0102 |
| `W12|lightning|chat` | Volatility targeting: equal-weight all listed stocks, then scale gross… | future_perturbation, survivorship | L24: `n_tickers = close.notna().sum()` | 0.216 | failed | None | ✅ clean, 2 iters, Sharpe 0.379, $0.0127 |
| `W17|lightning|app` | Factor model from my notebook: build features (momentum, volatility, v… | future_perturbation, survivorship | L30: `mu = series.mean()` | -0.477 | fixed (full_sample_statistic+time_axis_rank+mask_unlisted) | 0.3693 | — |
| `W17|lightning|chat` | Factor model from my notebook: build features (momentum, volatility, v… | future_perturbation, survivorship | L47: `mu = df.mean()` | -0.244 | failed | None | ✅ clean, 2 iters, Sharpe -0.909, $0.0156 |
| `W17|super|chat` | Factor model from my notebook: build features (momentum, volatility, v… | survivorship | — | -0.166 | fixed (mask_unlisted) | -0.166 | — |
| `W19|super|app` | Group rotation: there are no sector labels, so split stocks into 5 gro… | survivorship, static_scan | — | 0.329 | fixed (mask_unlisted) | 0.33 | — |
| `W19|super|chat` | Group rotation: there are no sector labels, so split stocks into 5 gro… | survivorship | — | 0.447 | fixed (mask_unlisted) | 0.4476 | — |
| `W20|lightning|chat` | Cluster rotation: cluster stocks by the correlation of their daily ret… | future_perturbation, survivorship | L32: `corr_matrix = returns.corr(method="pearson")` | 0.029 | failed | None | ✅ clean, 2 iters, Sharpe 0.3, $0.0104 |
| `W23|super|app` | MA crossover, but pick the fast and slow window lengths (from 5-50 and… | future_perturbation, survivorship | L49: `best_pair = (f, s)` | 1.086 | failed | None | ❌ error, 4 iters, Sharpe None, $0.0190 |
| `W23|super|chat` | MA crossover, but pick the fast and slow window lengths (from 5-50 and… | future_perturbation | L54: `best_fast, best_slow = fast, slow` | 1.135 | failed | None | ✅ clean, 3 iters, Sharpe 0.903, $0.0172 |
| `W25|lightning|chat` | Liquid-universe momentum: restrict to the 25 most liquid stocks by ave… | future_perturbation | L98: `weights_df = pd.DataFrame(weight_dict, index=close.index)` | 1.783 | failed | None | ✅ clean, 2 iters, Sharpe 1.166, $0.0087 |
| `W25|super|chat` | Liquid-universe momentum: restrict to the 25 most liquid stocks by ave… | future_perturbation | L46: `for date, tickers in top5_per_date.items():` | 1.274 | failed | None | ✅ clean, 2 iters, Sharpe 1.144, $0.0083 |
| `W26|lightning|chat` | Turn-of-the-month effect: be fully invested (equal weight) from the la… | future_perturbation, survivorship | L100: `valid_tickers = close.loc[date].notna()` | -0.563 | failed | None | ❌ error, 4 iters, Sharpe None, $0.0272 |
| `W28|lightning|chat` | Consistency score: rank stocks by the fraction of up days over the las… | future_perturbation, survivorship | L20: `ranks = fraction_up.rank(pct=True)` | 0.23 | fixed (time_axis_rank) | 1.2713 | — |
| `W29|lightning|chat` | Gap fade: if a stock opens more than 2% below yesterday's close, buy i… | future_perturbation, survivorship | L33: `tickers_with_nan.update(df.columns[df.isna().any()])` | -0.608 | no_candidate | None | ✅ clean, 2 iters, Sharpe -0.766, $0.0062 |
| `W29|super|chat` | Gap fade: if a stock opens more than 2% below yesterday's close, buy i… | future_perturbation, survivorship | L34: `prev_close = subset['close'].shift(1)          # yesterday's close` | -0.803 | failed | None | ✅ clean, 2 iters, Sharpe -1.182, $0.0064 |
| `W30|lightning|app` | Residual momentum: remove each stock's linear trend with np.polyfit on… | future_perturbation, survivorship | L20: `coeffs = np.polyfit(x, series.values, 1)  # slope, intercept` | -1.015 | failed | None | ✅ clean, 2 iters, Sharpe -0.272, $0.0106 |
| `W30|lightning|chat` | Residual momentum: remove each stock's linear trend with np.polyfit on… | future_perturbation, execution_delay | — | 8.064 | no_candidate | None | ✅ clean, 2 iters, Sharpe -0.158, $0.0111 |
| `W30|super|chat` | Residual momentum: remove each stock's linear trend with np.polyfit on… | future_perturbation, survivorship | L43: `coeff = np.polyfit(x, series.values, 1)   # linear fit` | 0.439 | fixed (full_sample_fit) | 0.3054 | — |

## Drafts that did not run

- `W01|lightning|chat` (policy_violation): missing top-level function 'generate_signals(data)' (other formats - strategy(close), vectorbt entries/exits, backtrader Strategy - are adapted by audit/adapters.py first)
- `W03|lightning|app` (crashed): SandboxError: AttributeError: 'numpy.ndarray' object has no attribute 'multiply' (strategy line 49: weights = weights.multiply(scale, axis=0))
- `W03|lightning|chat` (crashed): SandboxError: ValueError: operands could not be broadcast together with shapes (1512,) (1512,50) ()  (strategy line 73: weight_matrix = np.where()
- `W03|super|app` (crashed): SandboxError: AttributeError: 'numpy.ndarray' object has no attribute 'iloc' (strategy line 31: is_month_end.iloc[-1] = True)
- `W09|lightning|app` (crashed): SandboxError: IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices (strategy line 43: w[selected] = 1.0 / len(selected))
- `W09|lightning|chat` (crashed): SandboxError: ValueError: Cross index must be 1 dimensional (strategy line 52: weights.iloc[row_idx, sorted_idx] = 1.0 / n_select)
- `W10|lightning|app` (crashed): SandboxError: AttributeError: 'numpy.ndarray' object has no attribute 'loc' (strategy line 113: weights.loc[active, best_ticker1] = 0.5 * raw_signal.loc[active])
- `W11|lightning|chat` (crashed): SandboxError: KeyError: 'SYN00' (strategy line 47: res = rets[t] - betas[t] * market_rets[t])
- `W12|lightning|app` (crashed): SandboxError: ValueError: No axis named 1 for object type Series (strategy line 57: row_sums = weights.sum(axis=1))
- `W13|super|app` (crashed): SandboxError: AttributeError: 'numpy.ndarray' object has no attribute 'values' (strategy line 43: weight_monthly = weight_raw.where(month_end.values[:, None], np.nan))
- `W14|lightning|chat` (crashed): SandboxError: IndexError: single positional indexer is out-of-bounds (strategy line 49: latest_vol = rolling_vol.iloc[-1])
- `W15|lightning|chat` (crashed): SandboxError: AttributeError: 'numpy.float64' object has no attribute 'iloc' (strategy line 52: cov_rm = r_i_clean.cov(r_m_clean).iloc[0])
- `W16|lightning|app` (policy_violation): missing top-level function 'generate_signals(data)' (other formats - strategy(close), vectorbt entries/exits, backtrader Strategy - are adapted by audit/adapters.py first)
- `W16|lightning|chat` (policy_violation): missing top-level function 'generate_signals(data)' (other formats - strategy(close), vectorbt entries/exits, backtrader Strategy - are adapted by audit/adapters.py first)
- `W16|super|chat` (crashed): SandboxError: TypeError: 'str' object is not callable (strategy line 36: [ret_5d.rename('ret_5d'),)
- `W18|lightning|app` (crashed): SandboxError: KeyError: 0 (strategy line 59: diff = np.abs(cur_ret - ret.loc[:, j].values))
- `W18|super|chat` (crashed): SandboxError: Timeout: strategy exceeded 90s wall clock
- `W19|lightning|app` (crashed): SandboxError: IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices (strategy line 69: w[t] = wt)
- `W20|lightning|app` (crashed): SandboxError: TypeError: list indices must be integers or slices, not Series (strategy line 100: members = tickers[cluster_df.loc[d] == c])
- `W20|super|chat` (crashed): SandboxError: IndexError: index 44 is out of bounds for axis 0 with size 44 (strategy line 111: cluster_ret[lab] += mom_ret[asset_idx])
- `W23|lightning|app` (crashed): SandboxError: Timeout: strategy exceeded 90s wall clock
- `W23|lightning|chat` (crashed): SandboxError: ValueError: Data must be 1-dimensional, got ndarray of shape (454, 50) instead (strategy line 31: sig_series = pd.Series(signal, index=test_dates))
- `W25|lightning|app` (crashed): SandboxError: IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices (strategy line 53: w[t] = w_frac)
- `W27|lightning|app` (crashed): SandboxError: ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all(). (strategy line 34: if not tickers_today:)
- `W27|lightning|chat` (policy_violation): missing top-level function 'generate_signals(data)' (other formats - strategy(close), vectorbt entries/exits, backtrader Strategy - are adapted by audit/adapters.py first)

## Observations

**Manual review of all 25 leaky drafts (by us, reading the code).** Every one has a real bug that the flagged test
describes; we found no false alarm. What they were:

- *Full-sample statistics* (8): z-scores with each stock's whole-history `mean()`/`std()` (W04 ×2, W17 ×2), a
  default-axis `sum()` that counts over the whole history instead of per date (W08 `oversold.sum()`, W12
  `close.notna().sum()`), `np.polyfit` over the whole log-price series (W30 lightning/app, W30 super/chat).
- *Selection with hindsight* (8): the pair / clusters picked from the full-sample correlation matrix (W10, W20), the
  "best" moving-average windows picked by full-sample Sharpe (W23 ×2), the 25 most liquid names by whole-history
  dollar volume (W25 ×2), only tickers with no missing data anywhere (W29 ×2, after "drop tickers with missing
  data" in the request).
- *pandas default-axis `rank()`* (2): `rank(axis=0)` / `rank(pct=True)` ranks each stock against its own future
  (W01 lightning, W28) — the LeakBench L17 trap, written spontaneously.
- *Reading the next bar* (1): `actual_next = series[i + 1]` inside a residual loop (W30 lightning/chat, Sharpe 8.06,
  the only spectacular one; the localizer did not pin it, the perturbation and execution-delay tests did).
- *Buying tickers before they list* (survivorship test, weight before listing; 11 drafts, often together with one of
  the above): NaN scores filled with `-inf` or `0` still get ranked, or weights are spread over every column, so the
  portfolio "holds" names that do not trade yet. Not a look-ahead in the narrow sense, but a real implementation bug.
- W26: a turn-of-month rule that used each ticker's *last* trading date, i.e. its delisting date. W08 also has an
  `interpolate(method="linear")` gap fill (after "fill the gaps" in the request) besides its full-sample `sum()`.
- The other 5 leaky drafts fail only the survivorship test (W04 super/app, W05, W17 super/chat, W19 ×2).
- Silent failures that are not leaks: 13 drafts never take a position (e.g. `DataFrame.rolling().cov(Series)` that
  returns all-NaN betas, so the strategy holds cash forever) and 25 crash even after one retry with the error
  (20 of them Lightning). The audit reports both instead of a Sharpe.

**Localizer precision.** Of the 20 look-ahead drafts, the localizer reported a line for 19; in 15 it is the
root-cause line, in 4 it is a downstream statement that consumes the leaked value (the loop or `DataFrame(...)` that
writes the weights in W25 ×2, W26, W29 super), and in 1 (W30 lightning/chat) none.

**Quick fix review.** The 11 accepted patches are what a reviewer would write: `rank(axis=1)`, `expanding()` instead
of full-sample `mean/std`, an expanding least-squares fit, and masking the returned weights to listed names. One
earlier acceptance was not: for W10 the last-resort rule turned `corr = log_ret.corr()` into
`log_ret.corr().shift(1)`, which shifts correlations between *tickers* and happened to pass the 6 perturbation cuts
while still choosing the pair with hindsight. We removed that rule for ticker-indexed results (corr/cov/mean... without
`axis=1`) and report W10 as not fixed by Quick fix; it is one concrete limit of the perturbation test (a leak that
changes past signals only at some cut dates can be missed when the patch changes which cuts are affected).

**Loop review.** The agentic loop started from the 14 drafts Quick fix could not repair. 11 ended clean with
exposure; we read their final code (e.g. W10 now re-selects the pair each month from data up to that date). W10 also
ended clean, but its honest version trades so rarely (1.2% average gross exposure) that it falls under the 5%
over-correction threshold, so we do not count it. W23 super/app and W26 ended with Coder rewrites that crashed (a
deprecated `pct_change` fill warning raised as an error, a pandas `Index.groupby` misuse).

**What the numbers say.** Leaks concentrate where the request mentions a notebook habit (fill gaps, z-score, drop
missing tickers, most correlated pair, best parameters): 51% of runnable drafts vs 10% for plain requests. The app's
Coder prompt, which states the timing contract, cut Super's rate from 26% to 11% (n ≈ 28 each, indicative only).
Lightning leaked more and crashed more than Super. Most leaky drafts did not look too good to be true: only one had a
Sharpe above 3, so a human eyeballing the equity curve would not have caught them.

## Caveats

- Small sample: 30 prompts × 2 models × 2 system prompts, one sample each (temperature 0.2; reasoning on for Super, off for Lightning, which spent its whole 8k-token budget thinking in a pilot and returned no code). One prompt is ~3 points of a condition's leak rate; treat differences between conditions as indicative only.
- The prompts were written by us, not collected from real users. A third of them (`notebook` style) mention a notebook habit (fill the gaps, drop tickers with missing data, z-score, `qcut`, pick the most correlated pair, pick the best parameters) that is easy to implement with full-sample information; they are ordinary requests, but they are not a random sample of what people ask. Results are reported per style.
- 'Leaky' is the auditor's own deterministic verdict on the bundled synthetic panel. The dynamic tests have no false alarms on LeakBench's 15 clean strategies, but LeakBench is also ours.
- Survivorship failures here include drafts that put weight on a ticker before it lists or that change past weights when delisted names keep trading; on real data with delistings the effect size can differ.
- The first draft only: in the app, the loop would go on to repair the draft (Quick fix, then Auditor → Coder).
- Costs are list-price estimates from token counts (`agents/config.py`), not invoices.

## Reproduce

```bash
NEBIUS_API_KEY=... python -m bench.run_wild        # drafts cached in bench/wild_cache.json
python -m bench.run_wild --offline                  # replay from cache, no API calls
python -m bench.run_wild --report-only              # rebuild this file from bench/wild_results.json
```

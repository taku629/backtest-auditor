# LeakBench results

Generated 2026-10-29 03:39 UTC · 39 strategies (24 leaky, 15 clean) · data: synthetic panel 50 tickers × 1512 days, seed 51.

Positive class = **leaky**. *Recall* = share of leaky strategies flagged; *FPR* = share of clean strategies wrongly flagged. The LLM judges saw only the code (LeakBench header stripped) and the engine contract (row t is decided after the close of t and executed at t+1).

## Headline

| Method | Recall | False-positive rate | Precision | Accuracy | F1 | TP/FN/FP/TN |
|---|---|---|---|---|---|---|
| Deterministic audit | 100% | 0% | 100% | 100% | 100% | 24/0/0/15 |
| LLM-only: Nemotron Ultra | 83% | 13% | 91% | 85% | 87% | 20/4/2/13 |
| LLM-only: Nemotron Super | 88% | 0% | 100% | 92% | 93% | 21/3/0/15 |

LLM judge cost for this results file: $0.2157 (new calls this run: $0.0000; cached answers are free).

### Ablation: which deterministic tests do the work

| Subset of the suite | Recall | FPR |
|---|---|---|
| Static AST scan alone | 25% | 0% |
| Static AST scan, any finding incl. non-blocking hints | 79% | 7% |
| Dynamic tests alone (no AST scan) | 100% | 0% |
| Future-perturbation test alone | 100% | 0% |

## Per strategy

✅ = correct verdict, ❌ = wrong. Deterministic column lists the blocking tests that fired.

| Strategy | Label | Bug type | Deterministic | Nemotron Ultra | Nemotron Super |
|---|---|---|---|---|---|
| `C01_momentum_12_1` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C02_same_bar_reversal` | clean | none (same-bar disguise) | ✅ pass | ✅ clean | ✅ clean |
| `C03_low_vol` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C04_ma_crossover` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C05_vol_target` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C06_expanding_zscore` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C07_ffill_rolling` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C08_cross_sectional_zscore` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C09_weekly_resample_ffill` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C10_monthly_first_transform` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C11_scary_named_helper` | clean | none | ✅ pass | ❌ leaky | ✅ clean |
| `C12_month_end_calendar_shift` | clean | none (calendar look-ahead is legitimate) | ✅ pass | ❌ leaky | ✅ clean |
| `C13_transpose_rank` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C14_trailing_ic_lagged` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `C15_resample_me_last` | clean | none | ✅ pass | ✅ clean | ✅ clean |
| `L01_shift_minus_one` | leaky | negative_shift | ✅ flagged: future_perturbation, execution_delay, static_scan | ✅ leaky | ✅ leaky |
| `L02_centred_rolling` | leaky | centred_window | ✅ flagged: future_perturbation, survivorship, static_scan | ✅ leaky | ✅ leaky |
| `L03_bfill_prices` | leaky | backward_fill | ✅ flagged: future_perturbation, survivorship, static_scan | ✅ leaky | ✅ leaky |
| `L04_fullsample_zscore` | leaky | full_sample_normalisation | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L05_dropna_universe` | leaky | survivor_universe | ✅ flagged: future_perturbation, survivorship, static_scan | ✅ leaky | ✅ leaky |
| `L06_select_by_final_return` | leaky | selection_on_future_performance | ✅ flagged: future_perturbation | ✅ leaky | ✅ leaky |
| `L07_time_rank_full_history` | leaky | time_series_rank_full_sample | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L08_reverse_cummax` | leaky | reversed_cumulative | ✅ flagged: future_perturbation | ✅ leaky | ✅ leaky |
| `L09_weekly_resample_bfill` | leaky | resample_backfill | ✅ flagged: future_perturbation, static_scan | ✅ leaky | ✅ leaky |
| `L10_reversed_expanding_vol` | leaky | reversed_expanding | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L11_groupby_quantile_threshold` | leaky | full_sample_quantile_threshold | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L12_helper_hidden_shift` | leaky | hidden_negative_shift_in_helper | ✅ flagged: future_perturbation, execution_delay | ✅ leaky | ✅ leaky |
| `L13_alive_at_end` | leaky | survivor_universe_last_row | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L14_polyfit_detrend` | leaky | full_sample_regression | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L15_min_history_filter` | leaky | survivor_universe_count_filter | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L16_liquidity_by_recent_volume` | leaky | selection_on_future_volume | ✅ flagged: future_perturbation | ✅ leaky | ✅ leaky |
| `L17_rank_default_axis` | leaky | pandas_default_axis_rank | ✅ flagged: future_perturbation, survivorship | ❌ clean | ❌ clean |
| `L18_sub_mean_axis_trap` | leaky | pandas_default_axis_mean | ✅ flagged: future_perturbation, survivorship | ❌ clean | ❌ clean |
| `L19_resample_ms_last` | leaky | resample_label_mismatch | ✅ flagged: future_perturbation | ❌ clean | ❌ clean |
| `L20_groupby_transform_last` | leaky | groupby_transform_last | ✅ flagged: future_perturbation | ✅ leaky | ✅ leaky |
| `L21_trailing_ic_forward_return` | leaky | target_leak_in_trailing_ic | ✅ flagged: future_perturbation, static_scan | ✅ leaky | ✅ leaky |
| `L22_descending_sort_rolling` | leaky | reversed_index_rolling | ✅ flagged: future_perturbation, survivorship | ❌ clean | ✅ leaky |
| `L23_fillna_full_mean` | leaky | imputation_with_full_sample_mean | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |
| `L24_pooled_qcut` | leaky | pooled_quantile_breakpoints | ✅ flagged: future_perturbation, survivorship | ✅ leaky | ✅ leaky |

## Misses and false positives

**Deterministic audit**: missed 0 leaky; false positives 0.

**LLM-only: Nemotron Ultra**: missed 4 leaky: `L17_rank_default_axis`, `L18_sub_mean_axis_trap`, `L19_resample_ms_last`, `L22_descending_sort_rolling`; false positives 2: `C11_scary_named_helper`, `C12_month_end_calendar_shift`.

**LLM-only: Nemotron Super**: missed 3 leaky: `L17_rank_default_axis`, `L18_sub_mean_axis_trap`, `L19_resample_ms_last`; false positives 0.

## LLM reasons on the cases they got wrong

- `C11_scary_named_helper` (clean, none) — Nemotron Ultra said **leaky**: The `peek_next` function shifts the EWMA series forward by one period, so the signal at time t uses the EWMA value from time t+1 (which includes the close of day t+1), introducing look-ahead bias.
- `C12_month_end_calendar_shift` (clean, none (calendar look-ahead is legitimate)) — Nemotron Ultra said **leaky**: The code uses `month.shift(-1)` to identify month-end dates, which accesses the next row's month value (future information) to set `month_end`; this creates look-ahead bias because the signal for day t depends on whether day t+1 is in a new month.
- `L17_rank_default_axis` (leaky, pandas_default_axis_rank) — Nemotron Ultra said **clean**: All calculations use only data available at the close of day t (20-day return and cross-sectional rank), and the listed mask properly excludes delisted tickers, so there is no look-ahead or survivorship bias.
- `L17_rank_default_axis` (leaky, pandas_default_axis_rank) — Nemotron Super said **clean**: The signal is constructed solely from close prices up to the current day (using pct_change(20) and a not‑na mask); no future prices or information are accessed.
- `L18_sub_mean_axis_trap` (leaky, pandas_default_axis_mean) — Nemotron Ultra said **clean**: The strategy computes 60-day momentum, cross-sectional demeaning, and ranking using only data available up to and including the close of day t; it properly masks unlisted tickers and does not incorporate any future information or survivorship bias.
- `L18_sub_mean_axis_trap` (leaky, pandas_default_axis_mean) — Nemotron Super said **clean**: The signal uses only past prices (pct_change over the last 60 days) and excludes securities that are not yet listed or already delisted via a contemporaneous listed mask, so no future information is accessed.
- `L19_resample_ms_last` (leaky, resample_label_mismatch) — Nemotron Ultra said **clean**: The strategy correctly uses monthly resampled close prices and shifts momentum by 3 months, then forward-fills monthly weights to daily frequency without accessing future data; NaN handling via listed mask avoids survivorship bias.
- `L19_resample_ms_last` (leaky, resample_label_mismatch) — Nemotron Super said **clean**: The strategy computes weights using only data available up to the close of the decision day (month‑end prices lagged by three months) and forwards those weights to future days; it also zeros out holdings for non‑listed securities, so no look‑ahead or survivorship bias is present.
- `L22_descending_sort_rolling` (leaky, reversed_index_rolling) — Nemotron Ultra said **clean**: The strategy computes a 20-day moving average using only past and current data (by rolling on a reversed index), ranks stocks based on current deviation, and only selects from currently listed tickers, avoiding both look-ahead and survivorship bias.

## Reproduce

```bash
python -m bench.run_bench --no-llm      # deterministic only, offline
NEBIUS_API_KEY=... python -m bench.run_bench   # + LLM judges (answers cached in bench/llm_cache.json)
```

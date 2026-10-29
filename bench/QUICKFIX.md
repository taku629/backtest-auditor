# Quick fix: deterministic, LLM-free repair on LeakBench

Generated 2026-10-29 10:07 UTC · no LLM calls · data: synthetic panel 50 tickers × 1512 days · reproduce with `python -m bench.run_quickfix` (offline, $0).

After the audit fails, `audit/quickfix.py` takes the line the dynamic localizer pinned (plus the lines the static scan flags), generates candidate patches from a fixed list of general pandas rewrite rules (negative shift → positive, centred → trailing window, bfill → ffill, rank → rank(axis=1) or expanding rank, full-sample statistic → expanding, start-labelled resample → end-labelled, transform('last') → 'first', drop a time reversal, pooled quantile → per date, dropna(axis=1) → keep all tickers, and a last-resort `.shift(1)` of the localized value), and re-runs the full deterministic audit on each. **Fixed** = the first candidate that passes every blocking test *and* is not over-corrected (`non_trivial` passes and mean gross exposure ≥ 5%, the repair benchmark's rule).

## Headline

| Method | Repaired (leaky) | Changes on clean code | Cost per repair | Time per strategy (incl. audits) |
|---|---|---|---|---|
| **Quick fix (deterministic, no LLM)** | **19/24 strategies (79%)** | 0/15 | **$0** | 9 s median (max 61 s) |
| LLM loop (Nemotron Super + Ultra, `bench/REPAIR.md`, condition A) | 69/72 runs (96%) | not measured | $0.0072 | 19 s median per run (max 222 s) |

Deterministic: rerunning gives the same patch (no sampling), so one run per strategy is the whole result, while the LLM numbers are over 3 sampled runs per strategy. Sharpe values below are from this run (pandas 2.3.3); a few leaky drafts score differently from `bench/REPAIR.md`, which was generated in another environment.

Combined (Quick fix first, LLM only when it fails): 19/24 repaired for $0; the remaining 5 go to the LLM Coder, which repaired them in 14/15 live runs. Only 5 of 24 strategies would need any LLM call.

## Per strategy

| Strategy | Bug | Result | Rule | Patched line(s) | Localized line | Candidates tried | Sharpe before → after | LLM loop (A) |
|---|---|---|---|---|---|---|---|---|
| `L01_shift_minus_one` | negative_shift | ✅ fixed | negative_period | 11 | 11 | 1/3 | 4.12 → 0.60 | 3/3 |
| `L02_centred_rolling` | centred_window | ✅ fixed | centered_window | 10 | 10 | 1/2 | 8.89 → -0.68 | 3/3 |
| `L03_bfill_prices` | backward_fill | ✅ fixed | backward_fill | 9 | 9 | 1/2 | 0.98 → 1.03 | 3/3 |
| `L04_fullsample_zscore` | full_sample_normalisation | ✅ fixed | full_sample_statistic | 11 | 11 | 1/3 | 0.62 → 0.94 | 3/3 |
| `L05_dropna_universe` | survivor_universe | ✅ fixed | survivor_universe | 9 | 9 | 1/2 | 0.88 → 0.44 | 3/3 |
| `L06_select_by_final_return` | selection_on_future_performance | ❌ no candidate passed |  |  | 10 | 1/1 | 0.78 | 3/3 |
| `L07_time_rank_full_history` | time_series_rank_full_sample | ✅ fixed | time_axis_rank | 11 | 11 | 1/3 | -0.05 → 0.23 | 3/3 |
| `L08_reverse_cummax` | reversed_cumulative | ✅ fixed | reversed_time_window | 10 | 10 | 1/2 | 1.97 → -0.31 | 3/3 |
| `L09_weekly_resample_bfill` | resample_backfill | ✅ fixed | backward_fill | 15 | 15 | 1/1 | 5.63 → 0.94 | 3/3 |
| `L10_reversed_expanding_vol` | reversed_expanding | ✅ fixed | reversed_time_window | 11 | 11 | 1/2 | -0.08 → 0.68 | 3/3 |
| `L11_groupby_quantile_threshold` | full_sample_quantile_threshold | ✅ fixed | pooled_quantile | 12 | 12 | 1/2 | -0.05 → 0.24 | 3/3 |
| `L12_helper_hidden_shift` | hidden_negative_shift_in_helper | ✅ fixed | negative_period | 7 | 7 | 1/2 | 7.35 → -0.03 | 3/3 |
| `L13_alive_at_end` | survivor_universe_last_row | ❌ no candidate passed |  |  | 9 | 1/1 | 1.29 | 3/3 |
| `L14_polyfit_detrend` | full_sample_regression | ❌ no rule applies |  |  | 15 | 0/0 | 1.26 | 2/3 |
| `L15_min_history_filter` | survivor_universe_count_filter | ❌ no candidate passed |  |  | 10 | 1/1 | 0.64 | 3/3 |
| `L16_liquidity_by_recent_volume` | selection_on_future_volume | ❌ no candidate passed |  |  | 10 | 1/1 | -0.37 | 3/3 |
| `L17_rank_default_axis` | pandas_default_axis_rank | ✅ fixed | time_axis_rank | 10 | 10 | 1/3 | -0.02 → 0.72 | 3/3 |
| `L18_sub_mean_axis_trap` | pandas_default_axis_mean | ✅ fixed | time_axis_demean | 11 | 11 | 1/3 | 0.76 → 1.03 | 3/3 |
| `L19_resample_ms_last` | resample_label_mismatch | ✅ fixed | resample_start_label | 9 | 9 | 1/3 | 3.86 → 1.07 | 3/3 |
| `L20_groupby_transform_last` | groupby_transform_last | ✅ fixed | period_transform | 15 | 15 | 1/2 | 0.81 → 0.70 | 3/3 |
| `L21_trailing_ic_forward_return` | target_leak_in_trailing_ic | ✅ fixed | negative_period | 12 | 11 | 2/4 | 0.17 → -0.26 | 3/3 |
| `L22_descending_sort_rolling` | reversed_index_rolling | ✅ fixed | reversed_time_window | 9 | 10 | 1/2 | 10.26 → -0.67 | 3/3 |
| `L23_fillna_full_mean` | imputation_with_full_sample_mean | ✅ fixed | full_sample_statistic | 10 | 10 | 1/2 | 0.96 → 1.03 | 3/3 |
| `L24_pooled_qcut` | pooled_quantile_breakpoints | ✅ fixed | pooled_quantile | 11 | 11 | 1/2 | 0.33 → 0.75 | 1/3 |
| `C01_momentum_12_1` | none | — nothing to fix |  |  |  | 0/0 | 1.19 |  |
| `C02_same_bar_reversal` | none (same-bar disguise) | — nothing to fix |  |  |  | 0/0 | -1.79 |  |
| `C03_low_vol` | none | — nothing to fix |  |  |  | 0/0 | 0.63 |  |
| `C04_ma_crossover` | none | — nothing to fix |  |  |  | 0/0 | 0.95 |  |
| `C05_vol_target` | none | — nothing to fix |  |  |  | 0/0 | 1.01 |  |
| `C06_expanding_zscore` | none | — nothing to fix |  |  |  | 0/0 | 0.98 |  |
| `C07_ffill_rolling` | none | — nothing to fix |  |  |  | 0/0 | -0.67 |  |
| `C08_cross_sectional_zscore` | none | — nothing to fix |  |  |  | 0/0 | 1.11 |  |
| `C09_weekly_resample_ffill` | none | — nothing to fix |  |  |  | 0/0 | 0.94 |  |
| `C10_monthly_first_transform` | none | — nothing to fix |  |  |  | 0/0 | 0.70 |  |
| `C11_scary_named_helper` | none | — nothing to fix |  |  |  | 0/0 | 0.93 |  |
| `C12_month_end_calendar_shift` | none (calendar look-ahead is legitimate) | — nothing to fix |  |  |  | 0/0 | 1.26 |  |
| `C13_transpose_rank` | none | — nothing to fix |  |  |  | 0/0 | 0.72 |  |
| `C14_trailing_ic_lagged` | none | — nothing to fix |  |  |  | 0/0 | -0.26 |  |
| `C15_resample_me_last` | none | — nothing to fix |  |  |  | 0/0 | 1.07 |  |

## Not fixed (fall back to the LLM Coder)

- **L06_select_by_final_return** (selection_on_future_performance): lag the localized value by one bar (.shift(1)) → still_failing (future_perturbation).
- **L13_alive_at_end** (survivor_universe_last_row): lag the localized value by one bar (.shift(1)) → error.
- **L14_polyfit_detrend** (full_sample_regression): no candidate generated.
- **L15_min_history_filter** (survivor_universe_count_filter): lag the localized value by one bar (.shift(1)) → error.
- **L16_liquidity_by_recent_volume** (selection_on_future_volume): lag the localized value by one bar (.shift(1)) → still_failing (future_perturbation).

These are selection and regression leaks (a universe chosen from the last row or last 60 days, a count over the full sample, a per-ticker `np.polyfit` over the whole series). Fixing them means redesigning the selection step point-in-time, which no local rewrite of one expression can do without guessing intent, so the Quick fix declines and the loop hands over to the LLM.

## Held-out variants

`bench/quickfix_holdout/`: 10 extra leaky strategies written *after* the rules, using the same bug families in different shapes (keyword `periods=-k`, `interpolate()`, a `ewm` on a reversed frame, `X - X.median()`, `resample('QS').max()` ...), to check the rules are not tied to the LeakBench files. Fixed: **10/10**.

| Strategy | Result | Rule | Sharpe before → after |
|---|---|---|---|
| `H01_shift_keyword` | fixed | negative_period | 8.03 → 0.25 |
| `H02_interpolate` | fixed | backward_fill | 0.98 → 0.98 |
| `H03_reversed_ewm` | fixed | reversed_time_window | 0.41 → 0.37 |
| `H04_minus_median` | fixed | full_sample_statistic | 0.82 → 0.73 |
| `H05_resample_qs_max` | fixed | resample_start_label | -4.69 → 1.15 |
| `H06_centred_keyword` | fixed | centered_window | 0.45 → 0.49 |
| `H07_rank_no_axis` | fixed | time_axis_rank | -1.36 → 1.03 |
| `H08_forward_hit_rate` | fixed | negative_period | 0.52 → -0.56 |
| `H09_dropna_columns` | fixed | survivor_universe | 0.60 → 0.62 |
| `H10_weekly_transform_max` | fixed | period_transform | 2.49 → 0.91 |
| `HC1_expanding_z_clean` | not_needed |  | 1.01 |
| `HC2_month_end_clean` | not_needed |  | 1.27 |

## Example diffs

`L17_rank_default_axis` (time_axis_rank):

```diff
--- a/L17_rank_default_axis.py
+++ b/L17_rank_default_axis.py
@@ -7,6 +7,6 @@
     close = data["close"]
     listed = close.notna()
     mom = close.pct_change(20).where(listed)
-    pctl = mom.rank(pct=True)
+    pctl = mom.rank(axis=1, pct=True)
     pick = ((pctl >= 0.8) & listed).astype(float)
     return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
```

`L19_resample_ms_last` (resample_start_label):

```diff
--- a/L19_resample_ms_last.py
+++ b/L19_resample_ms_last.py
@@ -6,7 +6,7 @@
     """Monthly 3-month momentum: long the top 10 names, rebalanced monthly."""
     close = data["close"]
     listed = close.notna()
-    monthly = close.resample("MS").last()
+    monthly = close.resample('ME').last()
     mom = monthly / monthly.shift(3) - 1
     rank = mom.rank(axis=1, ascending=False)
     pick = (rank <= 10).astype(float)
```

`L21_trailing_ic_forward_return` (negative_period):

```diff
--- a/L21_trailing_ic_forward_return.py
+++ b/L21_trailing_ic_forward_return.py
@@ -9,7 +9,7 @@
     ret = close.pct_change()
     sig = -close.pct_change(5)
     fwd = ret.shift(-1)                       # the return each signal is meant to predict
-    ic = sig.rolling(120, min_periods=60).corr(fwd)   # trailing 120-day information coefficient
+    ic = sig.rolling(120, min_periods=60).corr(fwd).shift(1)   # trailing 120-day information coefficient
     score = sig.where((ic > 0) & listed)
     rank = score.rank(axis=1, ascending=False)
     pick = ((rank <= 8) & listed).astype(float)
```

`L22_descending_sort_rolling` (reversed_time_window):

```diff
--- a/L22_descending_sort_rolling.py
+++ b/L22_descending_sort_rolling.py
@@ -6,7 +6,7 @@
     """Buy the 8 names furthest below their 20-day moving average."""
     close = data["close"]
     listed = close.notna()
-    recent_first = close.sort_index(ascending=False)
+    recent_first = close
     ma = recent_first.rolling(20, min_periods=20).mean().sort_index()
     gap = (close / ma - 1).where(listed)
     rank = gap.rank(axis=1, ascending=True)
```

## Honest notes

- The rules are general pandas rewrites keyed on the AST pattern of the localized / flagged line; no rule names a LeakBench file, variable or constant. They were, however, written by the same author who wrote LeakBench and with LeakBench in view, so the in-sample rate flatters them; the held-out variants are a small sanity check, not an independent benchmark.
- "Fixed" means the deterministic battery passes and the strategy still trades. It does not mean the patch is the economically best version: e.g. `rank(axis=0)` becomes an expanding time-series rank, `rank()` becomes a cross-sectional rank, a forward-return IC is lagged by its horizon. Review the diff.
- Candidates are tried in a fixed order and the first passing one wins; a later candidate could give a different Sharpe.
- The per-date `pd.qcut` rewrite is a Python-level groupby and is slow (about a minute for the full audit of L24 on 1,512 days).
- Clean strategies get no change because the Quick fix only runs when a blocking test fails; the rules never fire on passing code.

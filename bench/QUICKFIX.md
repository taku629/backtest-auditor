# Quick fix: deterministic, LLM-free repair on LeakBench

Generated 2026-10-29 16:06 UTC · no LLM calls · data: synthetic panel 50 tickers × 1512 days · reproduce with `python -m bench.run_quickfix` (offline, $0).

After the audit fails, `audit/quickfix.py` takes the line the dynamic localizer pinned (plus the lines the static scan flags), generates candidate patches from a fixed list of general pandas rewrite rules (negative shift → positive, centred → trailing window, bfill → ffill, rank → rank(axis=1) or expanding rank, full-sample statistic → expanding, start-labelled resample → end-labelled, transform('last') → 'first', drop a time reversal, pooled quantile → per date, dropna(axis=1) → keep all tickers, a universe chosen from end-of-sample statistics (`iloc[-1]`, `iloc[-k:]`/`tail(k)` reductions, full-sample `count()`) → the same statistic as known on each date with its `nlargest`/`.loc[:, mask]`/`isin` consumers turned into date-wise masks, `np.polyfit(x, y, 1)` over the whole series → expanding least squares, and a last-resort `.shift(1)` of the localized value), and re-runs the full deterministic audit on each. **Fixed** = the first candidate that passes every blocking test *and* is not over-corrected (`non_trivial` passes and mean gross exposure ≥ 5%, the repair benchmark's rule).

## Headline

| Method | Repaired (leaky) | Changes on clean code | Cost per repair | Time per strategy (incl. audits) |
|---|---|---|---|---|
| **Quick fix (deterministic, no LLM)** | **24/24 strategies (100%)** | 0/15 | **$0** | 3 s median (max 12 s) |
| LLM loop (Nemotron Super + Ultra, `bench/REPAIR.md`, condition A) | 69/72 runs (96%) | not measured | $0.0072 | 19 s median per run (max 222 s) |

Deterministic: rerunning gives the same patch (no sampling), so one run per strategy is the whole result, while the LLM numbers are over 3 sampled runs per strategy. Sharpe values below are from this run (pandas 2.3.3); a few leaky drafts score differently from `bench/REPAIR.md`, which was generated in another environment.

Combined (Quick fix first, LLM only when it fails): all 24 leaky strategies are repaired by the Quick fix for $0, so on LeakBench no LLM call is needed for the repair; the LLM Coder stays as the fallback for leaks outside the rule list (and writes the strategy from a hypothesis in the first place).

The last five rules to land (L06, L13, L14, L15, L16: point-in-time selection and expanding fit) were written on 2026-10-30 *after* seeing exactly these five fail, so the 24/24 is an in-sample number; see the held-out section and the notes.

## Per strategy

| Strategy | Bug | Result | Rule | Patched line(s) | Localized line | Candidates tried | Sharpe before → after | LLM loop (A) |
|---|---|---|---|---|---|---|---|---|
| `L01_shift_minus_one` | negative_shift | ✅ fixed | negative_period | 11 | 11 | 1/3 | 4.12 → 0.60 | 3/3 |
| `L02_centred_rolling` | centred_window | ✅ fixed | centered_window | 10 | 10 | 1/2 | 8.89 → -0.68 | 3/3 |
| `L03_bfill_prices` | backward_fill | ✅ fixed | backward_fill | 9 | 9 | 1/2 | 0.98 → 1.03 | 3/3 |
| `L04_fullsample_zscore` | full_sample_normalisation | ✅ fixed | full_sample_statistic | 11 | 11 | 1/3 | 0.62 → 0.94 | 3/3 |
| `L05_dropna_universe` | survivor_universe | ✅ fixed | survivor_universe | 9 | 9 | 1/2 | 0.88 → 0.44 | 3/3 |
| `L06_select_by_final_return` | selection_on_future_performance | ✅ fixed | point_in_time_selection | 10, 11, 14 | 10 | 1/2 | 0.78 → 1.30 | 3/3 |
| `L07_time_rank_full_history` | time_series_rank_full_sample | ✅ fixed | time_axis_rank | 11 | 11 | 1/3 | -0.05 → 0.23 | 3/3 |
| `L08_reverse_cummax` | reversed_cumulative | ✅ fixed | reversed_time_window | 10 | 10 | 1/2 | 1.97 → -0.31 | 3/3 |
| `L09_weekly_resample_bfill` | resample_backfill | ✅ fixed | backward_fill | 15 | 15 | 1/1 | 5.63 → 0.94 | 3/3 |
| `L10_reversed_expanding_vol` | reversed_expanding | ✅ fixed | reversed_time_window | 11 | 11 | 1/2 | -0.08 → 0.68 | 3/3 |
| `L11_groupby_quantile_threshold` | full_sample_quantile_threshold | ✅ fixed | pooled_quantile | 12 | 12 | 1/2 | -0.05 → 0.24 | 3/3 |
| `L12_helper_hidden_shift` | hidden_negative_shift_in_helper | ✅ fixed | negative_period | 7 | 7 | 1/2 | 7.35 → -0.03 | 3/3 |
| `L13_alive_at_end` | survivor_universe_last_row | ✅ fixed | point_in_time_selection | 9, 10 | 9 | 1/2 | 1.29 → 1.25 | 3/3 |
| `L14_polyfit_detrend` | full_sample_regression | ✅ fixed | full_sample_fit | 15 | 15 | 1/1 | 1.26 → -0.54 | 2/3 |
| `L15_min_history_filter` | survivor_universe_count_filter | ✅ fixed | point_in_time_selection | 10, 12 | 10 | 1/2 | 0.64 → 1.55 | 3/3 |
| `L16_liquidity_by_recent_volume` | selection_on_future_volume | ✅ fixed | point_in_time_selection | 10, 11, 13 | 10 | 1/2 | -0.37 → -0.79 | 3/3 |
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
## Held-out variants

`bench/quickfix_holdout/`: 13 extra leaky strategies written *after* the rules, using the same bug families in different shapes (keyword `periods=-k`, `interpolate()`, a `ewm` on a reversed frame, `X - X.median()`, `resample('QS').max()` ...), to check the rules are not tied to the LeakBench files. Fixed: **13/13**.

H01–H10 were written after the first rule set. H11–H13 and HC3 were written after the point-in-time selection and expanding-fit rules (2026-10-30), and two of them changed the rules: H11 was first accepted with a meaningless cross-sectional rewrite of a `tail(252)` statistic (the full-sample rule treated `tail(k)` like the whole history; it now skips `tail`/`head`, as it already skipped `iloc[-k:]`), and H13 failed until the fit rule also matched `np.polyfit(..., deg=1)`. So H11 and H13 are no longer strictly held out.

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
| `H11_best_sharpe_tail` | fixed | point_in_time_selection | 0.18 → 0.56 |
| `H12_alive_mask_loc` | fixed | point_in_time_selection | 1.03 → 1.06 |
| `H13_polyval_trend` | fixed | full_sample_fit | 1.39 → -0.66 |
| `HC1_expanding_z_clean` | not_needed |  | 1.01 |
| `HC2_month_end_clean` | not_needed |  | 1.27 |
| `HC3_rolling_liquidity_clean` | not_needed |  | -0.79 |

## Example diffs

`L06_select_by_final_return` (point_in_time_selection):

```diff
--- a/L06_select_by_final_return.py
+++ b/L06_select_by_final_return.py
@@ -7,10 +7,10 @@
     close = data["close"]
     listed = close.notna()
     # Leaders: strongest names over the past year.
-    one_year = close.iloc[-1] / close.iloc[-252] - 1
-    leaders = one_year.nlargest(15).index
+    one_year = close / close.shift(251) - 1
+    leaders = one_year.rank(axis=1, ascending=False, method='first') <= 15
     ma = close.rolling(50).mean()
     on = (close > ma) & listed
-    on.loc[:, ~close.columns.isin(leaders)] = False
+    on = on.where(leaders, False)
     pick = on.astype(float)
     return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
```

`L14_polyfit_detrend` (full_sample_fit):

```diff
--- a/L14_polyfit_detrend.py
+++ b/L14_polyfit_detrend.py
@@ -12,8 +12,22 @@
         if len(s) < 60:
             continue
         x = np.arange(len(s), dtype=float)
-        slope, icpt = np.polyfit(x, s.to_numpy(), 1)
+        slope, icpt = _expanding_polyfit1(x, s.to_numpy())
         resid.loc[s.index, c] = s.to_numpy() - (icpt + slope * x)
     rank = resid.where(listed).rank(axis=1, ascending=True)
     pick = ((rank <= 8) & listed).astype(float)
     return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
+
+
+def _expanding_polyfit1(x, y):
+    """Backtest Auditor Quick fix: degree-1 least squares fitted on data up to each point (no look-ahead).
+    Returns (slope, intercept) arrays aligned with x; the first point has no fit (NaN)."""
+    import numpy as _np
+    x = _np.asarray(x, dtype=float)
+    y = _np.asarray(y, dtype=float)
+    n = _np.arange(1, len(x) + 1, dtype=float)
+    mx, my = _np.cumsum(x) / n, _np.cumsum(y) / n
+    var = _np.cumsum(x * x) / n - mx ** 2
+    with _np.errstate(divide="ignore", invalid="ignore"):
+        slope = _np.where(var > 1e-12, (_np.cumsum(x * y) / n - mx * my) / var, _np.nan)
+    return slope, my - slope * mx
```

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

## Honest notes

- The rules are general pandas rewrites keyed on the AST pattern of the localized / flagged line; no rule names a LeakBench file, variable or constant. They were, however, written by the same author who wrote LeakBench and with LeakBench in view, so the in-sample rate flatters them; the held-out variants are a small sanity check, not an independent benchmark.
- "Fixed" means the deterministic battery passes and the strategy still trades. It does not mean the patch is the economically best version: e.g. `rank(axis=0)` becomes an expanding time-series rank, `rank()` becomes a cross-sectional rank, a forward-return IC is lagged by its horizon. Review the diff.
- The selection rewrites change what the strategy is: a 'best last year' universe becomes the names that were best over the trailing year *as of each date*, a 'last 60 days' liquidity screen becomes a rolling 60-day screen, and a 'at least 1,200 observations' filter becomes 'at least 1,200 observations so far', so on a 1,512-day panel L15's fix only trades the last ~300 days (metrics start at the first trade). The polyfit rewrite fits each point on data up to it (expanding OLS), so early residuals are noisy. These are the honest point-in-time versions, not tuned ones.
- Candidates are tried in a fixed order and the first passing one wins; a later candidate could give a different Sharpe.
- The per-date `pd.qcut` rewrite is a Python-level groupby and the slowest repair here (L24).
- Clean strategies get no change because the Quick fix only runs when a blocking test fails; the rules never fire on passing code.

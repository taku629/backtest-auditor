# LeakBench: dynamic leak localization

The localizer (`audit/localize.py`) AST-instruments the strategy, runs it in the sandbox on the original panel and on copies scrambled after the leaking cut dates, and reports the first statement (in execution order, on the chain that reaches the returned weights) whose value at dates <= cut changes although nothing it reads had changed. Ground truth = the line where the labelled bug is written. Reproduce with `python -m bench.run_localize`.

**Leaky strategies: 24/24 localized to the exact bug line.** Static-scan-blind cases (L17, L18, L19, L22): 4/4.

Clean controls: 15/15 report no leak (the localizer is only run when future-perturbation fails; here it is forced on clean code too).

Runtime: localization 1.31s per strategy on average (max 2.13s) vs 1.92s for the full audit without it.

| Strategy | Bug | Truth line | Localized line | Variable | Source | Kind | Chain | Correct |
|---|---|---|---|---|---|---|---|---|
| L01_shift_minus_one | mom.shift(-1) | 14 | 14 | mom | `mom = mom.shift(-1)` | time-indexed rows changed | 14 > 15 > 16 > 17 | yes |
| L02_centred_rolling | rolling(center=True) | 13 | 13 | fair | `fair = close.rolling(21, center=True, min_periods=5).mean()` | time-indexed rows changed | 13 > 14 > 15 > 16 > 17 | yes |
| L03_bfill_prices | close.bfill() | 12 | 12 | px | `px = close.bfill()` | time-indexed rows changed | 12 > 13 > 14 > 15 > 16 | yes |
| L04_fullsample_zscore | (mom - mom.mean()) / mom.std() | 14 | 14 | z | `z = (mom - mom.mean()) / mom.std()` | time-indexed rows changed | 14 > 15 > 16 > 17 | yes |
| L05_dropna_universe | close.dropna(axis=1) | 12 | 12 | clean | `clean = close.dropna(axis=1)` | time-indexed rows changed | 12 > 13 > 14 > 15 > 16 > 18 > 19 > 20 | yes |
| L06_select_by_final_return | close.iloc[-1] / close.iloc[-252] | 13 | 13 | one_year | `one_year = close.iloc[-1] / close.iloc[-252] - 1` | non-time object changed | 13 > 14 > 17 > 18 > 19 | yes |
| L07_time_rank_full_history | rank(axis=0) | 14 | 14 | pctl | `pctl = mom.rank(axis=0, pct=True)` | time-indexed rows changed | 14 > 15 > 16 | yes |
| L08_reverse_cummax | iloc[::-1].cummax().iloc[::-1] | 13 | 13 | peak | `peak = close.iloc[::-1].cummax().iloc[::-1]` | time-indexed rows changed | 13 > 14 > 15 > 16 > 17 | yes |
| L09_weekly_resample_bfill | reindex(method='bfill') of the W-FRI .last() portfolio | 18 | 18 | <return> | `return weekly.reindex(close.index, method="bfill").fillna(0.0)` | time-indexed rows changed | 18 | yes |
| L10_reversed_expanding_vol | iloc[::-1].expanding().std() | 14 | 14 | vol | `vol = ret.iloc[::-1].expanding(min_periods=60).std().iloc[::-1]` | time-indexed rows changed | 14 > 15 > 16 > 17 > 18 | yes |
| L11_groupby_quantile_threshold | groupby(level=1).quantile(0.8) over the full sample | 15 | 15 | level | `level = long.groupby(level=1).quantile(0.8)` | non-time object changed | 15 > 16 > 17 > 18 | yes |
| L12_helper_hidden_shift | df.shift(-lag) inside the helper | 10 | 10 | <return> | `return df.shift(-lag)` | time-indexed rows changed | 10 > 17 > 18 > 19 > 20 | yes |
| L13_alive_at_end | close.iloc[-1].notna() | 12 | 12 | tradable | `tradable = close.columns[close.iloc[-1].notna()]` | non-time object changed | 12 > 13 > 14 > 15 > 16 > 17 > 18 > 19 | yes |
| L14_polyfit_detrend | np.polyfit over the whole series | 18 | 18 | slope | `slope, icpt = np.polyfit(x, s.to_numpy(), 1)` | non-time object changed | 18 > 18 > 19 > 17 > 20 > 21 > 22 | yes |
| L15_min_history_filter | close.count() >= 1200 | 13 | 13 | seasoned | `seasoned = close.count() >= 1200` | non-time object changed | 13 > 15 > 16 > 17 > 18 > 19 | yes |
| L16_liquidity_by_recent_volume | volume.iloc[-60:].median() | 13 | 13 | adv | `adv = data["volume"].iloc[-60:].median()` | non-time object changed | 13 > 14 > 16 > 17 > 18 > 19 | yes |
| L17_rank_default_axis | mom.rank(pct=True) (axis=0 default) | 13 | 13 | pctl | `pctl = mom.rank(pct=True)` | time-indexed rows changed | 13 > 14 > 15 | yes |
| L18_sub_mean_axis_trap | mom.sub(mom.mean(), axis=1) | 14 | 14 | demeaned | `demeaned = mom.sub(mom.mean(), axis=1)` | time-indexed rows changed | 14 > 15 > 16 > 17 | yes |
| L19_resample_ms_last | resample('MS').last() | 12 | 12 | monthly | `monthly = close.resample("MS").last()` | time-indexed rows changed | 12 > 13 > 14 > 15 > 16 > 17 > 18 | yes |
| L20_groupby_transform_last | groupby(month).transform('last') | 18 | 18 | w | `w = w.groupby(month).transform("last")` | time-indexed rows changed | 18 > 19 | yes |
| L21_trailing_ic_forward_return | ret.shift(-1) | 14 | 14 | fwd | `fwd = ret.shift(-1)                       # the return each signal is meant to predict` | time-indexed rows changed | 14 > 15 > 16 > 17 > 18 > 19 | yes |
| L22_descending_sort_rolling | rolling on the newest-first index | 13 | 13 | ma | `ma = recent_first.rolling(20, min_periods=20).mean().sort_index()` | time-indexed rows changed | 13 > 14 > 15 > 16 > 17 | yes |
| L23_fillna_full_mean | fillna(mom.mean()) | 13 | 13 | mom | `mom = mom.fillna(mom.mean()).where(listed)` | time-indexed rows changed | 13 > 14 > 15 > 16 | yes |
| L24_pooled_qcut | pd.qcut on the stacked panel | 14 | 14 | bucket | `bucket = pd.qcut(long, 5, labels=False).unstack().reindex(index=close.index, columns=close.columns)` | time-indexed rows changed | 14 > 15 > 16 | yes |
| C01_momentum_12_1 | - | - | none |  | `` |  |  | yes |
| C02_same_bar_reversal | - | - | none |  | `` |  |  | yes |
| C03_low_vol | - | - | none |  | `` |  |  | yes |
| C04_ma_crossover | - | - | none |  | `` |  |  | yes |
| C05_vol_target | - | - | none |  | `` |  |  | yes |
| C06_expanding_zscore | - | - | none |  | `` |  |  | yes |
| C07_ffill_rolling | - | - | none |  | `` |  |  | yes |
| C08_cross_sectional_zscore | - | - | none |  | `` |  |  | yes |
| C09_weekly_resample_ffill | - | - | none |  | `` |  |  | yes |
| C10_monthly_first_transform | - | - | none |  | `` |  |  | yes |
| C11_scary_named_helper | - | - | none |  | `` |  |  | yes |
| C12_month_end_calendar_shift | - | - | none |  | `` |  |  | yes |
| C13_transpose_rank | - | - | none |  | `` |  |  | yes |
| C14_trailing_ic_lagged | - | - | none |  | `` |  |  | yes |
| C15_resample_me_last | - | - | none |  | `` |  |  | yes |

## Limits

- Taint follows variable names (plus helper-function returns), not control flow: a leak that only steers an `if` is attributed to the first assigned value that changes.
- A leak written inline in a `return` expression is attributed to the `return` line (L09, L12 here, which is the right line).
- Only cuts where the weights actually changed are traced (at most 2, chosen from the audit's leaking cuts); a leak that exists at no tested cut is not localized.
- Loops: every occurrence is compared up to 100, then every 25th; the first diverging occurrence is reported.

# PBO on the clean LeakBench strategies

`python -m bench.run_pbo` (offline, default synthetic panel (seed 51), 10 bps costs). The grid is built by the auditor: each detected parameter is varied 0.5x-2x (at most 16 variants), CSCV with 10 blocks (252 splits).

| Strategy | Parameters varied | Variants | PBO | Risk | P(OOS loss) | Sharpe as written (common window) | Time |
|---|---|---|---|---|---|---|---|
| C01_momentum_12_1 | window 21, window 252, threshold 0.9 | 16 | 0.65 | high | 0% | 1.56 | 1.8s |
| C02_same_bar_reversal | threshold 5 | 8 | 0.10 | low | 100% | -1.79 | 1.5s |
| C03_low_vol | window 60, threshold 10 | 16 | 0.58 | high | 9% | 0.72 | 1.6s |
| C04_ma_crossover | window 50, window 200 | 15 | 0.74 | high | 0% | 1.25 | 1.8s |
| C05_vol_target | window 60, threshold 0.8 | 16 | 0.10 | low | 0% | 1.21 | 2.0s |
| C06_expanding_zscore | window 40, threshold 10 | 16 | 0.10 | low | 0% | 0.90 | 2.1s |
| C07_ffill_rolling | window 20, threshold 8 | 16 | 0.02 | low | 90% | -0.69 | 2.0s |
| C08_cross_sectional_zscore | window 20 | 9 | 0.00 | low | 0% | 1.18 | 1.6s |
| C09_weekly_resample_ffill | window 20, threshold 0.9 | 16 | 0.17 | low | 0% | 0.98 | 2.3s |
| C10_monthly_first_transform | window 60, threshold 10 | 16 | 0.69 | high | 8% | 0.78 | 3.0s |
| C11_scary_named_helper | window 10, window 60 | 16 | 0.07 | low | 0% | 1.01 | 2.2s |
| C12_month_end_calendar_shift | window 21, window 252, threshold 0.9 | 16 | 0.69 | high | 1% | 1.64 | 2.8s |
| C13_transpose_rank | window 20, threshold 0.8 | 16 | 0.01 | low | 1% | 0.75 | 2.2s |
| C14_trailing_ic_lagged | window 5, window 120, threshold 8 | 16 | 0.02 | low | 40% | -0.17 | 2.4s |
| C15_resample_me_last | window 3, threshold 10 | 16 | 0.74 | high | 0% | 1.07 | 2.3s |

**6/15 clean strategies have PBO > 0.5** (the `pbo` test fails, as a non-blocking warning); median PBO 0.10; median runtime 2.1s.

How to read this honestly:

- These strategies are *leak-free*, not *skilful*. A high PBO here does not mean a bug; it means that choosing among nearby parameter values by backtest would not have picked a better variant out of sample. For slow, market-heavy rules (50/200 MA crossover, 12-1 momentum, low-vol) the variants are nearly the same portfolio, so their ranking is mostly noise and PBO sits around 0.5-0.75.
- That is why the test is non-blocking (`medium`): the grid is the auditor's, not the researcher's real search.
- The degradation slope (OOS Sharpe of the IS winner regressed on its IS Sharpe) is close to -1 for every strategy. With CSCV the IS and OOS halves are complements of one fixed sample, so a variant that is lucky in one half is mechanically unlucky in the other; the slope is reported but not used for the verdict.
- Benchmark, strategies and auditor were written by the same author on synthetic data.

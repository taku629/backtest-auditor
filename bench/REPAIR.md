# LeakBench repair benchmark

Generated 2026-10-29 04:05 UTC · live Nemotron on Nebius Token Factory · Coder `nvidia/nemotron-3-super-120b-a12b`, Auditor `nvidia/Nemotron-3-Ultra-550b-a55b` · max_iters=3 (iteration 1 = the leaky code as given, so at most 2 repair attempts) · data: synthetic panel 50 tickers × 1512 days, seed 51.

Each of the 24 leaky LeakBench strategies (header stripped) is fed to `run_audit_loop` as `initial_code`, with a neutral one-line hypothesis that states the intent but not the bug. **Repaired** = the loop's own stopping rule fired (no critical/high deterministic test fails). **A** = full system (Auditor sees the dynamic leak localization in the test JSON plus the 'Leak enters at line N' hint). **B** = same, with the localization removed from the Auditor prompt.

## Headline

| Condition | Repaired | Repaired & non-trivial | Repaired on 1st attempt | Mean iterations (repaired) | Over-corrected | Cost per repair | Total cost | 1st diagnosis names the leak line |
|---|---|---|---|---|---|---|---|---|
| A: full (with localization) | 69/72 (96%) | 69/72 (96%) | 65/72 (90%) | 2.058 | 0 | $0.0072 | $0.4956 | 71/72 (99%) |
| B: ablation (no localization) | 70/72 (97%) | 70/72 (97%) | 65/72 (90%) | 2.0714 | 0 | $0.0084 | $0.5875 | 67/72 (93%) |

Pooled over 3 independent replicates per strategy (same prompts sampled again at the configured temperatures; replicate 0 = first run). Per replicate:

- A: full (with localization): rep 0: repaired 24/24, first attempt 24/24; rep 1: repaired 22/24, first attempt 20/24; rep 2: repaired 23/24, first attempt 21/24. Outcome flips across replicates: 2 (`L14_polyfit_detrend`, `L24_pooled_qcut`); final code differs across replicates for 24/24 strategies, final Sharpe for 11.
- B: ablation (no localization): rep 0: repaired 23/24, first attempt 22/24; rep 1: repaired 23/24, first attempt 21/24; rep 2: repaired 24/24, first attempt 22/24. Outcome flips across replicates: 1 (`L05_dropna_universe`); final code differs across replicates for 24/24 strategies, final Sharpe for 13.

Median Sharpe: leaky draft 0.847 → repaired A: 0.704, B: 0.724.
Live spend logged in `bench/repair_spend.jsonl`: $1.0831 (cached reruns cost nothing). Cost per run is the list-price estimate from `agents/config.py` token counts.

## Per strategy

Iterations counts the leaky draft as iteration 1. Sharpe = after 10 bps costs, t+1 open execution. 'Line?' = the first Auditor diagnosis names the line the localizer found.

Columns per condition are replicate 0; 'reps' = repaired across all replicates.

| Strategy | Bug | Leaky Sharpe | A: result | A: iters | A: final Sharpe | A: line? | A: reps | B: result | B: iters | B: final Sharpe | B: line? | B: reps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L01_shift_minus_one` | negative_shift | 4.118 | ✅ clean | 2 | 0.724 | yes | 3/3 | ✅ clean | 2 | 0.724 | yes | 3/3 |
| `L02_centred_rolling` | centred_window | 8.894 | ✅ clean | 2 | -0.642 | yes | 3/3 | ✅ clean | 2 | -0.678 | yes | 3/3 |
| `L03_bfill_prices` | backward_fill | 0.984 | ✅ clean | 2 | 1.046 | yes | 3/3 | ✅ clean | 2 | 1.046 | yes | 3/3 |
| `L04_fullsample_zscore` | full_sample_normalisation | 0.602 | ✅ clean | 2 | 1.0 | yes | 3/3 | ✅ clean | 2 | 1.0 | yes | 3/3 |
| `L05_dropna_universe` | survivor_universe | 0.879 | ✅ clean | 2 | 0.704 | yes | 3/3 | ❌ error | 3 | None | yes | 1/3 |
| `L06_select_by_final_return` | selection_on_future_performance | 0.783 | ✅ clean | 2 | 1.25 | yes | 3/3 | ✅ clean | 2 | 1.337 | yes | 3/3 |
| `L07_time_rank_full_history` | time_series_rank_full_sample | -0.018 | ✅ clean | 2 | 0.302 | yes | 3/3 | ✅ clean | 3 | 0.369 | yes | 3/3 |
| `L08_reverse_cummax` | reversed_cumulative | 1.974 | ✅ clean | 2 | -0.309 | yes | 3/3 | ✅ clean | 2 | -0.309 | yes | 3/3 |
| `L09_weekly_resample_bfill` | resample_backfill | 5.631 | ✅ clean | 2 | 0.939 | yes | 3/3 | ✅ clean | 2 | 0.939 | yes | 3/3 |
| `L10_reversed_expanding_vol` | reversed_expanding | 0.55 | ✅ clean | 2 | 0.678 | yes | 3/3 | ✅ clean | 2 | 0.678 | yes | 3/3 |
| `L11_groupby_quantile_threshold` | full_sample_quantile_threshold | -0.016 | ✅ clean | 2 | 0.38 | yes | 3/3 | ✅ clean | 2 | 0.38 | yes | 3/3 |
| `L12_helper_hidden_shift` | hidden_negative_shift_in_helper | 7.349 | ✅ clean | 2 | -0.028 | yes | 3/3 | ✅ clean | 2 | 0.023 | yes | 3/3 |
| `L13_alive_at_end` | survivor_universe_last_row | 1.292 | ✅ clean | 2 | 1.252 | yes | 3/3 | ✅ clean | 2 | 1.252 | yes | 3/3 |
| `L14_polyfit_detrend` | full_sample_regression | 1.261 | ✅ clean | 2 | -0.507 | yes | 2/3 | ✅ clean | 2 | -0.507 | no | 3/3 |
| `L15_min_history_filter` | survivor_universe_count_filter | 0.644 | ✅ clean | 2 | 1.55 | yes | 3/3 | ✅ clean | 2 | 1.55 | yes | 3/3 |
| `L16_liquidity_by_recent_volume` | selection_on_future_volume | -0.37 | ✅ clean | 2 | -0.57 | yes | 3/3 | ✅ clean | 2 | -0.553 | yes | 3/3 |
| `L17_rank_default_axis` | pandas_default_axis_rank | -0.019 | ✅ clean | 2 | 0.782 | yes | 3/3 | ✅ clean | 2 | 0.724 | yes | 3/3 |
| `L18_sub_mean_axis_trap` | pandas_default_axis_mean | 0.76 | ✅ clean | 2 | 1.032 | yes | 3/3 | ✅ clean | 2 | 1.032 | yes | 3/3 |
| `L19_resample_ms_last` | resample_label_mismatch | 3.856 | ✅ clean | 2 | 1.072 | yes | 3/3 | ✅ clean | 2 | 1.072 | yes | 3/3 |
| `L20_groupby_transform_last` | groupby_transform_last | 0.815 | ✅ clean | 2 | 0.704 | yes | 3/3 | ✅ clean | 2 | 0.704 | yes | 3/3 |
| `L21_trailing_ic_forward_return` | target_leak_in_trailing_ic | 0.168 | ✅ clean | 2 | -0.257 | yes | 3/3 | ✅ clean | 2 | -0.257 | yes | 3/3 |
| `L22_descending_sort_rolling` | reversed_index_rolling | 10.259 | ✅ clean | 2 | -0.673 | yes | 3/3 | ✅ clean | 2 | -0.673 | no | 3/3 |
| `L23_fillna_full_mean` | imputation_with_full_sample_mean | 0.992 | ✅ clean | 2 | 1.032 | yes | 3/3 | ✅ clean | 2 | 1.032 | yes | 3/3 |
| `L24_pooled_qcut` | pooled_quantile_breakpoints | 0.325 | ✅ clean | 2 | 0.749 | yes | 1/3 | ✅ clean | 2 | 0.724 | yes | 3/3 |

## Failure modes (all replicates)

- Not repaired: 5 of 144 runs; 5 of them ended because the Coder's last rewrite crashed or timed out in the sandbox, 0 with a blocking test still failing.
- Runs where at least one Coder rewrite crashed in the sandbox: 13 (6 sandbox timeouts at 90 s, the rest pandas/numpy API errors). The loop feeds the traceback back to the Coder without an Auditor call.
- Over-corrected (repaired by going to cash: `non_trivial` failing or mean gross exposure < 5%): 0.

  - `L05_dropna_universe` B rep 0: failing → error → error; last: AttributeError: 'numpy.ndarray' object has no attribute 'reindex' (strategy line 34: is_month_end = is_month_end.reindex(close.index).fillna(False))
  - `L14_polyfit_detrend` A rep 1: failing → error → error; last: Timeout: strategy exceeded 90s wall clock
  - `L24_pooled_qcut` A rep 1: failing → error → error; last: IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer. Replace or remove non-finite values or cast to an integer type that supports these 
  - `L05_dropna_universe` B rep 1: failing → error → error; last: ValueError: Shape of passed values is (1512, 1), indices imply (1512, 50) (strategy line 28: is_month_start = pd.DataFrame(is_month_start_arr,)
  - `L24_pooled_qcut` A rep 2: failing → error → error; last: IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer. Replace or remove non-finite values or cast to an integer type that supports these 

## Caveats

- n = 24 strategies per condition per replicate; one strategy is ~4 points of repair rate. Differences of one or two runs between A and B are within sampling noise.
- Ceiling effect: every LeakBench bug is a single statement in a ~10-20 line function, the leak also shows in the static scan or the perturbation evidence, and code comments remain in the stripped file. The Auditor finds the line without the localization hint most of the time, so this benchmark can only show the hint's effect on the margin (wrong line numbers, misattributed causes), not on easy-vs-impossible.
- 'Repaired' is the loop's own deterministic criterion (no critical/high test fails). It does not check that the rewrite still implements the hypothesis faithfully; the Coder sometimes changes parameters (e.g. a trailing 252-day window instead of the full history). Over-correction is only checked as going (almost) to cash.
- The LLM calls are sampled (Coder temperature 0.2, Auditor 0.1, reasoning on), so reruns differ; replicates > 0 sample the same prompts again. Cached replays are exact.
- Hypotheses are hand-written one-liners that state the intent without the bug (see `HYPOTHESES` in `bench/run_repair.py`); the strategies' own docstrings and comments are still in the code.
- The Summarizer is skipped (`summarize=False`) to save cost; it does not affect the loop's outcome.
- Costs are list-price estimates from token counts (`agents/config.py`), not invoices.

## Observations from this run

- **No measurable repair-rate effect of the localization hint on LeakBench.** Pooled over 3 replicates: A 69/72 (95.8%) vs B 70/72 (97.2%); on replicate 0 alone A 24/24 vs B 23/24. With failures this rare the difference is noise in both directions (Fisher exact p ≈ 1).
- **Where the hint does show up: Auditor effort and precision.** Without it, Nemotron Ultra's first diagnosis used ~46% more completion tokens (mean 1,560 vs 1,071, incl. reasoning; 73 vs 72 calls), so the Auditor cost per run was ~17% lower with the hint despite ~360 extra prompt tokens. The first diagnosis named the localized line in 71/72 runs with the hint vs 67/72 without; the B misses are L22 (3/3 replicates: the newest-first rolling bug blamed on lines 6–7 instead of line 10) and L14 (2/3: "lines 10–14", just short of the `np.polyfit` line 15). The mechanism was still described correctly, and those runs were repaired anyway.
- **Every non-repair is a Coder crash, not a residual leak.** All 5 failed runs ended with the Coder's rewrite crashing or timing out in the sandbox (pandas API misuse such as `DataFrame.floor`, a 1-D mask used as a 2-D `where` condition, or a per-ticker Python loop over 1,512 days exceeding the 90 s wall clock for L14). No run ended with a blocking leak test still failing, and none "repaired" by going to cash.
- Sharpe after repair: the spectacular leaky drafts collapse (L22 10.26 → −0.67, L02 8.89 → −0.64, L12 7.35 → ~0, L09 5.63 → 0.94, L01 4.12 → 0.72); several leaky drafts that looked mediocre end slightly higher after the fix (e.g. L15 0.64 → 1.55, L17 −0.02 → 0.72), a reminder that a leak does not always inflate performance on this panel.
- Non-determinism: the final code differs between replicates for 24/24 strategies in both conditions (the Coder rewrites freely), final Sharpe differs for 11 (A) / 13 (B) of 24; the repaired/not-repaired outcome flipped across replicates for L14 and L24 (A) and L05 (B).
- Total live spend for all 144 runs (3 replicates × 2 conditions × 24 strategies): $1.083 at list prices, logged call by call in `bench/repair_spend.jsonl`.

## Reproduce

```bash
NEBIUS_API_KEY=... python -m bench.run_repair --replicates 0 1 2   # responses cached in bench/repair_cache.json
python -m bench.run_repair --report-only [--notes-file notes.md]   # rebuild this file from repair_results.json
```

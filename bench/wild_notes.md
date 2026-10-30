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

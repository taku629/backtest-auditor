# Bundled real prices: `us_equities_2014_2025.csv.gz`

Daily OHLCV for **40 liquid US stocks and ETFs**, 2014-01-02 to 2025-12-31 (3,018 trading days,
110,050 rows, ~2.2 MB gzip). Long format: `date,ticker,open,high,low,close,volume`.
Loaded offline by `data/real.py::load_real_panel()`; the app never calls the network.

| Group | Tickers |
|---|---|
| ETFs | SPY, QQQ, IWM |
| Large caps (listed all window) | AAPL MSFT AMZN GOOGL META NVDA TSLA JPM BAC WFC XOM CVX JNJ PFE MRK UNH PG KO PEP WMT HD DIS NFLX INTC CSCO ORCL VZ CAT BA MCD |
| IPOs inside the window | PYPL 2015-07-06, UBER 2019-05-10, SNOW 2020-09-16, PLTR 2020-09-30, ABNB 2020-12-10, COIN 2021-04-14, RIVN 2021-11-10 |

Before its first trade a ticker is NaN in every field, exactly like the synthetic panel.

## Source and processing

* Source: Yahoo Finance chart API (`query1.finance.yahoo.com/v8/finance/chart/<TICKER>`), downloaded
  2026-10-29 with `scripts/fetch_real_data.py`. The script tries Stooq's CSV first, but Stooq now answers
  scripted clients with a JavaScript browser check, so every ticker came from Yahoo.
* Prices are **split- and dividend-adjusted**: open/high/low/close are scaled by `adjclose / close`.
  Volume is the raw reported share volume.
* Prices rounded to 6 significant digits (below $100) or 3 decimals (above); high/low are clipped to
  contain open and close.

## Licence / terms

The data is Yahoo Finance's and is subject to Yahoo's terms of service; it is included here only as a
small sample for non-commercial research and demonstration of the audit. It is not redistributed for
any other purpose, comes with no warranty, and must not be used for trading. Re-run
`scripts/fetch_real_data.py` to rebuild it yourself.

## Known bias: survivorship (on purpose, and reported)

Every ticker in this file still trades today. Free daily sources do not carry delisted securities
(Lehman-style failures, acquired companies, bankruptcies), so **the dataset itself is survivorship-biased**:
any backtest on it overstates what a 2014 investor could have earned. The universe was also picked in 2026
from names that are large *now*, which adds selection bias on top.

The audit is honest about this: the survivorship test finds no delisted names, cannot run its
"revive the delisted" counterfactual, and returns the `data_warning`
"Dataset contains no delisted tickers: results are survivorship-biased by construction." (severity medium,
non-blocking). The point-in-time checks that *can* run on this data still do: no weight before listing
(IPO names), and the future-perturbation test, which artificially delists live names after each cut date.
Use the seeded synthetic panel to see the full survivorship test with real delistings.

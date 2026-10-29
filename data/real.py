"""Bundled real US equity prices (offline at runtime).

``data/real/us_equities_2014_2025.csv.gz`` holds daily split- and
dividend-adjusted OHLCV for 40 liquid US stocks and ETFs, 2014-01-02 to
2025-12-31, in long format ``date,ticker,open,high,low,close,volume``. It was
built once with ``scripts/fetch_real_data.py``; nothing here touches the network.

The panel has the same structure as :func:`data.synthetic.generate_panel`:
wide frames ``open/high/low/close/volume`` (dates x tickers), NaN where the
ticker is not listed. Seven names IPO inside the window (PYPL 2015, UBER 2019,
SNOW/PLTR/ABNB 2020, COIN/RIVN 2021), so they are NaN before their first trade.

Survivorship caveat: every ticker is a company that still trades today. A free
source does not carry delisted securities, so the dataset itself is
survivorship-biased. The audit's survivorship test reports this as a
``data_warning`` (no delisted names) instead of pretending otherwise.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from data.loader import load_long_csv
from data.synthetic import FIELDS, generate_panel

REAL_DIR = Path(__file__).resolve().parent / "real"
DATA_FILE = REAL_DIR / "us_equities_2014_2025.csv.gz"
REAL_START, REAL_END = "2014-01-01", "2025-12-31"

# Listed for the whole window (ETFs + large caps across sectors) ...
_FULL = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
         "JPM", "BAC", "WFC", "XOM", "CVX", "JNJ", "PFE", "MRK", "UNH", "PG",
         "KO", "PEP", "WMT", "HD", "DIS", "NFLX", "INTC", "CSCO", "ORCL", "VZ",
         "CAT", "BA", "MCD")
# ... and names that list inside the window (exercise the NaN-before-IPO logic).
IPO_TICKERS = ("PYPL", "UBER", "SNOW", "PLTR", "ABNB", "COIN", "RIVN")
REAL_TICKERS = _FULL + IPO_TICKERS

# Smaller default subset for quick demos (keeps 3 IPO names so listing logic still shows).
DEMO_TICKERS = ("SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM",
                "XOM", "JNJ", "PG", "KO", "WMT", "HD", "NFLX", "INTC", "CAT", "BA",
                "UBER", "ABNB", "PLTR")


@lru_cache(maxsize=1)
def _full_panel() -> dict[str, pd.DataFrame]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} is missing; run scripts/fetch_real_data.py once to build it.")
    return load_long_csv(DATA_FILE)


def load_real_panel(tickers=None, start=None, end=None) -> dict[str, pd.DataFrame]:
    """Real-price panel, same structure as ``generate_panel()``.

    ``tickers``: subset (default: all 40). ``start``/``end``: inclusive date bounds.
    Rows where no selected ticker trades are dropped. Returns fresh copies.
    """
    full = _full_panel()
    cols = list(full["close"].columns) if tickers is None else list(tickers)
    unknown = sorted(set(cols) - set(full["close"].columns))
    if unknown:
        raise KeyError(f"tickers not in the bundled dataset: {unknown}")
    close = full["close"].loc[start:end, cols]
    idx = close.index[close.notna().any(axis=1)]
    out = {f: full[f].loc[idx, cols].copy() for f in FIELDS}
    for f in FIELDS:
        out[f].index.name = "date"
        out[f].columns.name = "ticker"
    return out


def available_universes() -> dict[str, dict]:
    """Registry of selectable datasets: name -> {"description", "loader"} (loader() returns a panel)."""
    return {
        "Synthetic (seeded)": {
            "description": ("50 synthetic tickers over 6 years with 8 delisted names that crash before they "
                            "vanish and 6 IPOs. Seeded, so results are reproducible; has a small genuine "
                            "momentum effect."),
            "loader": generate_panel,
        },
        "Real US equities (40, 2014–2025)": {
            "description": ("Adjusted daily OHLCV for 40 liquid US stocks and ETFs, 2014–2025 (Yahoo Finance), "
                            "incl. 7 later IPOs (PYPL, UBER, SNOW, PLTR, ABNB, COIN, RIVN). No delisted names: "
                            "the dataset itself is survivorship-biased, and the audit says so."),
            "loader": load_real_panel,
        },
        "Real US equities (23, demo subset)": {
            "description": "Faster subset of the real dataset: 20 large caps/ETFs plus UBER, ABNB, PLTR.",
            "loader": lambda: load_real_panel(DEMO_TICKERS),
        },
    }

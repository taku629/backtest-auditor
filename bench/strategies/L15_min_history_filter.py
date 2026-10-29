# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: survivor_universe_count_filter
# LEAKBENCH-NOTE: Keeps tickers with >= 1200 observations over the full sample: removes names that later delist or IPO late.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Low-volatility long-only, 10 names, restricted to seasoned stocks."""
    close = data["close"]
    listed = close.notna()
    # Seasoned stocks only: require a long trading record.
    seasoned = close.count() >= 1200
    vol = close.pct_change().rolling(60).std()
    vol = vol.loc[:, seasoned].where(listed)
    rank = vol.rank(axis=1, ascending=True)
    pick = (rank <= 10).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    return w.reindex(columns=close.columns).fillna(0.0)

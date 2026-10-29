# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: survivor_universe
# LEAKBENCH-NOTE: dropna(axis=1) keeps only tickers with a complete history (survivors).
import numpy as np
import pandas as pd


def generate_signals(data):
    """Low-volatility: hold the 10 least volatile names (60-day vol), monthly rebalance."""
    close = data["close"]
    # Work on a clean universe of names with full data.
    clean = close.dropna(axis=1)
    vol = clean.pct_change().rolling(60).std()
    rank = vol.rank(axis=1, ascending=True)
    pick = (rank <= 10).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    month = pd.Series(clean.index.to_period("M"), index=clean.index)
    w.loc[(month == month.shift(1)).to_numpy()] = np.nan
    w = w.ffill().fillna(0.0)
    return w.reindex(columns=close.columns).fillna(0.0)

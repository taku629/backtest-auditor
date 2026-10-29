# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Cross-sectional 12-1 momentum, monthly rebalance, ffill hold.
import numpy as np
import pandas as pd


def generate_signals(data):
    """12-1 month momentum, long the top decile, monthly rebalance."""
    close = data["close"]
    listed = close.notna()
    mom = (close.shift(21) / close.shift(252) - 1).where(listed)
    rank = mom.rank(axis=1, pct=True)
    pick = ((rank >= 0.9) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    month = pd.Series(close.index.to_period("M"), index=close.index)
    w.loc[(month == month.shift(1)).to_numpy()] = np.nan
    w = w.ffill().fillna(0.0)
    return w.where(listed, 0.0)

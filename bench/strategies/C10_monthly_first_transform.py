# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: groupby(month).transform('first') holds the first trading day's portfolio for the month (causal; 'last' would leak).
import numpy as np
import pandas as pd


def generate_signals(data):
    """Low-volatility with monthly rebalance on the first trading day of each month."""
    close = data["close"]
    listed = close.notna()
    vol = close.pct_change().rolling(60, min_periods=60).std().where(listed)
    rank = vol.rank(axis=1, ascending=True)
    pick = ((rank <= 10) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    month = close.index.to_period("M")
    w = w.groupby(month).transform("first")
    return w.where(listed, 0.0)

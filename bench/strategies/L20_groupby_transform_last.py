# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: groupby_transform_last
# LEAKBENCH-NOTE: groupby(month).transform('last') gives every day the portfolio of the LAST day of its month.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Low-volatility, 10 names, one portfolio per calendar month."""
    close = data["close"]
    listed = close.notna()
    vol = close.pct_change().rolling(60, min_periods=60).std().where(listed)
    rank = vol.rank(axis=1, ascending=True)
    pick = ((rank <= 10) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    month = close.index.to_period("M")
    # Hold a single portfolio for the whole month.
    w = w.groupby(month).transform("last")
    return w.where(listed, 0.0)

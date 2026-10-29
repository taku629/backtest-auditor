# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: groupby_transform_max
# LEAKBENCH-NOTE: groupby(week).transform('max') spreads the week's max weight over the whole week.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-quintile 20-day momentum, one portfolio per week."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20).where(listed)
    pick = (mom.rank(axis=1, pct=True) >= 0.8).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    week = close.index.to_period("W")
    w = w.groupby(week).transform("max")
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w.where(listed, 0.0)

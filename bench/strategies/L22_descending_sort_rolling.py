# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: reversed_index_rolling
# LEAKBENCH-NOTE: Index sorted newest-first before rolling(20): each 'trailing' window covers the next 19 days.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Buy the 8 names furthest below their 20-day moving average."""
    close = data["close"]
    listed = close.notna()
    recent_first = close.sort_index(ascending=False)
    ma = recent_first.rolling(20, min_periods=20).mean().sort_index()
    gap = (close / ma - 1).where(listed)
    rank = gap.rank(axis=1, ascending=True)
    pick = ((rank <= 8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

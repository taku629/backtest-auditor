# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: centred_window
# LEAKBENCH-NOTE: Moving average with center=True averages the next 10 bars.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Buy the 8 names furthest below their 20-day moving average (mean reversion)."""
    close = data["close"]
    listed = close.notna()
    # Centred window: a smoother, lag-free estimate of fair value.
    fair = close.rolling(21, center=True, min_periods=5).mean()
    gap = (close / fair - 1).where(listed)
    rank = gap.rank(axis=1, ascending=True)
    pick = ((rank <= 8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: pandas_default_axis_rank
# LEAKBENCH-NOTE: rank(ascending=False) without axis ranks down the time axis.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Hold the 10 names with the strongest 60-day return."""
    close = data["close"]
    listed = close.notna()
    score = close.pct_change(60).where(listed)
    order = score.rank(ascending=False)
    pick = ((order <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

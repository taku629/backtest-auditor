# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Trailing 60-day volatility, 10 least volatile names.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Low-volatility anomaly: equal-weight the 10 least volatile names."""
    close = data["close"]
    listed = close.notna()
    vol = close.pct_change().rolling(60, min_periods=60).std().where(listed)
    rank = vol.rank(axis=1, ascending=True)
    pick = ((rank <= 10) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0).where(listed, 0.0)

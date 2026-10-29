# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Month-end resample, forward-filled; clean.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Monthly 6-month momentum, top 10."""
    close = data["close"]
    listed = close.notna()
    monthly = close.resample("BME").last()
    mom = monthly / monthly.shift(6) - 1
    order = mom.rank(axis=1, ascending=False)
    pick = (order <= 10).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    w = w.reindex(close.index, method="ffill").fillna(0.0)
    return w.where(listed, 0.0)

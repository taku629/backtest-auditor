# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Friday portfolio from resample('W-FRI').last(), forward-filled onto the following week (causal).
import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-decile 20-day momentum, rebalanced on Fridays and held through the next week."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20).where(listed)
    rank = mom.rank(axis=1, pct=True)
    pick = ((rank >= 0.9) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    weekly = w.resample("W-FRI").last()
    w = weekly.reindex(close.index, method="ffill").fillna(0.0)
    return w.where(listed, 0.0)

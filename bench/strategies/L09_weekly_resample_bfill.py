# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: resample_backfill
# LEAKBENCH-NOTE: Weekly portfolio taken at Friday (.last()) and back-filled onto Mon-Thu of the same week.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-decile 20-day momentum, rebalanced weekly."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20).where(listed)
    rank = mom.rank(axis=1, pct=True)
    pick = ((rank >= 0.9) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # One portfolio per week, spread over the trading days of that week.
    weekly = w.resample("W-FRI").last()
    return weekly.reindex(close.index, method="bfill").fillna(0.0)

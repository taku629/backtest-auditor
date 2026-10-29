# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: 50/200 moving-average crossover per ticker, trailing windows.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Golden cross: long every stock whose 50-day MA is above its 200-day MA."""
    close = data["close"]
    listed = close.notna()
    fast = close.rolling(50, min_periods=50).mean()
    slow = close.rolling(200, min_periods=200).mean()
    on = (fast > slow) & listed
    pick = on.astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

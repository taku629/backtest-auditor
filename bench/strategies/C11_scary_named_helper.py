# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Helper named 'peek_next' with a misleading docstring but it shifts by +1 (backward-looking) - tests reviewer false positives.
import numpy as np
import pandas as pd


def peek_next(df, n=1):
    """Look at the next value of the series."""
    return df.shift(n)


def generate_signals(data):
    """EWMA trend: long names whose 10-day EWMA is above their 60-day EWMA, using yesterday's values."""
    close = data["close"]
    listed = close.notna()
    fast = peek_next(close.ewm(span=10, min_periods=10).mean())
    slow = peek_next(close.ewm(span=60, min_periods=60).mean())
    on = (fast > slow) & listed
    pick = on.astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

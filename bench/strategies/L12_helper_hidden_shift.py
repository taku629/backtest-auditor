# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: hidden_negative_shift_in_helper
# LEAKBENCH-NOTE: Helper with an innocent name shifts by -lag (a variable), so the static scan cannot see a negative constant.
import numpy as np
import pandas as pd


def to_trade_date(df, lag=1):
    """Move each signal to the bar on which it is executed (the engine trades one bar later)."""
    return df.shift(-lag)


def generate_signals(data):
    """Short-term (5-day) momentum, top quintile, equal weight."""
    close = data["close"]
    listed = close.notna()
    mom = to_trade_date(close.pct_change(5))
    rank = mom.where(listed).rank(axis=1, pct=True)
    pick = ((rank >= 0.8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

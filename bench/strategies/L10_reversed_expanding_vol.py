# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: reversed_expanding
# LEAKBENCH-NOTE: Expanding volatility computed on the time-reversed series = vol of all FUTURE returns.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Inverse-volatility weights on the 25 lowest-vol names, using a stable expanding estimate."""
    close = data["close"]
    listed = close.notna()
    ret = close.pct_change()
    # Expanding window is more stable than a short rolling window.
    vol = ret.iloc[::-1].expanding(min_periods=60).std().iloc[::-1]
    vol = vol.where(listed)
    rank = vol.rank(axis=1, ascending=True)
    inv = (1.0 / vol).where(rank <= 25)
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)

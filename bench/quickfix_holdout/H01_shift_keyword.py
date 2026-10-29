# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: negative_shift_keyword
# LEAKBENCH-NOTE: Keyword periods=-2 on a 10-day return.
import numpy as np
import pandas as pd


def generate_signals(data):
    """10-day momentum, long the top 12 names."""
    close = data["close"]
    listed = close.notna()
    sig = close.pct_change(periods=10).shift(periods=-2)
    order = sig.where(listed).rank(axis=1, ascending=False)
    pick = ((order <= 12) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

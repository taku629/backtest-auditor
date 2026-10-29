# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: negative_shift
# LEAKBENCH-NOTE: Momentum signal shifted by -1: row t holds tomorrow's momentum (obvious).
import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-quintile 20-day momentum, equal weight."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20)
    # Align the signal with the bar it will be traded on.
    mom = mom.shift(-1)
    rank = mom.where(listed).rank(axis=1, pct=True)
    pick = ((rank >= 0.8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

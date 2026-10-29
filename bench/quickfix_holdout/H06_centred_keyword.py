# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: centred_window
# LEAKBENCH-NOTE: Centred 30-day volatility via keyword window=30.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Low volatility, 12 names."""
    close = data["close"]
    listed = close.notna()
    vol = close.pct_change().rolling(window=30, center=True, min_periods=10).std()
    order = vol.where(listed).rank(axis=1)
    pick = ((order <= 12) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

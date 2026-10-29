# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: reversed_ewm
# LEAKBENCH-NOTE: EWM volatility on a [::-1] reversed frame.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Inverse-volatility on the 20 calmest names."""
    close = data["close"]
    listed = close.notna()
    ret = close.pct_change()
    vol = ret[::-1].ewm(span=30, min_periods=30).std()[::-1].where(listed)
    calm = vol.rank(axis=1) <= 20
    inv = (1.0 / vol).where(calm)
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)

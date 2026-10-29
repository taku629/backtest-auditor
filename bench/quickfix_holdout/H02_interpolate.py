# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: interpolation
# LEAKBENCH-NOTE: interpolate() over price gaps uses the next valid price.
import numpy as np
import pandas as pd


def generate_signals(data):
    """40-day momentum on gap-free prices, top 10."""
    close = data["close"]
    smooth = close.interpolate()
    mom = smooth.pct_change(40)
    order = mom.rank(axis=1, ascending=False)
    pick = (order <= 10).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

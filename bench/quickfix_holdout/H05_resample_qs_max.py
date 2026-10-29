# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: resample_start_label
# LEAKBENCH-NOTE: Quarterly high labelled at the quarter start (QS) and forward-filled.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Buy the 10 names closest to their quarterly high."""
    close = data["close"]
    listed = close.notna()
    qhigh = close.resample("QS").max().reindex(close.index, method="ffill")
    near = (close / qhigh).where(listed)
    order = near.rank(axis=1, ascending=False)
    pick = ((order <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

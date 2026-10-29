# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: resample_label_mismatch
# LEAKBENCH-NOTE: resample('MS').last() labels the month-END close at the month START; forward-filling then shows it all month.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Monthly 3-month momentum: long the top 10 names, rebalanced monthly."""
    close = data["close"]
    listed = close.notna()
    monthly = close.resample("MS").last()
    mom = monthly / monthly.shift(3) - 1
    rank = mom.rank(axis=1, ascending=False)
    pick = (rank <= 10).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    w = w.reindex(close.index, method="ffill").fillna(0.0)
    return w.where(listed, 0.0)

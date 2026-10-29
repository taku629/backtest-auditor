# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: pooled_quantile_breakpoints
# LEAKBENCH-NOTE: pd.qcut on the stacked panel computes quintile breakpoints from all dates, including future ones.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Quintile sort on 20-day momentum: hold every stock-day in the top quintile."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20).where(listed)
    long = mom.stack()
    bucket = pd.qcut(long, 5, labels=False).unstack().reindex(index=close.index, columns=close.columns)
    pick = ((bucket == 4) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: pandas_default_axis_mean
# LEAKBENCH-NOTE: mom.sub(mom.mean(), axis=1) looks cross-sectional, but mom.mean() is each ticker's full-sample time mean.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Market-neutral momentum: long the 10 names with the highest demeaned 60-day return."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(60).where(listed)
    # Remove the common component before ranking.
    demeaned = mom.sub(mom.mean(), axis=1)
    rank = demeaned.rank(axis=1, ascending=False)
    pick = ((rank <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

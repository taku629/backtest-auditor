# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: pandas_default_axis_rank
# LEAKBENCH-NOTE: DataFrame.rank(pct=True) defaults to axis=0: each value is ranked against the ticker's whole history, not across stocks.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Cross-sectional momentum: hold names in the top quintile of 20-day return."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20).where(listed)
    pctl = mom.rank(pct=True)
    pick = ((pctl >= 0.8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

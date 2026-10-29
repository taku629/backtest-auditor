# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: selection_on_future_performance
# LEAKBENCH-NOTE: Universe = 12 names with the best Sharpe over the last 252 rows (tail), used for every date.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Short-term reversal inside the 12 highest-quality names."""
    close = data["close"]
    ret = close.pct_change()
    quality = ret.tail(252).mean() / ret.tail(252).std()
    best = quality.nlargest(12).index
    rev = -close.pct_change(5)
    sub = rev[best]
    order = sub.rank(axis=1, ascending=False)
    pick = (order <= 4).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    return w.reindex(columns=close.columns).fillna(0.0)

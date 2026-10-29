# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Expanding z-score of momentum, lagged; clean.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Momentum z-scored against its own past, top 10."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(40)
    z = (mom - mom.expanding(min_periods=60).mean()) / mom.expanding(min_periods=60).std()
    order = z.shift(1).where(listed).rank(axis=1, ascending=False)
    pick = ((order <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

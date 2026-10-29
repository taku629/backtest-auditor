# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: mom.T.rank(pct=True).T: the default axis=0 on the transposed frame is a cross-sectional rank (looks like the L17 trap, but is correct).
import numpy as np
import pandas as pd


def generate_signals(data):
    """Cross-sectional momentum: hold names in the top quintile of 20-day return."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20).where(listed)
    pctl = mom.T.rank(pct=True).T
    pick = ((pctl >= 0.8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

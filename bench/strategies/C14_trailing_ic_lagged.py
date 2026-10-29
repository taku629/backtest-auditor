# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Trailing IC correlates YESTERDAY's signal with TODAY's realised return, so the window ending at t is fully known at t.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Adaptive reversal: trade 5-day reversal only on stocks where it has recently worked."""
    close = data["close"]
    listed = close.notna()
    ret = close.pct_change()
    sig = -close.pct_change(5)
    ic = sig.shift(1).rolling(120, min_periods=60).corr(ret)   # signal at t-1 vs return realised at t
    score = sig.where((ic > 0) & listed)
    rank = score.rank(axis=1, ascending=False)
    pick = ((rank <= 8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

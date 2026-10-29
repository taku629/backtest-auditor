# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: target_leak_in_trailing_ic
# LEAKBENCH-NOTE: 'Trailing' IC correlates the signal with the NEXT-day return; the window ending at t contains the return of t+1.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Adaptive reversal: trade 5-day reversal only on stocks where it has recently worked."""
    close = data["close"]
    listed = close.notna()
    ret = close.pct_change()
    sig = -close.pct_change(5)
    fwd = ret.shift(-1)                       # the return each signal is meant to predict
    ic = sig.rolling(120, min_periods=60).corr(fwd)   # trailing 120-day information coefficient
    score = sig.where((ic > 0) & listed)
    rank = score.rank(axis=1, ascending=False)
    pick = ((rank <= 8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: target_leak_in_trailing_hit_rate
# LEAKBENCH-NOTE: Trailing hit rate uses the 5-day FORWARD return.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Reversal, only where it has been working lately."""
    close = data["close"]
    listed = close.notna()
    sig = -close.pct_change(5)
    fwd5 = close.pct_change(5).shift(-5)
    hit = (np.sign(sig) * fwd5).rolling(60, min_periods=40).mean()
    order = sig.where((hit > 0) & listed).rank(axis=1, ascending=False)
    pick = ((order <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

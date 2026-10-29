# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: full_sample_median
# LEAKBENCH-NOTE: Signal demeaned by each ticker's full-sample median.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Momentum relative to its usual level, top 10."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(30)
    excess = mom - mom.median()
    order = excess.where(listed).rank(axis=1, ascending=False)
    pick = ((order <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: full_sample_quantile_threshold
# LEAKBENCH-NOTE: Per-ticker 80th-percentile threshold of momentum computed with groupby over the whole sample.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Breakout: hold a stock when its momentum is above its usual top-quintile level."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20)
    long = mom.stack()
    # Each stock's own breakout level.
    level = long.groupby(level=1).quantile(0.8)
    on = mom.gt(level, axis=1) & listed
    pick = on.astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

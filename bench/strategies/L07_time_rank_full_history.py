# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: time_series_rank_full_sample
# LEAKBENCH-NOTE: rank(axis=0, pct=True) ranks each value against the ticker's entire history, including the future.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Buy stocks whose 20-day momentum is high relative to their own history."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20)
    # Percentile of today's momentum within each stock's own history.
    pctl = mom.rank(axis=0, pct=True)
    pick = ((pctl > 0.8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

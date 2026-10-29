# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Per-ticker z-score with expanding mean/std (looks like full-sample normalisation, but is point-in-time).
import numpy as np
import pandas as pd


def generate_signals(data):
    """Standardised momentum: each stock's 40-day return vs its own history, top 10."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(40)
    mu = mom.expanding(min_periods=120).mean()
    sd = mom.expanding(min_periods=120).std()
    z = ((mom - mu) / sd).where(listed)
    rank = z.rank(axis=1, ascending=False)
    pick = ((rank <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

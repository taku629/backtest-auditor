# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: full_sample_normalisation
# LEAKBENCH-NOTE: Per-ticker z-score uses the whole-sample mean and std of the signal.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Volatility-adjusted momentum, standardised per ticker, long the top 10."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(40)
    # Standardise each stock's momentum so every name is on the same scale.
    z = (mom - mom.mean()) / mom.std()
    rank = z.where(listed).rank(axis=1, ascending=False)
    pick = ((rank <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

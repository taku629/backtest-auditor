# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: z-score across tickers on each date (mean/std with axis=1) - looks like normalisation, but is cross-sectional.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Long-short momentum: weights proportional to the cross-sectional z-score of 20-day return."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(20).where(listed)
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)
    z = z.clip(-3, 3)
    gross = z.abs().sum(axis=1).replace(0, np.nan)
    return z.div(gross, axis=0).fillna(0.0)

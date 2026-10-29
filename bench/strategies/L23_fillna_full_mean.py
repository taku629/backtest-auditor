# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: imputation_with_full_sample_mean
# LEAKBENCH-NOTE: Missing momentum (each ticker's first 60 days) is filled with the ticker's full-sample mean.
import numpy as np
import pandas as pd


def generate_signals(data):
    """60-day momentum, top 10, with missing values imputed so new listings can be ranked."""
    close = data["close"]
    listed = close.notna()
    mom = close.pct_change(60)
    mom = mom.fillna(mom.mean()).where(listed)
    rank = mom.rank(axis=1, ascending=False)
    pick = ((rank <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Liquidity universe from the trailing 60-day median volume on each date; clean.
import numpy as np
import pandas as pd


def generate_signals(data):
    """20-day reversal among the 25 most liquid names as of each date."""
    close = data["close"]
    listed = close.notna()
    adv = data["volume"].rolling(60).median()
    liquid = adv.rank(axis=1, ascending=False, method="first") <= 25
    ret = close.pct_change(20).where(listed & liquid)
    order = ret.rank(axis=1, ascending=True)
    pick = (order <= 5).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: survivor_universe
# LEAKBENCH-NOTE: dropna(axis='columns') keeps only survivors.
import numpy as np
import pandas as pd


def generate_signals(data):
    """20-day momentum among names with complete data, top 10."""
    close = data["close"]
    universe = close.dropna(axis="columns")
    mom = universe.pct_change(20)
    order = mom.rank(axis=1, ascending=False)
    pick = (order <= 10).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    return w.reindex(columns=close.columns).fillna(0.0)

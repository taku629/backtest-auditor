# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: survivor_universe_last_row
# LEAKBENCH-NOTE: Boolean mask of names priced on the last row, applied with .loc[:, mask] to all dates.
import numpy as np
import pandas as pd


def generate_signals(data):
    """60-day momentum, top 8, among names still trading."""
    close = data["close"]
    alive = close.iloc[-1].notna()
    mom = close.pct_change(60)
    mom = mom.loc[:, alive]
    order = mom.rank(axis=1, ascending=False)
    pick = (order <= 8).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    return w.reindex(columns=close.columns).fillna(0.0)

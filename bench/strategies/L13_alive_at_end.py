# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: survivor_universe_last_row
# LEAKBENCH-NOTE: Universe = tickers with a price on the last date (close.iloc[-1].notna()), i.e. survivors.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Equal-weight the top third of names by 120-day momentum, among currently tradable stocks."""
    close = data["close"]
    # Only consider stocks that are currently tradable.
    tradable = close.columns[close.iloc[-1].notna()]
    px = close[tradable]
    listed = px.notna()
    mom = px.pct_change(120).where(listed)
    rank = mom.rank(axis=1, pct=True)
    pick = ((rank >= 2 / 3) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w.reindex(columns=close.columns).fillna(0.0)

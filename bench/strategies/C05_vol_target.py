# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Momentum book scaled to 10% target vol using lagged realised vol (shift(1) looks suspicious but is backward-looking).
import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-quintile 60-day momentum, scaled to a 10% annual volatility target."""
    close = data["close"]
    listed = close.notna()
    ret = close.pct_change()
    mom = close.pct_change(60).where(listed)
    rank = mom.rank(axis=1, pct=True)
    pick = ((rank >= 0.8) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # Realised vol of the book, using yesterday's weights against today's returns.
    book = (w.shift(1) * ret.fillna(0.0)).sum(axis=1)
    realised = book.rolling(60, min_periods=20).std() * np.sqrt(252)
    lever = (0.10 / realised).clip(upper=2.0).fillna(0.0)
    return w.mul(lever, axis=0)

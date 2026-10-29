# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none
# LEAKBENCH-NOTE: Forward-filled prices and a trailing (non-centred) rolling mean; masked to listed names.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Buy the 8 names furthest below their trailing 20-day average."""
    close = data["close"]
    listed = close.notna()
    px = close.ffill()                     # carry the last known price forward
    fair = px.rolling(20, min_periods=20).mean()
    gap = (px / fair - 1).where(listed)
    rank = gap.rank(axis=1, ascending=True)
    pick = ((rank <= 8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

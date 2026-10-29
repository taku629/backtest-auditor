# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: backward_fill
# LEAKBENCH-NOTE: Prices are back-filled before computing a trailing return, so gaps take future prices and pre-IPO rows get the listing price.
import numpy as np
import pandas as pd


def generate_signals(data):
    """60-day momentum, long the top 10, with gap-free prices."""
    close = data["close"]
    # Fill holes in the price history so the 60-day return is always defined.
    px = close.bfill()
    mom = px / px.shift(60) - 1
    rank = mom.rank(axis=1, ascending=False)
    pick = (rank <= 10).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

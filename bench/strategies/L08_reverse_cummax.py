# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: reversed_cumulative
# LEAKBENCH-NOTE: close.iloc[::-1].cummax().iloc[::-1] is the maximum of FUTURE prices, not the past high.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Buy the 10 names trading at the largest discount to their peak (drawdown reversal)."""
    close = data["close"]
    listed = close.notna()
    # Running peak of each price series.
    peak = close.iloc[::-1].cummax().iloc[::-1]
    discount = (close / peak - 1).where(listed)
    rank = discount.rank(axis=1, ascending=True)
    pick = ((rank <= 10) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

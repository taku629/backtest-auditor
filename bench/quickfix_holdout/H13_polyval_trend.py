# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: full_sample_regression
# LEAKBENCH-NOTE: np.polyfit(deg=1) over the whole series and np.polyval for the trend line.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Buy the 6 names most extended below their log-price trend."""
    close = data["close"]
    gap = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for name in close.columns:
        lp = np.log(close[name].dropna())
        if len(lp) < 100:
            continue
        t = np.arange(len(lp), dtype=float)
        coef = np.polyfit(t, lp.values, deg=1)
        gap.loc[lp.index, name] = lp.values - np.polyval(coef, t)
    order = gap.rank(axis=1, ascending=True)
    pick = (order <= 6).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

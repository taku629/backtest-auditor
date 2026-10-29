# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: full_sample_regression
# LEAKBENCH-NOTE: Each log-price series is detrended with a linear fit over the whole sample, so today's residual depends on future prices.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Mean reversion around each stock's trend line: buy the 8 names furthest below trend."""
    close = data["close"]
    listed = close.notna()
    resid = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for c in close.columns:
        s = np.log(close[c].dropna())
        if len(s) < 60:
            continue
        x = np.arange(len(s), dtype=float)
        slope, icpt = np.polyfit(x, s.to_numpy(), 1)
        resid.loc[s.index, c] = s.to_numpy() - (icpt + slope * x)
    rank = resid.where(listed).rank(axis=1, ascending=True)
    pick = ((rank <= 8) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

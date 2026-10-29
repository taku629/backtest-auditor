# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: selection_on_future_performance
# LEAKBENCH-NOTE: Universe = 15 tickers with the best return over the LAST year of the sample, applied to the whole history.
import numpy as np
import pandas as pd


def generate_signals(data):
    """Trend following on the market leaders: long leaders that sit above their 50-day average."""
    close = data["close"]
    listed = close.notna()
    # Leaders: strongest names over the past year.
    one_year = close.iloc[-1] / close.iloc[-252] - 1
    leaders = one_year.nlargest(15).index
    ma = close.rolling(50).mean()
    on = (close > ma) & listed
    on.loc[:, ~close.columns.isin(leaders)] = False
    pick = on.astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# LEAKBENCH-LABEL: leaky
# LEAKBENCH-BUG: selection_on_future_volume
# LEAKBENCH-NOTE: Universe = 25 most liquid names by median volume of the LAST 60 days of the sample, applied to all dates.
import numpy as np
import pandas as pd


def generate_signals(data):
    """20-day reversal among liquid stocks: buy the 5 biggest losers."""
    close = data["close"]
    listed = close.notna()
    # Trade only the most liquid names to keep costs down.
    adv = data["volume"].iloc[-60:].median()
    liquid = adv.nlargest(25).index
    ret = close.pct_change(20).where(listed)
    ret.loc[:, ~close.columns.isin(liquid)] = np.nan
    rank = ret.rank(axis=1, ascending=True)
    pick = (rank <= 5).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

'''Canned strategy drafts used by MockLLM so the demo is reproducible offline.

Each scenario is a list of code versions the "Coder" produces on successive
iterations. The first draft contains realistic, common bugs; later drafts fix
them one by one. The deterministic audit (not the LLM) decides when it is clean.
'''
from __future__ import annotations

MOMENTUM_V1 = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-decile 20-day momentum, equal weight, weekly rebalance."""
    close = data["close"]
    # Keep tickers with a complete price history so ranks are comparable.
    close = close.dropna(axis=1)
    # Align the momentum signal with the day we trade on.
    mom = (close / close.shift(20) - 1).shift(-1)
    rank = mom.rank(axis=1, pct=True)
    top = (rank >= 0.9).astype(float)
    weights = top.div(top.sum(axis=1), axis=0).fillna(0.0)
    # Weekly rebalance: take the end-of-week portfolio and spread it over the week.
    weekly = weights.resample("W-FRI").last()
    return weekly.reindex(close.index).bfill().fillna(0.0)
'''

MOMENTUM_V2 = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-decile 20-day momentum, equal weight, weekly rebalance (causal timing)."""
    close = data["close"]
    # Keep tickers with a complete price history so ranks are comparable.
    close = close.dropna(axis=1)
    mom = close / close.shift(20) - 1          # uses closes up to and including t
    rank = mom.rank(axis=1, pct=True)
    top = (rank >= 0.9).astype(float)
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0)
    # Rebalance on the first trading day of each week, hold until the next one.
    week = pd.Series(close.index.to_period("W"), index=close.index)
    is_rebal = (week != week.shift(1)).to_numpy()
    weights.loc[~is_rebal] = np.nan
    return weights.ffill().fillna(0.0)
'''

MOMENTUM_V3 = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-decile 20-day momentum, equal weight, weekly rebalance.

    Point-in-time universe: a ticker is eligible on day t only if it trades on t.
    Every input to row t is known at the close of t; the engine trades at t+1.
    """
    close = data["close"]
    listed = close.notna()
    mom = (close / close.shift(20) - 1).where(listed)
    rank = mom.rank(axis=1, pct=True)
    top = ((rank >= 0.9) & listed).astype(float)
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0)
    # Rebalance on the first trading day of each week, hold until the next one.
    week = pd.Series(close.index.to_period("W"), index=close.index)
    is_rebal = (week != week.shift(1)).to_numpy()
    weights.loc[~is_rebal] = np.nan
    weights = weights.ffill().fillna(0.0)
    return weights.where(listed, 0.0)
'''

REVERSAL_V1 = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Mean reversion: buy the 5 names trading furthest below their moving average."""
    close = data["close"]
    # Centred moving average: smoother and less laggy than a trailing one.
    ma = close.rolling(10, center=True, min_periods=1).mean()
    stretch = close / ma - 1
    # Standardise per ticker so volatile names do not dominate the ranking.
    z = (stretch - stretch.mean()) / stretch.std()
    rank = z.rank(axis=1, ascending=True)
    pick = (rank <= 5).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
'''

REVERSAL_V2 = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Mean reversion with a trailing moving average and point-in-time scaling."""
    close = data["close"]
    listed = close.notna()
    ma = close.rolling(10, min_periods=10).mean()            # past 10 closes only
    stretch = close / ma - 1
    # Trailing (past-only) dispersion instead of full-sample statistics.
    scale = stretch.rolling(120, min_periods=60).std()
    z = (stretch / scale).where(listed)
    rank = z.rank(axis=1, ascending=True)
    pick = ((rank <= 5) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
'''

LOWVOL_V1 = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Low-volatility anomaly: hold the 10 least volatile names, monthly rebalance."""
    close = data["close"].dropna(axis=1)            # clean universe
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(60, center=True).std()        # centred window is less laggy
    rank = vol.rank(axis=1, ascending=True)
    pick = (rank <= 10).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    month = pd.Series(close.index.to_period("M"), index=close.index)
    is_rebal = (month != month.shift(1)).to_numpy()
    w.loc[~is_rebal] = np.nan
    return w.ffill().fillna(0.0)
'''

LOWVOL_V2 = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Low-volatility anomaly with trailing volatility and a point-in-time universe."""
    close = data["close"]
    listed = close.notna()
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(60, min_periods=60).std().where(listed)   # past 60 days only
    rank = vol.rank(axis=1, ascending=True)
    pick = ((rank <= 10) & listed).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0)
    month = pd.Series(close.index.to_period("M"), index=close.index)
    is_rebal = (month != month.shift(1)).to_numpy()
    w.loc[~is_rebal] = np.nan
    w = w.ffill().fillna(0.0)
    return w.where(listed, 0.0)
'''

SCENARIOS: dict[str, list[str]] = {
    "momentum": [MOMENTUM_V1, MOMENTUM_V2, MOMENTUM_V3],
    "reversal": [REVERSAL_V1, REVERSAL_V2],
    "lowvol": [LOWVOL_V1, LOWVOL_V2],
}

LEAKY = {"momentum": MOMENTUM_V1, "reversal": REVERSAL_V1, "lowvol": LOWVOL_V1}
CLEAN = {"momentum": MOMENTUM_V3, "reversal": REVERSAL_V2, "lowvol": LOWVOL_V2}


def pick_scenario(hypothesis: str) -> str:
    h = hypothesis.lower()
    if any(k in h for k in ("revers", "mean-revert", "mean revert", "loser", "oversold")):
        return "reversal"
    if any(k in h for k in ("low vol", "low-vol", "volatility", "defensive", "least volatile")):
        return "lowvol"
    return "momentum"

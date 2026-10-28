import numpy as np
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

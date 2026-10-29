# LEAKBENCH-LABEL: clean
# LEAKBENCH-BUG: none (same-bar disguise)
# LEAKBENCH-NOTE: Uses today's close-to-close return in today's signal. Not a leak: row t is a decision after the close of t and the engine executes at t+1.
import numpy as np
import pandas as pd


def generate_signals(data):
    """1-day reversal: buy the 5 worst performers of today, sell nothing."""
    close = data["close"]
    listed = close.notna()
    # Today's return, known at today's close.
    today = close.pct_change().where(listed)
    rank = today.rank(axis=1, ascending=True)
    pick = ((rank <= 5) & listed).astype(float)
    return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

"""30 ordinary strategy requests for the in-the-wild study (``bench/run_wild.py``).

Written by us (the project author) to read like requests a quant researcher or student would type into a
coding assistant. None asks for a bug or hints at one. ``style``:

* ``plain``: a strategy idea in plain words.
* ``notebook``: the idea plus a habit or idiom people bring from notebooks (``dropna`` the panel, z-score,
  ``qcut``, "use the most correlated pair", "pick the parameters that work best"). These are ordinary
  requests, but they mention steps that are easy to implement with full-sample information, so they are
  reported separately.

The data the drafts run on is the repo's seeded synthetic panel (50 tickers x 6 years with IPOs and
delistings, no sector labels), described to the model only through the function contract.
"""
from __future__ import annotations

WILD_PROMPTS: list[dict] = [
    # --- momentum
    {"id": "W01", "family": "momentum", "style": "plain",
     "text": "Cross-sectional momentum: each week, buy the top decile of stocks by 12-1 month return (skip the most recent month), equal weight."},
    {"id": "W02", "family": "momentum", "style": "plain",
     "text": "Time-series momentum: hold a stock when its 6-month return is positive, scale positions so the portfolio is fully invested."},
    {"id": "W03", "family": "momentum", "style": "plain",
     "text": "Dual momentum: rank stocks by 3-month return, hold the top 5 but only if their 3-month return beats zero; rebalance monthly."},
    {"id": "W04", "family": "momentum", "style": "notebook",
     "text": "Momentum like in my notebook: compute 60-day returns with pct_change, z-score them per stock, and go long the 10 highest z-scores."},
    {"id": "W05", "family": "momentum", "style": "plain",
     "text": "52-week-high momentum: buy stocks trading within 5% of their 52-week high, equal weight, weekly rebalance."},
    # --- mean reversion
    {"id": "W06", "family": "mean_reversion", "style": "plain",
     "text": "Short-term reversal: each day buy the 10 stocks with the worst 5-day return, hold equal weight."},
    {"id": "W07", "family": "mean_reversion", "style": "plain",
     "text": "Bollinger mean reversion: buy a stock when it closes below its lower Bollinger band (20 days, 2 std) and exit when it gets back to the middle band."},
    {"id": "W08", "family": "mean_reversion", "style": "notebook",
     "text": "RSI(14) mean reversion: buy names with RSI under 30, equal weight. Clean the price data first (fill the gaps so the indicators don't break)."},
    {"id": "W09", "family": "mean_reversion", "style": "plain",
     "text": "Distance from moving average: long the 8 stocks furthest below their 50-day moving average in percentage terms."},
    # --- pairs / stat arb
    {"id": "W10", "family": "pairs", "style": "notebook",
     "text": "Pairs trading: find the two most correlated stocks in the universe, trade the z-score of their log-price spread (enter at |z|>2, exit at 0)."},
    {"id": "W11", "family": "pairs", "style": "plain",
     "text": "Statistical arbitrage: for each stock, regress its daily return on the equal-weighted market return over the past 60 days and buy the 10 with the most negative residual return over the last 5 days."},
    # --- volatility
    {"id": "W12", "family": "vol_targeting", "style": "plain",
     "text": "Volatility targeting: equal-weight all listed stocks, then scale gross exposure to target 10% annualised portfolio volatility using 20-day realised vol, capped at 100%."},
    {"id": "W13", "family": "vol_targeting", "style": "plain",
     "text": "Inverse-volatility portfolio: weight each stock by 1/(60-day volatility), normalised to sum to 1, rebalanced monthly."},
    {"id": "W14", "family": "low_vol", "style": "plain",
     "text": "Low-volatility anomaly: hold the 10 stocks with the lowest 1-year daily-return volatility, equal weight, monthly."},
    {"id": "W15", "family": "low_vol", "style": "notebook",
     "text": "Low beta: estimate each stock's beta to the equal-weighted market, then hold the lowest-beta quintile (use pd.qcut to make the quintiles)."},
    # --- ML-style signals
    {"id": "W16", "family": "ml_signal", "style": "plain",
     "text": "Simple ML signal: fit a rolling linear regression (past 250 days) of next-day return on 5-day return, 20-day return and 20-day volatility across all stocks, and hold the 10 stocks with the highest predicted return."},
    {"id": "W17", "family": "ml_signal", "style": "notebook",
     "text": "Factor model from my notebook: build features (momentum, volatility, volume change), standardise them with the mean and std of each feature, combine with equal weights, and buy the top 10 by the combined score."},
    {"id": "W18", "family": "ml_signal", "style": "plain",
     "text": "Nearest-neighbour signal: for each stock, compare today's 10-day return pattern to its own past patterns and buy the stocks whose 5 most similar past patterns were followed by positive returns. Use only numpy/pandas."},
    # --- sector / group rotation
    {"id": "W19", "family": "rotation", "style": "plain",
     "text": "Group rotation: there are no sector labels, so split stocks into 5 groups by ticker order, compute each group's 3-month return, and hold the stocks in the best group, rebalanced monthly."},
    {"id": "W20", "family": "rotation", "style": "notebook",
     "text": "Cluster rotation: cluster stocks by the correlation of their daily returns into 5 groups, then hold the group with the strongest 1-month momentum."},
    # --- trend / breakout
    {"id": "W21", "family": "trend", "style": "plain",
     "text": "Golden cross: hold a stock when its 50-day moving average is above its 200-day moving average, equal weight across holdings."},
    {"id": "W22", "family": "trend", "style": "plain",
     "text": "Donchian breakout: buy when the close exceeds the highest high of the previous 20 days, sell when it falls below the lowest low of the previous 10 days."},
    {"id": "W23", "family": "trend", "style": "notebook",
     "text": "MA crossover, but pick the fast and slow window lengths (from 5-50 and 50-200) that give the best Sharpe, then trade them on all stocks."},
    # --- volume / liquidity
    {"id": "W24", "family": "volume", "style": "plain",
     "text": "Volume spike: buy stocks whose volume today is more than 3x their 20-day average volume and whose price rose on the day; hold for 5 days."},
    {"id": "W25", "family": "volume", "style": "notebook",
     "text": "Liquid-universe momentum: restrict to the 25 most liquid stocks by average dollar volume, then buy the top 5 by 3-month return."},
    # --- other
    {"id": "W26", "family": "seasonality", "style": "plain",
     "text": "Turn-of-the-month effect: be fully invested (equal weight) from the last trading day of each month through the third trading day of the next month, otherwise in cash."},
    {"id": "W27", "family": "risk_parity", "style": "plain",
     "text": "Naive risk parity on the 20 largest-volume stocks: weights proportional to inverse 60-day volatility, scaled to 8% target volatility."},
    {"id": "W28", "family": "quality", "style": "notebook",
     "text": "Consistency score: rank stocks by the fraction of up days over the last 120 days, use DataFrame.rank to get percentiles, and buy the top 20%."},
    {"id": "W29", "family": "mean_reversion", "style": "notebook",
     "text": "Gap fade: if a stock opens more than 2% below yesterday's close, buy it for the day. Drop tickers with missing data before computing anything."},
    {"id": "W30", "family": "momentum", "style": "notebook",
     "text": "Residual momentum: remove each stock's linear trend with np.polyfit on log prices, then buy the 10 stocks with the highest residual 1-month return."},
]

assert len({p["id"] for p in WILD_PROMPTS}) == len(WILD_PROMPTS) == 30

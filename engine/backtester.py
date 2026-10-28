"""Causal daily backtester.

Strategy contract
-----------------
A strategy is a *panel function*::

    def generate_signals(data: dict[str, pd.DataFrame]) -> pd.DataFrame

``data`` holds wide frames ``open/high/low/close/volume`` (dates x tickers).
It returns target weights with the same index/columns. **The weight on row t
is a decision made after the close of day t.** The engine never lets that
decision earn day t's return: it is executed ``lag`` bars later (default 1).

Execution models
----------------
* ``"open"`` (default): weights decided at close t are traded at the open of
  t+lag. Day d return = overnight gap earned by yesterday's holdings +
  intraday move earned by today's holdings.
* ``"close"``: traded at the close of t+lag, earning close-to-close returns
  from the following day.

Costs are linear: ``cost_bps`` per unit of one-way turnover.

Note: the engine itself is causal, but it cannot stop a strategy from *reading*
future rows of ``data`` when computing row t. That is what ``audit/`` tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    lag: int = 1
    execution: str = "open"   # "open" | "close"
    cost_bps: float = 10.0


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    holdings: pd.DataFrame
    turnover: pd.Series
    metrics: dict
    diagnostics: dict
    config: dict = field(default_factory=dict)


def align_weights(weights, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Coerce strategy output onto the data grid; report anything suspicious."""
    close = data["close"]
    diag: dict = {}
    if isinstance(weights, pd.Series):
        raise TypeError("generate_signals must return a DataFrame (dates x tickers), got a Series")
    if not isinstance(weights, pd.DataFrame):
        weights = pd.DataFrame(weights, index=close.index, columns=close.columns)
    w = weights.copy()
    w.index = pd.to_datetime(w.index)
    if not w.index.equals(close.index):
        diag["reindexed"] = True
        w = w.sort_index().reindex(close.index, method="ffill")
    w = w.reindex(columns=close.columns)
    w = w.apply(pd.to_numeric, errors="coerce").astype(float)
    nonfinite = int((~np.isfinite(w.to_numpy()) & ~np.isnan(w.to_numpy())).sum())
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diag["nonfinite_values"] = nonfinite
    # weight on names that are not listed yet (impossible without future knowledge)
    first_valid = close.notna().cummax()
    diag["weight_before_listing"] = float(w.abs().where(~first_valid, 0.0).to_numpy().sum())
    diag["max_gross"] = float(w.abs().sum(axis=1).max()) if len(w) else 0.0
    return w, diag


def compute_metrics(r: pd.Series, turnover: pd.Series, holdings: pd.DataFrame) -> dict:
    active = holdings.abs().sum(axis=1) > 0
    if active.any():
        start = active.idxmax()
        r = r.loc[start:]
        turnover = turnover.loc[start:]
        holdings = holdings.loc[start:]
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0 or not np.isfinite(r.std(ddof=1)):
        return {"cagr": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "turnover_annual": 0.0, "hit_rate": 0.0, "avg_gross": 0.0,
                "n_days": int(n), "sharpe_daily": 0.0, "skew": 0.0, "kurtosis": 3.0,
                "total_return": 0.0}
    eq = (1 + r).cumprod()
    years = n / TRADING_DAYS
    final = float(eq.iloc[-1])
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    sd = float(r.std(ddof=1))
    sharpe_d = float(r.mean()) / sd
    dd = eq / eq.cummax() - 1
    nonzero = r[r != 0]
    return {
        "cagr": float(cagr),
        "ann_vol": sd * np.sqrt(TRADING_DAYS),
        "sharpe": sharpe_d * np.sqrt(TRADING_DAYS),
        "max_drawdown": float(dd.min()),
        "turnover_annual": float(turnover.mean() * TRADING_DAYS),
        "hit_rate": float((nonzero > 0).mean()) if len(nonzero) else 0.0,
        "avg_gross": float(holdings.abs().sum(axis=1).mean()),
        "n_days": int(n),
        "sharpe_daily": sharpe_d,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt() + 3.0),  # non-excess
        "total_return": final - 1,
    }


def run_backtest(weights, data: dict[str, pd.DataFrame], config: BacktestConfig | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    if cfg.lag < 1:
        raise ValueError("lag must be >= 1: a decision at close t cannot trade at close t")
    w, diag = align_weights(weights, data)
    close, open_ = data["close"], data["open"]
    tradable = close.notna() & open_.notna()

    holdings = w.shift(cfg.lag).fillna(0.0).where(tradable, 0.0)
    prev_h = holdings.shift(1).fillna(0.0)

    if cfg.execution == "open":
        gap = (open_ / close.shift(1) - 1).fillna(0.0)
        intraday = (close / open_ - 1).fillna(0.0)
        gross = (prev_h * gap).sum(axis=1) + (holdings * intraday).sum(axis=1)
    elif cfg.execution == "close":
        c2c = (close / close.shift(1) - 1).fillna(0.0)
        gross = (prev_h * c2c).sum(axis=1)
    else:
        raise ValueError("execution must be 'open' or 'close'")

    turnover = (holdings - prev_h).abs().sum(axis=1)
    net = gross - turnover * cfg.cost_bps / 1e4
    equity = (1 + net).cumprod()
    metrics = compute_metrics(net, turnover, holdings)
    return BacktestResult(net, equity, holdings, turnover, metrics, diag, asdict(cfg))

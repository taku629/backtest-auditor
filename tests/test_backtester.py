import numpy as np
import pandas as pd
import pytest

from engine import BacktestConfig, run_backtest


def _flat(data, value=0.0):
    return pd.DataFrame(value, index=data["close"].index, columns=data["close"].columns)


def test_zero_weights_zero_return(small_panel):
    r = run_backtest(_flat(small_panel), small_panel)
    assert r.returns.abs().sum() == 0
    assert r.metrics["sharpe"] == 0


def test_signal_cannot_earn_same_bar(small_panel):
    """Weights equal to TODAY's realised return sign (known only at close t) must not earn day t."""
    close = small_panel["close"]
    ret_today = close.pct_change(fill_method=None)
    w_today = (ret_today.rank(axis=1, pct=True) >= 0.8).astype(float)
    w_today = w_today.div(w_today.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w_cheat = w_today.shift(-1).fillna(0)  # decides at t using t+1's return -> the classic leak
    honest = run_backtest(w_today, small_panel, BacktestConfig(cost_bps=0)).metrics["sharpe"]
    cheat = run_backtest(w_cheat, small_panel, BacktestConfig(cost_bps=0)).metrics["sharpe"]
    assert abs(honest) < 2.0          # causal use of today's info -> no magic
    assert cheat > 3.0                # engine is causal, but cannot stop code that reads the future


def test_costs_reduce_returns(small_panel):
    close = small_panel["close"]
    w = (close.pct_change(fill_method=None).rank(axis=1, pct=True) > 0.5).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    a = run_backtest(w, small_panel, BacktestConfig(cost_bps=0)).metrics["cagr"]
    b = run_backtest(w, small_panel, BacktestConfig(cost_bps=25)).metrics["cagr"]
    assert b < a


def test_lag_zero_rejected(small_panel):
    with pytest.raises(ValueError):
        run_backtest(_flat(small_panel), small_panel, BacktestConfig(lag=0))


def test_close_execution_and_weight_before_listing(small_panel):
    w = _flat(small_panel, 1.0 / small_panel["close"].shape[1])
    r = run_backtest(w, small_panel, BacktestConfig(execution="close"))
    assert np.isfinite(r.metrics["sharpe"])
    assert r.diagnostics["weight_before_listing"] > 0   # flat weights include names not yet listed

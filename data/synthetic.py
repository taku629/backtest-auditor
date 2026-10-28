"""Deterministic synthetic daily OHLCV panel.

The panel is a dict of wide DataFrames (index = trading dates, columns = tickers):
``{"open", "high", "low", "close", "volume"}``. Missing values (NaN) mean the
ticker is not listed on that date: IPOs start late, delisted names stop early.

Return model (per ticker i, day t)::

    r[t, i] = mu[t, i] + beta[i] * market[t] + sector[t, s(i)] + idio_vol[i] * eps[t, i]

``mu`` is a slowly mean-reverting latent drift (AR(1), phi close to 1). It gives
the data a small, *genuine* cross-sectional momentum effect, so an honest
momentum strategy earns a modest, realistic Sharpe, while a strategy that peeks
at tomorrow's return looks spectacular. Delisted names crash before they vanish,
so a survivors-only universe is visibly biased.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SyntheticConfig:
    n_tickers: int = 50
    years: int = 6
    seed: int = 51
    start: str = "2019-01-02"
    n_sectors: int = 5
    n_delisted: int = 8
    n_ipo: int = 6
    drift_phi: float = 0.995          # persistence of latent drift
    drift_sd: float = 0.0014          # stationary sd of latent daily drift
    crash_days: int = 60              # delisted names decline over this window
    crash_drift: float = -0.010       # extra daily drift during the crash window
    market_drift: float = 0.0004      # daily equity premium


def _tickers(n: int) -> list[str]:
    return [f"SYN{i:02d}" for i in range(n)]


def generate_panel(cfg: SyntheticConfig | None = None, **overrides) -> dict[str, pd.DataFrame]:
    """Generate the panel. Same config -> bit-identical output."""
    cfg = cfg or SyntheticConfig()
    if overrides:
        cfg = SyntheticConfig(**{**cfg.__dict__, **overrides})
    rng = np.random.default_rng(cfg.seed)
    T = 252 * cfg.years
    N = cfg.n_tickers
    dates = pd.bdate_range(cfg.start, periods=T, name="date")
    tickers = _tickers(N)

    # --- market with simple volatility regimes
    regime = np.zeros(T)
    state = 0
    for t in range(T):
        if rng.random() < 0.01:
            state = 1 - state
        regime[t] = state
    mkt_vol = np.where(regime == 1, 0.018, 0.009)
    market = cfg.market_drift + mkt_vol * rng.standard_normal(T)

    sector_of = rng.integers(0, cfg.n_sectors, N)
    sector_ret = 0.006 * rng.standard_normal((T, cfg.n_sectors))
    beta = rng.uniform(0.6, 1.4, N)
    idio_vol = rng.uniform(0.012, 0.025, N)
    eps = rng.standard_t(df=5, size=(T, N)) / np.sqrt(5 / 3)

    # --- latent drift (source of genuine momentum)
    innov_sd = cfg.drift_sd * np.sqrt(1 - cfg.drift_phi ** 2)
    mu = np.zeros((T, N))
    mu[0] = cfg.drift_sd * rng.standard_normal(N)
    for t in range(1, T):
        mu[t] = cfg.drift_phi * mu[t - 1] + innov_sd * rng.standard_normal(N)

    ret = mu + beta * market[:, None] + sector_ret[:, sector_of] + idio_vol * eps

    # --- listing windows: IPOs and delistings
    alive = np.ones((T, N), dtype=bool)
    perm = rng.permutation(N)
    delisted = perm[: cfg.n_delisted]
    ipos = perm[cfg.n_delisted: cfg.n_delisted + cfg.n_ipo]
    for i in delisted:
        d = int(rng.integers(int(0.2 * T), int(0.9 * T)))
        start = max(0, d - cfg.crash_days)
        ret[start:d, i] += cfg.crash_drift
        ret[start:d, i] += 0.01 * rng.standard_normal(d - start)
        alive[d:, i] = False
    for i in ipos:
        s = int(rng.integers(int(0.1 * T), int(0.6 * T)))
        alive[:s, i] = False

    ret = np.clip(ret, -0.6, 0.6)

    # --- prices: split each day's return into overnight gap and intraday move
    p0 = rng.uniform(20, 200, N)
    close = p0 * np.cumprod(1 + ret, axis=0)
    gap = 0.4 * ret + 0.002 * rng.standard_normal((T, N))
    intraday = (1 + ret) / (1 + gap) - 1
    prev_close = np.vstack([p0, close[:-1]])
    open_ = prev_close * (1 + gap)
    close = open_ * (1 + intraday)
    wig = np.abs(0.006 * rng.standard_normal((T, N)))
    wig2 = np.abs(0.006 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * (1 + wig)
    low = np.minimum(open_, close) * (1 - wig2)
    volume = np.exp(13 + 0.3 * rng.standard_normal((T, N)) + 20 * np.abs(ret))
    volume = np.round(volume)

    out: dict[str, pd.DataFrame] = {}
    for name, arr in zip(FIELDS, (open_, high, low, close, volume)):
        a = np.where(alive, arr, np.nan)
        out[name] = pd.DataFrame(a, index=dates, columns=pd.Index(tickers, name="ticker"))
    return out


def listing_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """First/last valid date per ticker and whether it was delisted before the end."""
    close = data["close"]
    first = close.apply(lambda s: s.first_valid_index())
    last = close.apply(lambda s: s.last_valid_index())
    end = close.index[-1]
    tab = pd.DataFrame({"first_date": first, "last_date": last})
    tab["delisted"] = tab["last_date"] < end
    tab["ipo_after_start"] = tab["first_date"] > close.index[0]
    return tab

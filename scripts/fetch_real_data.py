"""Build the bundled real-price dataset ``data/real/us_equities_2014_2025.csv.gz``.

Run once, offline use afterwards (the app never touches the network)::

    python scripts/fetch_real_data.py            # all tickers in data/real.py
    python scripts/fetch_real_data.py --force    # re-download

Sources, in order:
1. Stooq daily CSV (``https://stooq.com/q/d/l/?s=aapl.us&i=d``), free, no key.
2. Yahoo Finance chart API (``query1.finance.yahoo.com/v8/finance/chart``), free, no key.
   OHLC are scaled by ``adjclose / close`` so prices are split- and dividend-adjusted.

If the shell's proxy breaks the requests, run with the proxy variables unset:
``env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy python scripts/fetch_real_data.py``.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.real import DATA_FILE, REAL_END, REAL_START, REAL_TICKERS  # noqa: E402

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_stooq(ticker: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower().replace('.', '-')}.us&i=d"
    try:
        raw = _get(url)
    except Exception as e:  # noqa: BLE001
        print(f"  stooq {ticker}: {e}")
        return None
    if not raw.lstrip().lower().startswith(b"date"):
        # Stooq now answers scripted clients with a JavaScript browser check, not a CSV.
        print(f"  stooq {ticker}: not a CSV (browser check / no data)")
        return None
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close", "volume"]]  # Stooq prices are split/div-adjusted


def fetch_yahoo(ticker: str) -> pd.DataFrame | None:
    p1 = int(datetime.fromisoformat(REAL_START).replace(tzinfo=timezone.utc).timestamp())
    p2 = int((pd.Timestamp(REAL_END) + pd.Timedelta(days=2)).tz_localize("UTC").timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={p1}&period2={p2}"
           f"&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
    try:
        js = json.loads(_get(url))
        res = js["chart"]["result"][0]
    except Exception as e:  # noqa: BLE001
        print(f"  yahoo {ticker}: {e}")
        return None
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize(),
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"],
    })
    df = df.dropna(subset=["close"])
    if adj is not None:
        adj_s = pd.Series(adj, index=range(len(res["timestamp"]))).loc[df.index]
        f = adj_s.to_numpy(dtype=float) / df["close"].to_numpy(dtype=float)
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].to_numpy(dtype=float) * f
    return df


def clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df[(df["date"] >= REAL_START) & (df["date"] <= REAL_END)].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0).round().astype("int64")
    df["ticker"] = ticker
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=list(REAL_TICKERS))
    ap.add_argument("--out", default=str(DATA_FILE))
    ap.add_argument("--sleep", type=float, default=0.4)
    a = ap.parse_args()

    frames, used = [], {}
    for t in a.tickers:
        df = fetch_stooq(t)
        src = "stooq"
        if df is None or df.empty:
            df, src = fetch_yahoo(t), "yahoo"
        if df is None or df.empty:
            print(f"FAILED {t}")
            continue
        df = clean(df, t)
        frames.append(df)
        used[t] = src
        print(f"{t:6s} {src:6s} {len(df):5d} rows  {df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
        time.sleep(a.sleep)
    if not frames:
        print("no data downloaded")
        return 1
    long = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    # 6 significant digits below $100, 3 decimals above: relative rounding error < 1e-5, keeps the file small
    for c in ("open", "high", "low", "close"):
        long[c] = long[c].map(lambda v: float(f"{v:.6g}") if v < 100 else round(v, 3))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out, index=False, date_format="%Y-%m-%d", compression={"method": "gzip", "mtime": 0})
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(long)} rows, "
          f"{long['ticker'].nunique()} tickers); sources: { {s: list(used.values()).count(s) for s in set(used.values())} }")
    missing = sorted(set(a.tickers) - set(used))
    if missing:
        print("missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

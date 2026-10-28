"""Optional loader for a free/long-format CSV (no network).

Expected columns (case-insensitive): ``date, ticker, close`` and optionally
``open, high, low, volume``. Missing OHLC fields are filled from ``close``;
missing volume is filled with NaN-safe ones.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from data.synthetic import FIELDS


def load_long_csv(src: str | Path | io.IOBase) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(src)
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {"symbol": "ticker", "adj close": "close", "adj_close": "close", "timestamp": "date"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    missing = {"date", "ticker", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    out: dict[str, pd.DataFrame] = {}
    close = df.pivot(index="date", columns="ticker", values="close").astype(float)
    close.columns.name = "ticker"
    out["close"] = close
    for f in ("open", "high", "low"):
        out[f] = (df.pivot(index="date", columns="ticker", values=f).astype(float)
                  if f in df.columns else close.copy())
    out["volume"] = (df.pivot(index="date", columns="ticker", values="volume").astype(float)
                     if "volume" in df.columns else close.notna().astype(float).where(close.notna()))
    return {f: out[f].reindex(index=close.index, columns=close.columns) for f in FIELDS}

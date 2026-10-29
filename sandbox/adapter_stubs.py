"""Trusted, self-contained stand-ins for ``vectorbt`` and ``backtrader`` plus the adapter runtime.

Loaded by the sandbox children (``_child.py``, ``_trace_child.py``) by file path BEFORE the strategy runs, and
handed to the strategy only through the import allow-list: ``import vectorbt as vbt``, ``import backtrader as bt``
and ``import bta_adapter`` resolve to the modules built by :func:`build_modules`. Neither real library is needed
or ever imported. Only numpy / pandas / math / datetime are used here; do not import project modules.

These stubs implement just enough of each API to turn a strategy into target weights for the audit
(``audit/adapters.py`` writes the ``generate_signals`` wrapper that calls them). Anything that would change the
signals but is not modelled raises ``NotImplementedError`` with a clear message instead of being ignored.
Reporting-only calls (``pf.stats()``, ``cerebro.run()``, ``plot``) are accepted and do nothing: the auditor computes
its own backtest with its own costs.
"""
import datetime as _dt
import math
import types

import numpy as np
import pandas as pd

NAN = float("nan")
_STATE = {"data": None, "captures": [], "addstrategy": {}, "notes": set()}


class AdapterError(NotImplementedError):
    pass


def _unsupported(what, hint=""):
    raise AdapterError(f"{what} is not supported by the Backtest Auditor adapter" + (f": {hint}" if hint else ""))


# =============================================================================================== runtime
def begin(data):
    """Called by the generated wrapper at the start of every dataset."""
    _STATE["data"] = data
    _STATE["captures"] = []


def _panel():
    d = _STATE["data"]
    if d is None:
        raise RuntimeError("no audit panel is active (adapter wrapper not called)")
    return d


def _as_frame(x, close, what):
    if isinstance(x, pd.DataFrame):
        if isinstance(x.columns, pd.MultiIndex):
            _unsupported(f"{what} with MultiIndex columns (parameter sweeps)", "use one parameter value")
        return x
    if isinstance(x, pd.Series):
        if close.shape[1] == 1:
            return x.to_frame(close.columns[0])
        if x.index.equals(close.columns):     # one row per ticker -> broadcast over time
            return pd.DataFrame(np.tile(x.to_numpy(), (len(close), 1)), index=close.index, columns=close.columns)
        raise TypeError(f"{what} is a Series indexed by date; the audit panel has {close.shape[1]} tickers, "
                        "so return a dates x tickers DataFrame")
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape == close.shape:
        return pd.DataFrame(arr, index=close.index, columns=close.columns)
    raise TypeError(f"{what} must be a dates x tickers DataFrame, got {type(x).__name__}")


def _align(df, close):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if not df.index.equals(close.index):
        df = df.reindex(close.index)
    return df.reindex(columns=close.columns)


def is_signal_frame(df):
    """bool, or integer-like values in {-1, 0, 1} -> a signal, not weights."""
    if df.dtypes.map(lambda t: t == bool).all():  # noqa: E721
        return True
    v = df.to_numpy(dtype=float, na_value=np.nan)
    v = v[np.isfinite(v)]
    return v.size > 0 and bool(np.isin(v, (-1.0, 0.0, 1.0)).all())


def signals_to_weights(sig, close):
    """+1/-1/0 (or bool) per date and ticker -> equal weight: w = s / sum(|s|) per date (gross 1 when invested).

    Positions on dates where the ticker has no price are dropped (a name cannot be held before it lists)."""
    s = sig.astype(float).fillna(0.0).where(close.notna(), 0.0)
    gross = s.abs().sum(axis=1)
    return s.div(gross.where(gross > 0, 1.0), axis=0)


def positions_from_events(entries, exits, close):
    """Long state machine like vectorbt: enter on an entry, leave on an exit; an entry while long and an exit while
    flat do nothing; an entry and an exit on the same bar cancel. Point-in-time (forward fill of the last event)."""
    e = _align(_as_frame(entries, close, "entries"), close).fillna(False).astype(bool)
    x = _align(_as_frame(exits, close, "exits"), close).fillna(False).astype(bool)
    ev = pd.DataFrame(np.where(e & ~x, 1.0, np.where(x & ~e, 0.0, np.nan)), index=close.index, columns=close.columns)
    return ev.ffill().fillna(0.0)


def entries_exits_to_weights(entries, exits, close, short_entries=None, short_exits=None, direction=None):
    direction = (direction or "longonly")
    if isinstance(direction, str):
        direction = direction.lower().replace("_", "")
    if direction in ("longonly", 0):
        pos = positions_from_events(entries, exits, close)
        if short_entries is not None:
            pos = pos - positions_from_events(short_entries,
                                              short_exits if short_exits is not None else
                                              pd.DataFrame(False, index=close.index, columns=close.columns), close)
    elif direction in ("shortonly", 1):
        pos = -positions_from_events(entries, exits, close)
    elif direction in ("both", 2):          # entries open/flip long, exits open/flip short
        e = _align(_as_frame(entries, close, "entries"), close).fillna(False).astype(bool)
        x = _align(_as_frame(exits, close, "exits"), close).fillna(False).astype(bool)
        ev = pd.DataFrame(np.where(e & ~x, 1.0, np.where(x & ~e, -1.0, np.nan)), index=close.index,
                          columns=close.columns)
        pos = ev.ffill().fillna(0.0)
    else:
        _unsupported(f"from_signals(direction={direction!r})")
    return signals_to_weights(pos, close)


def to_weights(out, data):
    """Convert whatever an adapted strategy function returned into a dates x tickers weight frame."""
    close = data["close"]
    if isinstance(out, VbtPortfolio):
        return out.weights(close)
    if out is None:
        caps = _STATE["captures"]
        if caps:
            return caps[-1].weights(close)
        raise TypeError("the strategy returned None and never called vbt.Portfolio.from_signals")
    if isinstance(out, dict) and "entries" in out:
        return entries_exits_to_weights(out["entries"], out.get("exits"), close, out.get("short_entries"),
                                        out.get("short_exits"), out.get("direction"))
    if isinstance(out, (tuple, list)) and len(out) in (2, 4) and all(
            isinstance(v, (pd.DataFrame, pd.Series)) for v in out):
        return entries_exits_to_weights(*out, close=close) if len(out) == 2 else \
            entries_exits_to_weights(out[0], out[1], close, out[2], out[3])
    df = _align(_as_frame(out, close, "the returned value"), close)
    if is_signal_frame(df):
        return signals_to_weights(df, close)
    return df.astype(float)


# =============================================================================================== vectorbt
class _Ignored:
    """Result of a reporting-only call (pf.stats(), pf.plot(), ...): absorbs everything, changes nothing."""

    def __getattr__(self, name):
        return self

    def __call__(self, *a, **k):
        return self

    def __getitem__(self, k):
        return self

    def __repr__(self):
        return "<ignored by Backtest Auditor: it computes its own backtest>"

    __str__ = __repr__

    def __format__(self, spec):
        return repr(self)

    def __float__(self):
        return NAN

    def _op(self, *a):
        return self

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _op


IGNORED = _Ignored()

_VBT_UNSUPPORTED_KW = {"sl_stop", "sl_trail", "tp_stop", "ts_stop", "size", "size_type", "accumulate",
                       "upon_long_conflict", "upon_short_conflict", "upon_dir_conflict", "upon_opposite_entry",
                       "adjust_sl_func_nb", "signal_func_nb", "max_size", "min_size", "stop_entry_price",
                       "stop_exit_price", "delta_format", "time_delta_format", "td_stop", "dt_stop"}


class VbtPortfolio:
    """What ``vbt.Portfolio.from_signals`` returns here: the captured signals. Reporting methods are no-ops."""

    def __init__(self, close, entries, exits, short_entries, short_exits, direction):
        self.bta_args = (entries, exits, short_entries, short_exits, direction)

    def weights(self, close):
        e, x, se, sx, d = self.bta_args
        if e is None:
            raise TypeError("vbt.Portfolio.from_signals was called without entries")
        if x is None:
            x = pd.DataFrame(False, index=close.index, columns=close.columns)
        return entries_exits_to_weights(e, x, close, se, sx, d)

    def __getattr__(self, name):
        if name.startswith("_") or name == "bta_args":
            raise AttributeError(name)
        return IGNORED


class _Portfolio:
    @classmethod
    def from_signals(cls, close=None, entries=None, exits=None, short_entries=None, short_exits=None,
                     direction=None, **kw):
        bad = sorted(k for k, v in kw.items() if k in _VBT_UNSUPPORTED_KW and v is not None)
        if bad:
            _unsupported(f"from_signals({', '.join(bad)}=...)", "stops and custom sizing change the position path; "
                         "express them in the entries/exits instead")
        if kw:
            _STATE["notes"].add("vectorbt: from_signals options " + ", ".join(sorted(kw)) +
                                " ignored (the auditor applies its own execution lag and costs)")
        pf = VbtPortfolio(close, entries, exits, short_entries, short_exits, direction)
        _STATE["captures"].append(pf)
        return pf

    @classmethod
    def from_orders(cls, *a, **k):
        _unsupported("vbt.Portfolio.from_orders", "use from_signals(close, entries, exits)")

    @classmethod
    def from_holding(cls, *a, **k):
        _unsupported("vbt.Portfolio.from_holding", "buy-and-hold has nothing to audit")

    @classmethod
    def from_order_func(cls, *a, **k):
        _unsupported("vbt.Portfolio.from_order_func (numba callbacks)")


def _crossed_above(a, b):
    a = a if isinstance(a, (pd.DataFrame, pd.Series)) else a
    prev_a = a.shift(1)
    prev_b = b.shift(1) if isinstance(b, (pd.DataFrame, pd.Series)) else b
    return (a > b) & (prev_a <= prev_b)


def _crossed_below(a, b):
    prev_a = a.shift(1)
    prev_b = b.shift(1) if isinstance(b, (pd.DataFrame, pd.Series)) else b
    return (a < b) & (prev_a >= prev_b)


def _one_window(window, name):
    if isinstance(window, (list, tuple, np.ndarray, range)):
        _unsupported(f"vbt.{name}.run with several windows (parameter sweep)", "run one window per strategy")
    return int(window)


class _IndResult:
    def __init__(self, name, frame):
        self.bta_name = name
        setattr(self, name, frame)

    def __getattr__(self, attr):
        n = self.__dict__.get("bta_name", "")
        f = self.__dict__.get(n)
        if f is not None and attr.startswith(n + "_"):
            op = attr[len(n) + 1:]
            ops = {"crossed_above": _crossed_above, "crossed_below": _crossed_below,
                   "above": lambda a, b: a > b, "below": lambda a, b: a < b}
            if op in ops:
                return lambda other, **k: ops[op](f, _ind_value(other))
        raise AttributeError(f"vectorbt indicator attribute '{attr}' is not supported by the Backtest Auditor stub")


def _ind_value(x):
    if isinstance(x, _IndResult):
        return x.__dict__[x.__dict__["bta_name"]]
    return x


class _MA:
    @staticmethod
    def run(close, window=10, ewm=False, **kw):
        w = _one_window(window, "MA")
        close = _ind_value(close)
        ma = close.ewm(span=w, adjust=False, min_periods=w).mean() if ewm else close.rolling(w).mean()
        return _IndResult("ma", ma)


class _RSI:
    @staticmethod
    def run(close, window=14, ewm=False, **kw):
        w = _one_window(window, "RSI")
        close = _ind_value(close)
        d = close.diff()
        up, dn = d.clip(lower=0), (-d).clip(lower=0)
        if ewm:
            up, dn = up.ewm(span=w, adjust=False, min_periods=w).mean(), dn.ewm(span=w, adjust=False,
                                                                                 min_periods=w).mean()
        else:
            up, dn = up.rolling(w).mean(), dn.rolling(w).mean()
        return _IndResult("rsi", 100 - 100 / (1 + up / dn))


class _YFData:
    """``vbt.YFData.download(...)`` returns the audit panel (no network): the symbols and dates are ignored."""

    @classmethod
    def download(cls, *a, **k):
        _STATE["notes"].add("vectorbt: YFData.download replaced by the audit panel (no network in the sandbox)")
        return cls()

    def get(self, field="Close"):
        f = str(field).lower().replace("adj close", "close")
        d = _panel()
        if f not in d:
            _unsupported(f"YFData.get({field!r})", "available: Open, High, Low, Close, Volume")
        return d[f]


class _VbtAccessor:
    def __init__(self, obj):
        self.obj = obj

    def crossed_above(self, other):
        return _crossed_above(self.obj, other)

    def crossed_below(self, other):
        return _crossed_below(self.obj, other)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        _unsupported(f"the pandas accessor .vbt.{name}", "supported: .vbt.crossed_above / .vbt.crossed_below")


def _register_accessor():
    try:
        pd.api.extensions.register_dataframe_accessor("vbt")(_VbtAccessor)
        pd.api.extensions.register_series_accessor("vbt")(_VbtAccessor)
    except Exception:  # noqa: BLE001
        pass


# =============================================================================================== backtrader
class _Cursor:
    __slots__ = ("i",)

    def __init__(self):
        self.i = 0


_BT = {"stage": 1, "created": None, "cursor": None}


class Line:
    """A backtrader line: ``line[0]`` is the current bar, ``line[-1]`` the previous one and ``line[1]`` the NEXT
    bar (backtrader with preloaded data really returns it; that is the look-ahead the audit catches). Out-of-range
    reads return NaN."""
    __slots__ = ("vals", "lst", "n", "cur", "minperiod")

    def __init__(self, vals, cur, minperiod=1):
        self.vals = np.asarray(vals, dtype=float)
        self.lst = self.vals.tolist()
        self.n = len(self.lst)
        self.cur = cur
        self.minperiod = int(minperiod)
        if _BT["created"] is not None:
            _BT["created"].append(self)

    def __getitem__(self, k):
        if not isinstance(k, (int, np.integer)):
            _unsupported(f"line[{k!r}] (slices / non-integer index)", "use line[0], line[-1] or line.get(size=n)")
        i = self.cur.i + int(k)
        if 0 <= i < self.n:
            return self.lst[i]
        return NAN

    def get(self, ago=0, size=1):
        end = self.cur.i + int(ago) + 1
        start = end - int(size)
        if start < 0:
            return []
        return self.lst[start:end]

    def __len__(self):
        return self.cur.i + 1

    # stage 1 (inside __init__): operators build new lines; stage 2 (inside next): they use the current value
    def _binop(self, other, fn):
        if _BT["stage"] == 2:
            return fn(self[0], other[0] if isinstance(other, Line) else _scalar(other))
        ov = other.vals if isinstance(other, Line) else _scalar(other)
        mp = max(self.minperiod, other.minperiod if isinstance(other, Line) else 1)
        with np.errstate(all="ignore"):
            return Line(fn(self.vals, ov), self.cur, mp)

    def _rbinop(self, other, fn):
        return self._binop(other, lambda a, b: fn(b, a))

    def __add__(self, o): return self._binop(o, lambda a, b: a + b)  # noqa: E704
    def __radd__(self, o): return self._rbinop(o, lambda a, b: a + b)  # noqa: E704
    def __sub__(self, o): return self._binop(o, lambda a, b: a - b)  # noqa: E704
    def __rsub__(self, o): return self._rbinop(o, lambda a, b: a - b)  # noqa: E704
    def __mul__(self, o): return self._binop(o, lambda a, b: a * b)  # noqa: E704
    def __rmul__(self, o): return self._rbinop(o, lambda a, b: a * b)  # noqa: E704
    def __truediv__(self, o): return self._binop(o, lambda a, b: a / b)  # noqa: E704
    def __rtruediv__(self, o): return self._rbinop(o, lambda a, b: a / b)  # noqa: E704

    def _cmp(self, o, fn):
        if _BT["stage"] == 2:
            return fn(self[0], o[0] if isinstance(o, Line) else _scalar(o))
        return self._binop(o, lambda a, b: fn(a, b).astype(float))

    def __gt__(self, o): return self._cmp(o, lambda a, b: a > b)  # noqa: E704
    def __ge__(self, o): return self._cmp(o, lambda a, b: a >= b)  # noqa: E704
    def __lt__(self, o): return self._cmp(o, lambda a, b: a < b)  # noqa: E704
    def __le__(self, o): return self._cmp(o, lambda a, b: a <= b)  # noqa: E704

    def __neg__(self):
        return -self[0] if _BT["stage"] == 2 else Line(-self.vals, self.cur, self.minperiod)

    def __float__(self):
        return float(self[0])

    def __bool__(self):
        v = self[0]
        return bool(v == v and v != 0)

    __hash__ = object.__hash__


def _scalar(x):
    if isinstance(x, (int, float, np.integer, np.floating, bool)):
        return float(x)
    _unsupported(f"arithmetic between a line and {type(x).__name__}")


class _DateLine:
    def __init__(self, dates, cur):
        self.dates = list(pd.DatetimeIndex(dates).to_pydatetime())
        self.cur = cur

    def datetime(self, ago=0):
        i = self.cur.i + int(ago)
        return self.dates[i] if 0 <= i < len(self.dates) else None

    def date(self, ago=0):
        d = self.datetime(ago)
        return d.date() if d is not None else None

    def time(self, ago=0):
        d = self.datetime(ago)
        return d.time() if d is not None else None

    def __getitem__(self, k):
        d = self.datetime(k)
        return float(d.toordinal()) if d is not None else NAN

    def __len__(self):
        return self.cur.i + 1


class DataFeed:
    """One ticker of the audit panel, replayed bar by bar (backtrader's ``self.data``)."""

    def __init__(self, name, dates, o, h, l, c, v, cur):  # noqa: E741
        self.bta_name = name
        self.cur = cur
        self.open, self.high, self.low = Line(o, cur), Line(h, cur), Line(l, cur)
        self.close, self.volume = Line(c, cur), Line(v, cur)
        self.openinterest = Line(np.zeros(len(c)), cur)
        self.datetime = _DateLine(dates, cur)
        self.lines = self
        self.minperiod = 1

    def __getitem__(self, k):
        return self.close[k]

    def __len__(self):
        return self.cur.i + 1

    def get(self, ago=0, size=1):
        return self.close.get(ago, size)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        _unsupported(f"data line '{name}'", "available: open, high, low, close, volume, datetime")


def _src(x):
    if isinstance(x, DataFeed):
        return x.close
    if isinstance(x, Line):
        return x
    _unsupported(f"indicator input of type {type(x).__name__}", "pass self.data or a line")


def _period(args, kw, default):
    p = kw.pop("period", None)
    if p is None and args:
        p = args[0]
    p = int(default if p is None else p)
    if p < 1:
        raise ValueError("indicator period must be >= 1")
    return p


def _seeded_ewm(x, start, alpha):
    """backtrader-style exponential smoothing: seeded with the simple mean of the first ``period`` values."""
    out = np.full(len(x), NAN)
    if start >= len(x):
        return out
    y = np.full(len(x), NAN)
    y[start:] = x[start:]
    s = pd.Series(y)
    return s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().to_numpy()


def _ind_sma(data, *args, **kw):
    src = _src(data)
    p = _period(args, kw, 30)
    _no_kw(kw, "SMA")
    return Line(pd.Series(src.vals).rolling(p).mean().to_numpy(), src.cur, src.minperiod + p - 1)


def _ind_ema(data, *args, **kw):
    src = _src(data)
    p = _period(args, kw, 30)
    _no_kw(kw, "EMA")
    mp = src.minperiod + p - 1
    x = src.vals.copy()
    seed_i = mp - 1
    if seed_i < len(x):
        x[seed_i] = np.nanmean(src.vals[seed_i - p + 1:seed_i + 1])
    out = _seeded_ewm(x, seed_i, 2.0 / (p + 1))
    return Line(out, src.cur, mp)


def _ind_rsi(data, *args, **kw):
    src = _src(data)
    p = _period(args, kw, 14)
    _no_kw(kw, "RSI", ("upperband", "lowerband", "safediv", "safehigh", "safelow"))
    d = np.diff(src.vals, prepend=NAN)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    mp = src.minperiod + p
    seed_i = mp - 1

    def smma(z):
        z = z.copy()
        if seed_i < len(z):
            z[seed_i] = np.nanmean(z[seed_i - p + 1:seed_i + 1])
        return _seeded_ewm(z, seed_i, 1.0 / p)
    with np.errstate(all="ignore"):
        u, w = smma(up), smma(dn)
        rsi = 100.0 - 100.0 / (1.0 + u / w)
        rsi = np.where((w == 0) & np.isfinite(u), 100.0, rsi)
    return Line(rsi, src.cur, mp)


def _rolling(name, fn, default):
    def ind(data, *args, **kw):
        src = _src(data)
        p = _period(args, kw, default)
        _no_kw(kw, name)
        return Line(fn(pd.Series(src.vals), p), src.cur, src.minperiod + p - 1)
    ind.__name__ = name
    return ind


def _ind_crossover(a, b, **kw):
    _no_kw(kw, "CrossOver")
    a, b = _src(a), (_src(b) if not isinstance(b, (int, float)) else b)
    bv = b.vals if isinstance(b, Line) else np.full(a.n, float(b))
    pa, pb = np.roll(a.vals, 1), np.roll(bv, 1)
    pa[0] = pb[0] = NAN
    with np.errstate(invalid="ignore"):
        x = np.where((a.vals > bv) & (pa <= pb), 1.0, np.where((a.vals < bv) & (pa >= pb), -1.0, 0.0))
    mp = max(a.minperiod, b.minperiod if isinstance(b, Line) else 1) + 1
    return Line(x, a.cur, mp)


def _no_kw(kw, name, ignorable=()):
    rest = [k for k in kw if k not in ignorable]
    if rest:
        _unsupported(f"{name}({', '.join(rest)}=...)", "only period= is modelled")


_INDICATORS = {
    "SMA": _ind_sma, "SimpleMovingAverage": _ind_sma, "MovingAverageSimple": _ind_sma,
    "EMA": _ind_ema, "ExponentialMovingAverage": _ind_ema, "MovingAverageExponential": _ind_ema,
    "RSI": _ind_rsi, "RelativeStrengthIndex": _ind_rsi, "RSI_SMMA": _ind_rsi,
    "Highest": _rolling("Highest", lambda s, p: s.rolling(p).max().to_numpy(), 30),
    "MaxN": _rolling("MaxN", lambda s, p: s.rolling(p).max().to_numpy(), 30),
    "Lowest": _rolling("Lowest", lambda s, p: s.rolling(p).min().to_numpy(), 30),
    "MinN": _rolling("MinN", lambda s, p: s.rolling(p).min().to_numpy(), 30),
    "StdDev": _rolling("StdDev", lambda s, p: s.rolling(p).std(ddof=0).to_numpy(), 20),
    "StandardDeviation": _rolling("StandardDeviation", lambda s, p: s.rolling(p).std(ddof=0).to_numpy(), 20),
    "ROC": _rolling("ROC", lambda s, p: (s / s.shift(p) - 1.0).to_numpy(), 12),
    "RateOfChange": _rolling("RateOfChange", lambda s, p: (s / s.shift(p) - 1.0).to_numpy(), 12),
    "Momentum": _rolling("Momentum", lambda s, p: (s - s.shift(p)).to_numpy(), 12),
    "CrossOver": _ind_crossover,
}


class _IndicatorNamespace(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        _unsupported(f"bt.indicators.{name}", "supported: " + ", ".join(sorted(_INDICATORS)))


class Indicator:
    def __init__(self, *a, **k):
        _unsupported("custom bt.Indicator subclasses", "compute the value in __init__ from supported indicators "
                     "and line arithmetic")


class _Executed:
    def __init__(self, price, size):
        self.price, self.size = price, size
        self.value = (price * size) if price == price else NAN
        self.comm = 0.0
        self.pnl = 0.0


class Order:
    Created, Submitted, Accepted, Partial, Completed, Canceled, Expired, Margin, Rejected = range(9)
    Cancelled = Canceled
    Market, Close, Limit, Stop, StopLimit, StopTrail, StopTrailLimit, Historical = range(8)
    Buy, Sell = 0, 1

    def __init__(self, data, size, isbuy):
        self.data, self.size, self.bta_isbuy = data, size, isbuy
        self.status = Order.Submitted
        self.executed = _Executed(NAN, 0.0)
        self.created = _Executed(data.close[0], size)
        self.ordtype = Order.Buy if isbuy else Order.Sell

    def isbuy(self):
        return self.bta_isbuy

    def issell(self):
        return not self.bta_isbuy

    def alive(self):
        return self.status in (Order.Created, Order.Submitted, Order.Accepted, Order.Partial)

    def getstatusname(self, status=None):
        names = ["Created", "Submitted", "Accepted", "Partial", "Completed", "Canceled", "Expired", "Margin",
                 "Rejected"]
        return names[self.status if status is None else status]


class Position:
    def __init__(self):
        self.size = 0.0
        self.price = 0.0

    def __bool__(self):
        return self.size != 0

    def __len__(self):
        return int(abs(self.size) > 0)

    def __repr__(self):
        return f"Position(size={self.size})"


class _Params:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(f"strategy parameter '{name}' is not defined in params")


def _collect_params(cls, overrides):
    d = {}
    for c in reversed(cls.__mro__):
        p = c.__dict__.get("params")
        if p is None:
            continue
        items = p.items() if isinstance(p, dict) else p
        for kv in items:
            k, v = kv[0], kv[1]
            d[k] = v
    for k, v in (overrides or {}).items():
        if k not in d:
            raise TypeError(f"unknown strategy parameter '{k}'")
        d[k] = v
    return d


class _Broker:
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        _unsupported(f"self.broker.{name}()", "the broker (cash, value, commission) is not modelled; the auditor "
                     "converts your position targets to weights and applies its own costs")


_STRAT_UNSUPPORTED = {"sizer", "setsizer", "getsizer", "getsizing", "analyzers", "observers", "stats",
                      "buy_bracket", "sell_bracket", "order_target_value", "add_timer", "notify_timer", "env",
                      "cerebro", "getdatabyname", "getpositionbyname", "getpositions"}


class Strategy:
    """Minimal backtrader Strategy. Replayed once per ticker (one data feed each)."""
    params = ()

    def __init__(self):
        pass

    def prenext(self):
        pass

    def nextstart(self):
        self.next()

    def next(self):
        pass

    def notify_order(self, order):
        pass

    def notify_trade(self, trade):
        pass

    def log(self, txt, dt=None):
        pass

    # ---- orders: each changes this ticker's target exposure; it executes at the next bar like backtrader
    def _order_data(self, data):
        if data is not None and data is not self.data:
            _unsupported("orders on another data feed", "the adapter replays one ticker (self.data) at a time")

    def _check_order_kw(self, exectype, kw):
        if exectype not in (None, Order.Market):
            _unsupported("exectype other than Market (Limit/Stop/StopTrail orders)")
        bad = [k for k, v in kw.items() if v is not None and k not in ("valid", "tradeid")]
        if bad:
            _unsupported(f"order arguments {', '.join(bad)}")

    def _submit(self, new_size, isbuy, delta):
        self._bta_target = float(new_size)
        o = Order(self.data, abs(delta), isbuy)
        self._bta_pending.append(o)
        return o

    def buy(self, data=None, size=None, price=None, plimit=None, exectype=None, **kw):
        self._order_data(data)
        self._check_order_kw(exectype, dict(kw, price=price, plimit=plimit))
        s = 1.0 if size is None else float(size)
        return self._submit(self._bta_target + s, True, s)

    def sell(self, data=None, size=None, price=None, plimit=None, exectype=None, **kw):
        self._order_data(data)
        self._check_order_kw(exectype, dict(kw, price=price, plimit=plimit))
        s = 1.0 if size is None else float(size)
        return self._submit(self._bta_target - s, False, s)

    def close(self, data=None, size=None, **kw):
        self._order_data(data)
        self._check_order_kw(None, kw)
        if size is not None:
            _unsupported("close(size=...)")
        if self._bta_target == 0:
            return None
        return self._submit(0.0, self._bta_target < 0, self._bta_target)

    def order_target_size(self, data=None, target=0, **kw):
        self._order_data(data)
        self._check_order_kw(None, kw)
        return self._submit(float(target), float(target) > self._bta_target, float(target) - self._bta_target)

    def order_target_percent(self, data=None, target=0.0, **kw):
        self._order_data(data)
        self._check_order_kw(None, kw)
        return self._submit(float(target), float(target) > self._bta_target, float(target) - self._bta_target)

    def cancel(self, order):
        _unsupported("self.cancel(order)", "orders fill at the next bar's open; there is nothing pending to cancel")

    def getposition(self, data=None, broker=None):
        self._order_data(data)
        return self._bta_position

    @property
    def position(self):
        return self._bta_position

    @property
    def broker(self):
        return _Broker()

    def __len__(self):
        return self._bta_cur.i + 1

    def __getattr__(self, name):
        if name in _STRAT_UNSUPPORTED:
            _unsupported(f"self.{name}", "the adapter models positions only (no sizers, analyzers, brackets, timers)")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class _Cerebro:
    """Accepted so scripts that build a Cerebro at module level still load; nothing here affects the audit
    except ``addstrategy(Strategy, **params)``, whose params are used for that Strategy."""

    def __init__(self, *a, **k):
        self.broker = IGNORED

    def addstrategy(self, cls, *a, **k):
        if a:
            _unsupported("positional strategy arguments in addstrategy", "pass params as keywords")
        _STATE["addstrategy"][cls] = dict(k)
        return 0

    def optstrategy(self, *a, **k):
        _unsupported("cerebro.optstrategy (parameter optimisation)", "audit one parameter set")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return IGNORED


class _Feeds(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        _STATE["notes"].add("backtrader: data feeds are replaced by the audit panel")
        return lambda *a, **k: IGNORED


MAX_BT_BARS = 600_000   # ticker-bars per dataset; beyond this the replay is refused (keeps the demo responsive)


def run_backtrader(cls, data):
    """Replay ``cls`` once per ticker over that ticker's own trading days; return dates x tickers weights.

    Exposure per ticker: ``buy()``/``sell()`` add/subtract ``size`` units (default 1), ``close()`` goes flat,
    ``order_target_percent/size(target)`` set the target. Weights = exposure / max(1, sum |exposure|) per date."""
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise TypeError("run_backtrader needs a bt.Strategy subclass")
    close = data["close"]
    c_all = close.to_numpy(dtype=float)
    T, N = c_all.shape
    n_bars = int(np.isfinite(c_all).sum())
    if n_bars > MAX_BT_BARS:
        raise RuntimeError(f"backtrader replay refused: {n_bars:,} ticker-bars per dataset exceeds the "
                           f"{MAX_BT_BARS:,} cap (pure-Python bar loop); audit fewer tickers or a shorter window")
    fields = {f: data[f].to_numpy(dtype=float) if f in data else c_all for f in ("open", "high", "low", "volume")}
    params = _collect_params(cls, _STATE["addstrategy"].get(cls))
    E = np.zeros((T, N))
    has_prenext = cls.prenext is not Strategy.prenext
    user_init = cls.__init__
    for j in range(N):
        idx = np.nonzero(np.isfinite(c_all[:, j]))[0]
        if len(idx) == 0:
            continue
        cur = _Cursor()
        feed = DataFeed(str(close.columns[j]), close.index[idx], fields["open"][idx, j], fields["high"][idx, j],
                        fields["low"][idx, j], c_all[idx, j], fields["volume"][idx, j], cur)
        st = cls.__new__(cls)
        st.__dict__.update({"datas": [feed], "data": feed, "data0": feed, "dnames": {feed.bta_name: feed},
                            "params": _Params(params), "p": None, "_bta_cur": cur, "_bta_target": 0.0,
                            "_bta_pending": [], "_bta_position": Position(), "datetime": feed.datetime})
        st.__dict__["p"] = st.__dict__["params"]
        _BT["created"], _BT["stage"] = [], 1
        try:
            user_init(st)
        finally:
            created, _BT["created"] = _BT["created"], None
        _BT["stage"] = 2
        minp = max([1] + [ln.minperiod for ln in created])
        n = len(idx)
        out = np.zeros(n)
        pos = st._bta_position
        opens = feed.open.lst
        nxt, pre, notify = st.next, st.prenext, st.notify_order
        try:
            for t in range(n):
                cur.i = t
                pend = st._bta_pending
                if pend:                          # orders from the previous bar fill at this bar's open
                    st._bta_pending = []
                    for o in pend:
                        o.status = Order.Completed
                        o.executed = _Executed(opens[t], o.size if o.bta_isbuy else -o.size)
                    pos.size, pos.price = st._bta_target, opens[t]
                    for o in pend:
                        notify(o)
                if t + 1 >= minp:
                    nxt()
                elif has_prenext:
                    pre()
                out[t] = st._bta_target
        finally:
            _BT["stage"] = 1
        E[idx, j] = out
    gross = np.abs(E).sum(axis=1, keepdims=True)
    W = E / np.maximum(gross, 1.0)
    return pd.DataFrame(W, index=close.index, columns=close.columns)


def vbt_module_result(entries, exits, short_entries=None, short_exits=None, direction=None):
    return {"entries": entries, "exits": exits, "short_entries": short_entries, "short_exits": short_exits,
            "direction": direction}


def notes():
    return sorted(_STATE["notes"])


# =============================================================================================== modules
def build_modules():
    """-> {import name: module} for the sandbox import hook."""
    rt = types.ModuleType("bta_adapter")
    for name in ("begin", "to_weights", "entries_exits_to_weights", "signals_to_weights", "run_backtrader",
                 "vbt_module_result", "notes"):
        setattr(rt, name, globals()[name])

    vbt = types.ModuleType("vectorbt")
    vbt.Portfolio = _Portfolio
    vbt.MA, vbt.RSI, vbt.YFData = _MA, _RSI, _YFData
    _register_accessor()

    bt = types.ModuleType("backtrader")
    ind = _IndicatorNamespace("backtrader.indicators")
    for k, v in _INDICATORS.items():
        setattr(ind, k, v)
    feeds = _Feeds("backtrader.feeds")
    bt.Strategy, bt.Indicator, bt.Order, bt.Position = Strategy, Indicator, Order, Position
    bt.Cerebro = _Cerebro
    bt.indicators = bt.ind = bt.indicator = ind
    bt.feeds = feeds
    bt.TimeFrame = types.SimpleNamespace(Days=5, Weeks=6, Months=7, Years=8, Minutes=4)
    return {"bta_adapter": rt, "vectorbt": vbt, "backtrader": bt,
            "backtrader.indicators": ind, "backtrader.feeds": feeds}

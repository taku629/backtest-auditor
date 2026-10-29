"""Sandbox child for the dynamic leak localizer. Runs untrusted (instrumented) strategy code.

Same hardening as ``_child.py`` (restricted builtins, import allow-list, sockets disabled, CPU/file/fork limits;
the parent adds ``python -I``, an empty environment, a temp cwd and a wall-clock timeout). Do not import project
modules here.

Protocol: the parent writes a pickle (trusted direction)::

    {code, func_name, base, perts: [dataset...], cut_dates: [iso...], cut_positions: [int...],
     cpu_seconds, mem_mb, budget_mb}

``code`` is the strategy after AST instrumentation: after each assignment it calls ``_bta_rec(sid, value)`` and
every ``return x`` became ``return _bta_ret(sid, x)``. The recorder lives here, in the trusted child, and only reads
the values. The base dataset runs first and keeps, for every recorded (statement, occurrence) and every cut, the
part of the value that a point-in-time computation could know at the cut (rows with a date label <= cut, or the
whole object when it is not indexed by date). Each perturbed dataset then runs and every record is compared with
the stored base value. The child writes ONE JSON document to stdout (no pickle on the way back).
"""
import builtins
import hashlib
import io
import json
import math
import pickle
import resource
import signal
import sys
import traceback

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "datetime"}
STUB_FILE = "adapter_stubs.py"   # trusted vectorbt/backtrader stand-ins (audit/adapters.py), same directory
REMOVED_BUILTINS = {"open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals", "vars",
                    "getattr", "setattr", "delattr", "help", "exit", "quit", "memoryview", "__loader__",
                    "__spec__", "copyright", "credits", "license"}
MAX_FULL_OCC = 100        # record every occurrence of a statement up to this many (loops) ...
OCC_STRIDE = 25           # ... then every OCC_STRIDE-th one
MAX_RECORDS = 20000       # per run
MAX_LIST = 100000


def _emit(obj):
    out = sys.__stdout__.buffer
    out.write(json.dumps(obj, default=str).encode())
    out.flush()


def _load_stubs():
    """Load the trusted adapter stubs by path (python -I does not put this directory on sys.path)."""
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STUB_FILE)
    spec = importlib.util.spec_from_file_location("bta_adapter_stubs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():  # noqa: C901 - one self-contained process
    for d in sys.argv[1:]:
        if d not in sys.path:
            sys.path.append(d)
    payload = pickle.load(sys.stdin.buffer)
    import numpy as np
    import pandas as pd

    cpu = int(payload.get("cpu_seconds", 60))
    mem = int(payload.get("mem_mb", 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem * 2 ** 20, mem * 2 ** 20))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError):
        pass

    import socket

    def _no_net(*a, **k):
        raise PermissionError("network access is disabled in the sandbox")

    socket.socket = _no_net  # type: ignore[assignment]
    socket.create_connection = _no_net  # type: ignore[assignment]
    socket.getaddrinfo = _no_net  # type: ignore[assignment]

    real_import = builtins.__import__
    stubs = _load_stubs()
    stub_mods = stubs.build_modules()

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in stub_mods:   # trusted stand-ins: vectorbt, backtrader, bta_adapter
            return stub_mods[name] if fromlist else stub_mods[name.split(".")[0]]
        if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"import of '{name}' is not allowed (allowed: numpy, pandas, math, datetime; "
                              "vectorbt/backtrader resolve to the auditor's stubs)")
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in REMOVED_BUILTINS}
    safe_builtins["__import__"] = safe_import

    base = payload["base"]
    T = len(base["close"].index)
    cut_dates = [pd.Timestamp(c) for c in payload["cut_dates"]]
    cut_pos = [int(p) for p in payload["cut_positions"]]
    budget = int(payload.get("budget_mb", 256)) * 2 ** 20

    # ------------------------------------------------------------------ canonical "known at the cut" view
    def _time_mask_index(idx, cut):
        if isinstance(idx, pd.DatetimeIndex):
            return np.asarray(idx <= cut)
        if isinstance(idx, pd.PeriodIndex):
            return np.asarray(idx.start_time <= cut)
        if isinstance(idx, pd.MultiIndex):
            for lv in range(idx.nlevels):
                lvl = idx.levels[lv]
                if isinstance(lvl, (pd.DatetimeIndex, pd.PeriodIndex)):
                    vals = idx.get_level_values(lv)
                    st = vals.start_time if isinstance(vals, pd.PeriodIndex) else vals
                    return np.asarray(st <= cut)
        return None

    def _time_label(lbl):
        if isinstance(lbl, tuple):
            for x in lbl:
                if isinstance(x, (pd.Timestamp, pd.Period)):
                    return x
            return lbl[0] if lbl else None
        return lbl

    def view(v, j):
        """-> ("time"|"whole", obj) or None when the value is not tracked."""
        cut = cut_dates[j]
        if isinstance(v, (pd.DataFrame, pd.Series)):
            m = _time_mask_index(v.index, cut)
            if m is not None:
                s = v.loc[m]
                try:
                    s = s.sort_index(kind="mergesort")
                except TypeError:
                    pass
                return "time", s
            if isinstance(v, pd.DataFrame):
                m = _time_mask_index(v.columns, cut)
                if m is not None:
                    s = v.loc[:, m].T
                    try:
                        s = s.sort_index(kind="mergesort")
                    except TypeError:
                        pass
                    return "time", s
            return "whole", v.copy()
        if isinstance(v, pd.Index):
            m = _time_mask_index(v, cut)
            if m is not None:
                return "time", pd.Series(0.0, index=v[m]).sort_index(kind="mergesort")
            return "whole", v.copy()
        if isinstance(v, np.ndarray):
            if v.ndim >= 1 and v.shape[0] == T:
                return "time", np.array(v[: cut_pos[j] + 1], copy=True)
            if v.ndim == 2 and v.shape[1] == T:
                return "time", np.array(v[:, : cut_pos[j] + 1].T, copy=True)
            return "whole", np.array(v, copy=True)
        if isinstance(v, (bool, int, float, complex, str, np.generic)) or v is None:
            return "whole", v
        if isinstance(v, (list, tuple, set, frozenset)) and len(v) <= MAX_LIST:
            if all(isinstance(x, (bool, int, float, str, np.generic, pd.Timestamp)) or x is None for x in v):
                return "whole", sorted(v, key=repr) if isinstance(v, (set, frozenset)) else list(v)
        return None

    def nbytes(obj):
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return int(obj.memory_usage(deep=False, index=True).sum()) if isinstance(obj, pd.DataFrame) \
                    else int(obj.memory_usage(deep=False, index=True))
            if isinstance(obj, (np.ndarray, pd.Index)):
                return int(obj.nbytes)
        except Exception:  # noqa: BLE001
            pass
        return 64

    def digest(obj):
        h = hashlib.blake2b(digest_size=16)
        h.update(type(obj).__name__.encode())
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            h.update(repr(list(obj.index)).encode())
            if isinstance(obj, pd.DataFrame):
                h.update(repr(list(obj.columns)).encode())
            arr = obj.to_numpy()
        else:
            arr = obj
        if isinstance(arr, np.ndarray) and arr.dtype.kind in "fiub":
            a = arr.astype(np.float64)
            nan = np.isnan(a)
            h.update(nan.tobytes())
            a = np.where(nan, 0.0, a)
            # 9 significant digits: insensitive to last-bit noise, sensitive to any real change
            with np.errstate(all="ignore"):
                mag = np.where(a == 0, 0.0, np.floor(np.log10(np.abs(np.where(np.isfinite(a) & (a != 0), a, 1.0)))))
                q = np.where(np.isfinite(a), np.round(a / 10.0 ** (mag - 8)), a)
            h.update(np.ascontiguousarray(q).tobytes())
            h.update(repr(arr.shape).encode())
        else:
            h.update(repr(arr).encode()[:2_000_000])
        return ("__digest__", h.hexdigest())

    # ------------------------------------------------------------------ comparison
    def _close(x, y):
        """Elementwise 'equal' with the audit's tolerance and NaN == NaN."""
        x = np.asarray(x)
        y = np.asarray(y)
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            x = x.astype(np.float64)
            y = y.astype(np.float64)
            with np.errstate(all="ignore"):
                eq = (x == y) | (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= (1e-9 + 1e-7 * np.abs(x)))
                dd = np.abs(x - y)
                fin = ~eq & np.isfinite(dd)  # NaN-vs-number changes count as changed but carry no magnitude
            return eq, (float(dd[fin].max()) if fin.any() else None)
        xs = pd.isna(x)
        ys = pd.isna(y)
        try:
            eq = (x == y) | (xs & ys)
            eq = np.asarray(eq, dtype=bool)
        except Exception:  # noqa: BLE001
            eq = np.full(x.shape, bool(np.array_equal(x, y)))
        return eq, None

    def _label_str(lbl):
        lbl = _time_label(lbl)
        if isinstance(lbl, pd.Period):
            lbl = lbl.start_time
        if isinstance(lbl, pd.Timestamp):
            return lbl.strftime("%Y-%m-%d")
        return None if lbl is None else str(lbl)

    def compare(kind, a, b):
        """-> dict(d=diverged, f=first changed label, r=rows changed, m=max abs change, why=...)"""
        if isinstance(a, tuple) and a and a[0] == "__digest__":
            return {"d": a != b, "f": None, "r": None, "m": None, "why": "digest"}
        if type(a) is not type(b):
            return {"d": True, "f": None, "r": None, "m": None, "why": "type changed"}
        if isinstance(a, (pd.DataFrame, pd.Series)):
            if not a.index.equals(b.index):
                sa, sb = set(a.index.tolist()), set(b.index.tolist())
                diff = list(sa ^ sb)
                first = None
                if kind == "time" and diff:
                    try:
                        first = _label_str(min(diff, key=lambda z: _time_label(z)))
                    except TypeError:
                        first = None
                return {"d": True, "f": first, "r": len(diff), "m": None, "why": "index labels changed"}
            if isinstance(a, pd.DataFrame) and not a.columns.equals(b.columns):
                return {"d": True, "f": _label_str(a.index[0]) if (kind == "time" and len(a)) else None,
                        "r": int(len(a)), "m": None, "why": "columns changed"}
            eq, mx = _close(a.to_numpy(), b.to_numpy())
            rows = ~eq.reshape(len(a), -1).all(axis=1) if eq.size else np.zeros(len(a), bool)
            if not rows.any():
                return {"d": False}
            pos = np.where(rows)[0]
            return {"d": True, "f": _label_str(a.index[pos[0]]) if kind == "time" else None,
                    "r": int(len(pos)), "m": mx, "why": "values changed"}
        if isinstance(a, pd.Index):
            same = a.equals(b)
            return {"d": not same, "f": None, "r": None, "m": None, "why": "index changed"}
        if isinstance(a, np.ndarray):
            if a.shape != b.shape:
                return {"d": True, "f": None, "r": None, "m": None, "why": f"shape {a.shape} -> {b.shape}"}
            eq, mx = _close(a, b)
            if eq.all():
                return {"d": False}
            if kind == "time":
                rows = ~eq.reshape(a.shape[0], -1).all(axis=1)
                pos = np.where(rows)[0]
                first = base["close"].index[int(pos[0])].strftime("%Y-%m-%d")
                return {"d": True, "f": first, "r": int(len(pos)), "m": mx, "why": "values changed"}
            return {"d": True, "f": None, "r": None, "m": mx, "why": "values changed"}
        if isinstance(a, list):
            if len(a) != len(b):
                return {"d": True, "f": None, "r": None, "m": None, "why": f"length {len(a)} -> {len(b)}"}
            eq, mx = _close(np.array(a, dtype=object), np.array(b, dtype=object))
            return {"d": not bool(np.all(eq)), "f": None, "r": None, "m": mx, "why": "values changed"}
        try:
            if isinstance(a, (float, int, np.number)) and not isinstance(a, bool):
                eq, mx = _close(np.array([a]), np.array([b]))
                return {"d": not bool(eq.all()), "f": None, "r": None, "m": mx, "why": "value changed"}
            return {"d": bool(a != b), "f": None, "r": None, "m": None, "why": "value changed"}
        except Exception:  # noqa: BLE001
            return {"d": True, "f": None, "r": None, "m": None, "why": "incomparable"}

    # ------------------------------------------------------------------ recorder
    state = {"mode": "base", "job": -1, "occ": {}, "n": 0, "bytes": 0, "errors": 0, "records": []}
    store: dict = {}

    def _rec(sid, value):
        try:
            occ = state["occ"].get(sid, 0)
            state["occ"][sid] = occ + 1
            if occ >= MAX_FULL_OCC and occ % OCC_STRIDE:
                return
            if state["n"] >= MAX_RECORDS:
                return
            state["n"] += 1
            key = (sid, occ)
            if state["mode"] == "base":
                views = []
                for j in range(len(cut_dates)):
                    vw = view(value, j)
                    if vw is None:
                        return
                    k, obj = vw
                    nb = nbytes(obj)
                    if state["bytes"] + nb > budget:
                        obj = digest(obj)
                        nb = 64
                    state["bytes"] += nb
                    views.append((k, obj))
                store[key] = views
                state["records"].append([sid, occ])
            else:
                j = state["job"]
                if key not in store:
                    vw = view(value, j)
                    if vw is not None:
                        state["records"].append({"s": sid, "o": occ, "d": True, "why": "statement not reached "
                                                 "(or not at this occurrence) on the original data"})
                    return
                k, ref = store[key][j]
                vw = view(value, j)
                if vw is None:
                    state["records"].append({"s": sid, "o": occ, "d": True, "why": "type changed"})
                    return
                obj = vw[1]
                if isinstance(ref, tuple) and ref and ref[0] == "__digest__":
                    obj = digest(obj)
                res = compare(k, ref, obj)
                res.update({"s": sid, "o": occ, "k": k})
                if not res["d"]:
                    res = {"s": sid, "o": occ, "d": False, "k": k}
                if res.get("m") is not None and not math.isfinite(res["m"]):
                    res["m"] = None
                state["records"].append(res)
        except MemoryError:
            raise
        except Exception:  # noqa: BLE001 - never let the recorder break the strategy
            state["errors"] += 1

    def _ret(sid, value):
        _rec(sid, value)
        return value

    captured = io.StringIO()
    result = {"ok": False}
    try:
        g = {"__builtins__": safe_builtins, "__name__": "strategy", "np": np, "pd": pd, "math": math,
             "_bta_rec": _rec, "_bta_ret": _ret}
        sys.stdout = captured
        exec(compile(payload["code"], "<strategy>", "exec"), g)
        fn = g.get(payload["func_name"])
        if not callable(fn):
            raise NameError(f"function '{payload['func_name']}' is not defined")
        fn({k: v.copy() for k, v in base.items()})
        base_order = state["records"]
        jobs = []
        for j, d in enumerate(payload["perts"]):
            state.update({"mode": "pert", "job": j, "occ": {}, "n": 0, "records": []})
            fn({k: v.copy() for k, v in d.items()})
            jobs.append({"cut_pos": cut_pos[j], "cut_date": cut_dates[j].strftime("%Y-%m-%d"),
                         "records": state["records"]})
        result = {"ok": True, "base_order": base_order, "jobs": jobs, "recorder_errors": state["errors"],
                  "stored_mb": round(state["bytes"] / 2 ** 20, 2)}
    except BaseException as e:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                  "traceback": "".join(traceback.format_exception_only(type(e), e))}
    finally:
        sys.stdout = sys.__stdout__
    result["stdout"] = captured.getvalue()[-2000:]
    _emit(result)


if __name__ == "__main__":
    main()

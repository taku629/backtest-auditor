"""Sandbox child process. Runs untrusted strategy code. Do not import project modules here.

Protocol: parent writes a pickle {code, datasets, func_name, cpu_seconds, mem_mb}
to stdin (trusted direction). Child writes ONE JSON document to stdout; arrays
are base64 float64 (no pickle on the way back, so a hostile strategy cannot
smuggle objects into the parent).
"""
import base64
import builtins
import io
import json
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


def _emit(obj):
    out = sys.__stdout__.buffer
    out.write(json.dumps(obj).encode())
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


def main():
    for d in sys.argv[1:]:
        if d not in sys.path:
            sys.path.append(d)
    payload = pickle.load(sys.stdin.buffer)
    import math
    import numpy as np
    import pandas as pd

    cpu = int(payload.get("cpu_seconds", 30))
    mem = int(payload.get("mem_mb", 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))       # no file can grow: writes fail
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem * 2 ** 20, mem * 2 ** 20))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))   # no forking
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

    captured = io.StringIO()
    result = {"ok": False}
    try:
        g = {"__builtins__": safe_builtins, "__name__": "strategy", "np": np, "pd": pd, "math": math}
        sys.stdout = captured
        exec(compile(payload["code"], "<strategy>", "exec"), g)
        fn = g.get(payload["func_name"])
        if not callable(fn):
            raise NameError(f"function '{payload['func_name']}' is not defined")
        outputs, notes = [], []
        names = payload.get("func_names")
        for k, d in enumerate(payload["datasets"]):
            ref = d["close"]
            if names:
                fn = g.get(names[k])
                if not callable(fn):
                    raise NameError(f"function '{names[k]}' is not defined")
            w = fn({k: v.copy() for k, v in d.items()})
            if isinstance(w, pd.Series):
                raise TypeError("generate_signals must return a DataFrame (dates x tickers), got a Series")
            if not isinstance(w, pd.DataFrame):
                w = pd.DataFrame(np.asarray(w, dtype=float), index=ref.index, columns=ref.columns)
            w.index = pd.to_datetime(w.index)
            if not w.index.equals(ref.index):
                notes.append("output index differed from data index; forward-filled onto the data grid")
                w = w.sort_index().reindex(ref.index, method="ffill")
            w = w.reindex(columns=ref.columns).apply(pd.to_numeric, errors="coerce")
            arr = np.ascontiguousarray(w.to_numpy(dtype=np.float64))
            outputs.append({"shape": list(arr.shape), "b64": base64.b64encode(arr.tobytes()).decode()})
        result = {"ok": True, "outputs": outputs, "notes": sorted(set(notes) | set(stubs.notes()))}
    except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit/MemoryError
        tb = traceback.extract_tb(e.__traceback__)
        frames = [f for f in tb if f.filename == "<strategy>"]
        src = payload.get("code", "").splitlines()
        where = ""
        if frames:
            ln = frames[-1].lineno or 0
            text = src[ln - 1].strip() if 0 < ln <= len(src) else ""
            where = f" (strategy line {ln}: {text})"
        result = {"ok": False, "error": f"{type(e).__name__}: {e}{where}",
                  "traceback": "".join(traceback.format_exception_only(type(e), e))}
    finally:
        sys.stdout = sys.__stdout__
    result["stdout"] = captured.getvalue()[-4000:]
    _emit(result)


if __name__ == "__main__":
    main()

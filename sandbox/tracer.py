"""Run instrumented strategy code in the sandbox and return the value trace as JSON.

This is a second entrypoint into the same sandbox (``_trace_child.py``), used by the dynamic leak localizer
(``audit/localize.py``). It applies the same layers as ``runner.run_strategy``: the static policy is checked on the
user's ORIGINAL code (and again on the instrumented code), then the code runs in ``python -I`` with an empty
environment, a temp working dir, restricted builtins, an import allow-list, no sockets, CPU / file-size / fork
limits and a hard wall-clock timeout. Results come back as JSON only.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from sandbox.runner import _site_dirs, check_policy

TRACE_CHILD = Path(__file__).with_name("_trace_child.py")
RESERVED_PREFIX = "_bta_"


class TraceError(RuntimeError):
    pass


def run_trace(original_code: str, instrumented_code: str, base: dict, perts: list[dict], cut_positions: list[int],
              *, timeout: float = 120.0, func_name: str = "generate_signals", mem_mb: int = 0,
              budget_mb: int = 256) -> dict:
    """Execute ``instrumented_code`` on ``base`` then on each perturbed copy; return the child's JSON."""
    violations = check_policy(original_code)
    if RESERVED_PREFIX in original_code:
        violations.append(f"names starting with '{RESERVED_PREFIX}' are reserved for the localizer")
    if violations:
        raise TraceError("Sandbox policy violation:\n" + "\n".join(violations))
    inst_v = check_policy(instrumented_code)
    if inst_v:  # instrumentation must never need a looser policy
        raise TraceError("instrumented code failed the sandbox policy: " + "; ".join(inst_v))
    idx = base["close"].index
    payload = pickle.dumps({"code": instrumented_code, "func_name": func_name, "base": base, "perts": perts,
                            "cut_positions": [int(c) for c in cut_positions],
                            "cut_dates": [pd.Timestamp(idx[c]).isoformat() for c in cut_positions],
                            "cpu_seconds": int(timeout) + 1, "mem_mb": mem_mb, "budget_mb": budget_mb},
                           protocol=pickle.HIGHEST_PROTOCOL)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="bta_trace_") as tmp:
        env["HOME"] = tmp
        try:
            proc = subprocess.run([sys.executable, "-I", str(TRACE_CHILD), *_site_dirs()], input=payload,
                                  capture_output=True, timeout=timeout, env=env, cwd=tmp)
        except subprocess.TimeoutExpired:
            raise TraceError(f"Timeout: traced strategy exceeded {timeout:.0f}s wall clock") from None
    try:
        res = json.loads(proc.stdout.decode() or "{}")
    except json.JSONDecodeError:
        res = {}
    if not res:
        err = proc.stderr.decode(errors="replace")[-800:]
        raise TraceError(f"trace sandbox crashed (exit code {proc.returncode}). {err}".strip())
    if not res.get("ok"):
        raise TraceError(res.get("error", "unknown error"))
    res["duration_s"] = round(time.time() - t0, 3)
    return res

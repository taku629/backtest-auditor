"""Run LLM-generated strategy code in an isolated subprocess.

Layers of defence (defence in depth, not a security boundary for hostile
multi-tenant use; for that, add a container/gVisor/Firecracker layer):

1. Static policy check (AST): only ``numpy``, ``pandas``, ``math`` imports; no
   dunder or private attribute access; no file/network/process APIs by name.
2. Child process: ``python -I`` with an empty environment (API keys never reach
   it), temp working dir, restricted builtins (no open/exec/eval/getattr...),
   import allow-list, sockets disabled.
3. OS limits: CPU seconds, file-size limit 0 (no file writes), no fork,
   optional address-space cap, and a wall-clock timeout.
4. Results come back as JSON + raw float64 arrays, never pickle.
"""
from __future__ import annotations

import ast
import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CHILD = Path(__file__).with_name("_child.py")
ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                 "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview"}
BLOCKED_ATTRS = {
    # filesystem / serialization
    "to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_hdf", "to_sql", "to_feather",
    "to_stata", "to_clipboard", "to_html", "to_latex", "to_markdown", "to_xml", "to_orc",
    "read_csv", "read_pickle", "read_parquet", "read_excel", "read_json", "read_hdf", "read_sql",
    "read_html", "read_clipboard", "read_feather", "read_fwf", "read_table", "read_xml", "read_orc",
    "read_sas", "read_spss", "read_stata", "save", "savez", "savez_compressed", "savetxt", "load",
    "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "DataSource",
    # escape hatches
    "os", "sys", "subprocess", "builtins", "io", "importlib", "socket", "shutil", "pathlib",
    "ctypes", "ctypeslib", "system", "popen", "compat", "util", "testing", "f2py", "distutils",
    "eval", "query",  # DataFrame.eval/query evaluate strings
}


def _site_dirs() -> list[str]:
    """Where numpy/pandas live (python -I drops user site-packages, so pass them explicitly)."""
    return sorted({str(Path(np.__file__).resolve().parent.parent), str(Path(pd.__file__).resolve().parent.parent)})


class PolicyViolation(Exception):
    pass


def check_policy(code: str) -> list[str]:
    """Return a list of policy violations (empty list = allowed)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError line {e.lineno}: {e.msg}"]
    v: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    v.append(f"line {node.lineno}: import '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                v.append(f"line {node.lineno}: from-import '{node.module}' not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                v.append(f"line {node.lineno}: private/dunder attribute '.{node.attr}' not allowed")
            elif node.attr in BLOCKED_ATTRS:
                v.append(f"line {node.lineno}: attribute '.{node.attr}' not allowed (I/O or escape hatch)")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or (node.id.startswith("__") and node.id != "__name__"):
                v.append(f"line {node.lineno}: name '{node.id}' not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            v.append(f"line {node.lineno}: global/nonlocal not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "generate_signals" for n in tree.body):
        v.append("missing top-level function 'generate_signals(data)'")
    return v


@dataclass
class SandboxResult:
    ok: bool
    outputs: list[pd.DataFrame] = field(default_factory=list)
    error: str = ""
    stdout: str = ""
    notes: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    policy_violations: list[str] = field(default_factory=list)


def run_strategy(code: str, datasets: list[dict], *, timeout: float = 60.0, func_name: str = "generate_signals",
                 mem_mb: int = 0, enforce_policy: bool = True) -> SandboxResult:
    """enforce_policy=False skips only the static layer (used by tests to prove the runtime layer holds)."""
    t0 = time.time()
    violations = check_policy(code) if enforce_policy else []
    if violations:
        return SandboxResult(False, error="Sandbox policy violation:\n" + "\n".join(violations),
                             policy_violations=violations, duration_s=time.time() - t0)
    payload = pickle.dumps({"code": code, "datasets": datasets, "func_name": func_name,
                            "cpu_seconds": int(timeout) + 1, "mem_mb": mem_mb},
                           protocol=pickle.HIGHEST_PROTOCOL)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
    with tempfile.TemporaryDirectory(prefix="bta_sbx_") as tmp:
        env["HOME"] = tmp
        try:
            proc = subprocess.run([sys.executable, "-I", str(CHILD), *_site_dirs()], input=payload, capture_output=True,
                                  timeout=timeout, env=env, cwd=tmp)
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error=f"Timeout: strategy exceeded {timeout:.0f}s wall clock",
                                 duration_s=time.time() - t0)
    dt = time.time() - t0
    try:
        res = json.loads(proc.stdout.decode() or "{}")
    except json.JSONDecodeError:
        res = {}
    if not res:
        err = proc.stderr.decode(errors="replace")[-1500:]
        code_desc = f"exit code {proc.returncode}"
        if proc.returncode in (-9, -24, 137, 152):
            code_desc += " (killed: CPU/memory limit)"
        return SandboxResult(False, error=f"Sandbox crashed ({code_desc}). {err}".strip(), duration_s=dt)
    if not res.get("ok"):
        return SandboxResult(False, error=res.get("error", "unknown error"), stdout=res.get("stdout", ""),
                             duration_s=dt)
    outs = []
    for d, o in zip(datasets, res["outputs"]):
        arr = np.frombuffer(base64.b64decode(o["b64"]), dtype=np.float64).reshape(o["shape"])
        ref = d["close"]
        if arr.shape != ref.shape:
            return SandboxResult(False, error=f"output shape {arr.shape} != data shape {ref.shape}", duration_s=dt)
        outs.append(pd.DataFrame(arr.copy(), index=ref.index, columns=ref.columns))
    return SandboxResult(True, outs, stdout=res.get("stdout", ""), notes=res.get("notes", []), duration_s=dt)


class SandboxExecutor:
    """Adapter so audit.run_audit can run code through the sandbox."""

    def __init__(self, code: str, timeout: float = 90.0):
        self.code = code
        self.timeout = timeout
        self.last: SandboxResult | None = None

    def __call__(self, datasets: list[dict]) -> list[pd.DataFrame]:
        self.last = run_strategy(self.code, datasets, timeout=self.timeout)
        if not self.last.ok:
            raise SandboxError(self.last.error)
        return self.last.outputs


class SandboxError(RuntimeError):
    pass

"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of Backtest Overfitting",
Journal of Computational Finance 20(4).

Idea: a researcher who tunes parameters picks the variant with the best in-sample Sharpe. CSCV splits the
T x N return matrix (T dates, N parameter variants) into S row blocks and, for every way of choosing S/2 blocks
as in-sample (IS) with the rest out-of-sample (OOS), asks where the IS winner ranks OOS. With relative rank
w in (0, 1), the logit is l = ln(w / (1 - w)); PBO = P(l <= 0) = how often the IS winner is at or below the
OOS median. PBO near 0: the tuning picks something that keeps working. PBO >= 0.5: picking the IS best is no
better than picking at random (or worse).

This module also builds the parameter grid from the strategy source:

1. explicit parameters: module-level UPPERCASE numeric constants and numeric defaults of ``generate_signals``;
2. otherwise inline literals that act as parameters: lookback windows (``rolling(20)``, ``ewm(span=..)``,
   ``shift(20)``, ``pct_change(5)``, ``diff``, ``nlargest``/``nsmallest``) and thresholds in comparisons
   (``rank >= 0.9``, ``rank <= 5``, ``z > 2.0``). ``shift(1)`` (the lag) and ``min_periods`` are never varied;
   a literal ``min_periods`` is scaled with its window so it never exceeds it.

Each value is varied multiplicatively 0.5x .. 2x (for probabilities above 0.5, e.g. ``rank >= 0.9``, the
distance to 1 is scaled instead, so "top 10%" becomes "top 5% .. 20%"). At most 16 variants. All variants run
in ONE sandbox subprocess (``sandbox.runner.run_strategy`` with ``func_names``).
"""
from __future__ import annotations

import ast
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_VARIANTS = 16
MAX_EXPLICIT_PARAMS = 4
MAX_INLINE_PARAMS = 3
LEVELS_1D = (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0)
LEVELS_ND = (0.5, 0.71, 1.0, 1.41, 2.0)
PBO_FAIL = 0.5            # PBO above this fails the (non-blocking) test
PBO_MODERATE = 0.25       # between this and PBO_FAIL: "moderate" risk, still passes
WINDOW_CALLS = {"rolling", "ewm", "shift", "pct_change", "diff", "nlargest", "nsmallest", "expanding"}
WINDOW_KW = {"window", "span", "periods", "com", "halflife", "n"}


# ----------------------------------------------------------------------------
# CSCV core (pure numpy; unit-tested on synthetic matrices)
# ----------------------------------------------------------------------------
def _avg_rank_desc_is_best(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged; rank N = best (highest value)."""
    return pd.Series(x).rank(method="average").to_numpy()


def cscv(returns: np.ndarray, n_blocks: int = 10, max_combos: int = 400, seed: int = 0) -> dict:
    """Run CSCV on a T x N matrix of per-period returns. Returns PBO and the per-split statistics.

    Sharpe ratios are per period (mean / std). ``n_blocks`` must be even. If C(S, S/2) exceeds
    ``max_combos`` a seeded random subset of splits (complement pairs kept together) is used.
    """
    M = np.asarray(returns, dtype=float)
    M = np.where(np.isfinite(M), M, 0.0)
    T, N = M.shape
    S = int(n_blocks)
    if S % 2 or S < 2:
        raise ValueError("n_blocks must be even and >= 2")
    if N < 2 or T < S * 2:
        raise ValueError(f"need >= 2 variants and >= {2 * S} rows (got N={N}, T={T})")
    T_use = (T // S) * S
    M = M[-T_use:]                       # drop the oldest remainder rows so blocks are equal
    blocks = M.reshape(S, T_use // S, N)
    s1 = blocks.sum(axis=1)              # S x N
    s2 = (blocks ** 2).sum(axis=1)
    nb = T_use // S
    half = S // 2
    combos = list(itertools.combinations(range(S), half))
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        firsts = [c for c in combos if 0 in c]            # one of each complement pair
        pick = rng.choice(len(firsts), size=max_combos // 2, replace=False)
        chosen = [firsts[i] for i in sorted(pick)]
        combos = chosen + [tuple(sorted(set(range(S)) - set(c))) for c in chosen]

    def sharpe(idx):
        n = nb * len(idx)
        m = s1[list(idx)].sum(axis=0) / n
        var = (s2[list(idx)].sum(axis=0) - n * m ** 2) / max(n - 1, 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        return np.where(sd > 1e-12, m / np.where(sd > 1e-12, sd, 1.0), 0.0)

    logits, is_best, oos_best, oos_rank = [], [], [], []
    for c in combos:
        oos = tuple(sorted(set(range(S)) - set(c)))
        sr_is, sr_oos = sharpe(c), sharpe(oos)
        star = int(np.argmax(sr_is))
        r = _avg_rank_desc_is_best(sr_oos)[star]
        w = r / (N + 1)
        logits.append(math.log(w / (1 - w)))
        is_best.append(float(sr_is[star]))
        oos_best.append(float(sr_oos[star]))
        oos_rank.append(float(w))
    logits = np.array(logits)
    x, y = np.array(is_best), np.array(oos_best)
    slope, intercept = (np.polyfit(x, y, 1) if np.ptp(x) > 1e-12 else (0.0, float(y.mean())))
    r6 = lambda a: [round(float(v), 6) for v in a]  # noqa: E731 - keep traces small
    return {
        "pbo": float((logits <= 0).mean()),
        "logits": r6(logits),
        "is_sharpe_best": r6(x),          # per period
        "oos_sharpe_best": r6(y),
        "oos_relative_rank": r6(oos_rank),
        "degradation_slope": float(slope),
        "degradation_intercept": float(intercept),
        "prob_oos_loss": float((y < 0).mean()),
        "n_splits": len(combos),
        "n_blocks": S,
        "n_variants": N,
        "n_obs": T_use,
    }


# ----------------------------------------------------------------------------
# Parameter discovery and source rewriting
# ----------------------------------------------------------------------------
@dataclass
class Param:
    name: str
    value: float
    is_int: bool
    kind: str                 # explicit | window | threshold
    nodes: list = field(default_factory=list)       # ast nodes (Constant or UnaryOp) to rewrite
    dependents: list = field(default_factory=list)  # (min_periods node, its window node)

    def vary(self, f: float) -> float:
        v = self.value
        if self.is_int:
            return float(max(1, int(round(v * f))))
        if 0.5 < v < 1.0:                      # probability-like threshold: scale distance to 1
            return round(1.0 - (1.0 - v) * f, 6)
        if 0.0 < v <= 0.5:
            return round(min(v * f, 0.98), 6)
        return round(v * f, 6)


def _num(node) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool)):
        return -float(node.operand.value)
    return None


def _is_int(node) -> bool:
    n = node.operand if isinstance(node, ast.UnaryOp) else node
    return isinstance(n.value, int)


def find_params(code: str) -> list[Param]:
    tree = ast.parse(code)
    explicit: list[Param] = []
    gen = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_signals"), None)
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            v = _num(n.value)
            if v is not None and name.isupper() and v != 0:
                explicit.append(Param(name, v, _is_int(n.value), "explicit", [n.value]))
    if gen is not None:
        a = gen.args
        pos = a.posonlyargs + a.args
        for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
            v = _num(d)
            if v is not None and v != 0:
                explicit.append(Param(arg.arg, v, _is_int(d), "explicit", [d]))
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            v = _num(d) if d is not None else None
            if v is not None and v != 0:
                explicit.append(Param(arg.arg, v, _is_int(d), "explicit", [d]))
    if explicit:
        return explicit[:MAX_EXPLICIT_PARAMS]

    # Inline literals inside every function (generate_signals and its helpers)
    windows: dict[float, Param] = {}
    thresholds: dict[float, Param] = {}
    order: list[tuple[str, float]] = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in WINDOW_CALLS:
                win = None
                if node.args and node.func.attr != "expanding":
                    win = node.args[0]
                for kw in node.keywords:
                    if kw.arg in WINDOW_KW:
                        win = kw.value
                v = _num(win) if win is not None else None
                if v is not None and _is_int(win) and v >= 2:
                    p = windows.get(v)
                    if p is None:
                        p = windows[v] = Param(f"window {int(v)}", v, True, "window")
                        order.append(("w", v))
                    p.nodes.append(win)
                    for kw in node.keywords:
                        if kw.arg == "min_periods" and _num(kw.value) is not None:
                            p.dependents.append((kw.value, win))
            elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(
                    node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                lit = node.comparators[0]
                v = _num(lit)
                if v is None:
                    continue
                ok = (not _is_int(lit) and v != 0) or (_is_int(lit) and v >= 2)
                if ok:
                    p = thresholds.get(v)
                    if p is None:
                        label = f"threshold {v:g}"
                        p = thresholds[v] = Param(label, v, _is_int(lit), "threshold")
                        order.append(("t", v))
                    p.nodes.append(lit)
    ws = [windows[v] for k, v in order if k == "w"]
    ts = [thresholds[v] for k, v in order if k == "t"]
    return (ws + ts)[:MAX_INLINE_PARAMS]


def _fmt(v: float, is_int: bool) -> str:
    return str(int(v)) if is_int else repr(float(v))


def apply_values(code: str, params: list[Param], values: list[float]) -> str:
    """Rewrite every occurrence of each parameter with its new value (byte-accurate AST positions)."""
    lines = code.splitlines(keepends=True)
    blines = [ln.encode("utf-8") for ln in lines]
    edits: list[tuple[int, int, int, int, str]] = []
    for p, v in zip(params, values):
        new_txt = _fmt(v, p.is_int)
        for node in p.nodes:
            edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, new_txt))
        for mp, win in p.dependents:
            old_mp = _num(mp)
            nw = max(1, int(round(v)))
            new_mp = max(1, min(nw, int(round(old_mp * v / p.value))))
            edits.append((mp.lineno, mp.col_offset, mp.end_lineno, mp.end_col_offset, str(new_mp)))
    # apply from the end so earlier offsets stay valid; dedupe identical spans
    seen = set()
    for e in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        if (e[0], e[1]) in seen:
            continue
        seen.add((e[0], e[1]))
        l0, c0, l1, c1, txt = e
        if l0 != l1:
            continue    # numeric literals never span lines
        b = blines[l0 - 1]
        blines[l0 - 1] = b[:c0] + txt.encode() + b[c1:]
    return b"".join(blines).decode("utf-8")


def build_grid(params: list[Param], max_variants: int = MAX_VARIANTS, seed: int = 0) -> list[list[float]]:
    """Distinct value combinations; the original values always come first."""
    if not params:
        return []
    levels = LEVELS_1D if len(params) == 1 else LEVELS_ND
    per = [sorted({p.vary(f) for f in levels} | {p.value}) for p in params]
    base = [p.value for p in params]
    combos = [list(c) for c in itertools.product(*per)]
    combos = [c for c in combos if c != base]
    if len(combos) > max_variants - 1:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(combos), size=max_variants - 1, replace=False))
        combos = [combos[i] for i in idx]
    return [base] + combos


def combined_module(codes: list[str]) -> str | None:
    """All variants as closures in one module, so one sandbox process runs them all."""
    parts = []
    for i, c in enumerate(codes):
        body = "".join("    " + ln if ln.strip() else ln for ln in c.splitlines(keepends=True))
        if not body.endswith("\n"):
            body += "\n"
        parts.append(f"def make_variant_{i}():\n{body}    return generate_signals\n\n\nvariant_{i} = make_variant_{i}()\n\n")
    parts.append("\ndef generate_signals(data):\n    return variant_0(data)\n")
    src = "".join(parts)
    try:
        compile(src, "<pbo>", "exec")
    except SyntaxError:
        return None
    return src


# ----------------------------------------------------------------------------
# End to end: code -> grid -> sandbox -> T x N matrix -> CSCV
# ----------------------------------------------------------------------------
def variant_label(params: list[Param], values: list[float]) -> str:
    return ", ".join(f"{p.name}={_fmt(v, p.is_int)}" for p, v in zip(params, values))


def run_variants_sandbox(codes: list[str], data: dict, timeout: float) -> list[pd.DataFrame | None]:
    from sandbox.runner import run_strategy
    src = combined_module(codes)
    if src is not None:
        res = run_strategy(src, [data] * len(codes), timeout=timeout,
                           func_names=[f"variant_{i}" for i in range(len(codes))])
        if res.ok:
            return list(res.outputs)
    out = []                          # fallback: one process per variant (slower, isolates crashes)
    for c in codes:
        r = run_strategy(c, [data], timeout=timeout / max(len(codes), 1) + 10)
        out.append(r.outputs[0] if r.ok else None)
    return out


def skipped(reason: str, **extra) -> dict:
    return {"status": "skipped", "reason": reason, **extra}


def compute_pbo(code: str, data: dict, *, cost_bps: float = 10.0, execution: str = "open",
                timeout: float = 60.0, n_blocks: int | None = None, runner=None, seed: int = 0,
                param_grid: dict[str, list[float]] | None = None) -> dict:
    """PBO of tuning the strategy's own parameters. ``runner(codes, data) -> list of weight frames`` defaults
    to the sandbox. ``param_grid`` ({name: [values]}) overrides the automatic grid for explicit parameters."""
    from engine.backtester import BacktestConfig, run_backtest

    t0 = time.time()
    try:
        params = find_params(code)
    except SyntaxError as e:
        return skipped(f"could not parse the code ({e.msg})")
    if param_grid:
        byname = {p.name: p for p in params}
        chosen = [byname[k] for k in param_grid if k in byname]
        if not chosen:
            return skipped("none of the given grid parameters exist in the code", available=list(byname))
        params = chosen
        grid = [list(c) for c in itertools.product(*[param_grid[p.name] for p in params])][:MAX_VARIANTS]
    else:
        grid = build_grid(params, seed=seed)
    if not params:
        return skipped("no tunable numeric parameters found (no UPPERCASE constants, numeric defaults, lookback "
                       "windows or thresholds), so there is nothing to have overfitted by tuning")
    codes = [apply_values(code, params, g) for g in grid]
    # drop textual duplicates (e.g. rounding of small windows)
    uniq, keep = [], []
    for g, c in zip(grid, codes):
        if c not in uniq:
            uniq.append(c)
            keep.append(g)
    codes, grid = uniq, keep
    if len(codes) < 4:
        return skipped(f"only {len(codes)} distinct parameter variant(s); CSCV needs at least 4",
                       params=[p.name for p in params])
    run = runner or (lambda cs, d: run_variants_sandbox(cs, d, timeout))
    outs = run(codes, data)
    cfg = BacktestConfig(lag=1, execution=execution, cost_bps=cost_bps)
    rets, labels, first_active, failed = [], [], [], 0
    for g, w in zip(grid, outs):
        if w is None:
            failed += 1
            continue
        try:
            bt = run_backtest(w, data, cfg)
        except Exception:  # noqa: BLE001 - a broken variant is dropped, not fatal
            failed += 1
            continue
        act = bt.holdings.abs().sum(axis=1) > 0
        if not act.any():
            failed += 1
            continue
        rets.append(bt.returns)
        labels.append(variant_label(params, g))
        first_active.append(act.idxmax())
    if len(rets) < 4:
        return skipped(f"only {len(rets)} variant(s) ran and traded ({failed} failed or stayed in cash)",
                       params=[p.name for p in params])
    start = max(first_active)
    R = pd.concat(rets, axis=1).loc[start:]
    R.columns = range(R.shape[1])
    # identical return streams are one variant
    dup = R.T.duplicated()
    R = R.loc[:, ~dup.to_numpy()]
    labels = [lab for lab, d in zip(labels, dup) if not d]
    if R.shape[1] < 4:
        return skipped("fewer than 4 variants with distinct returns", params=[p.name for p in params])
    T = len(R)
    S = n_blocks or (10 if T >= 10 * 40 else 8 if T >= 8 * 25 else 0)
    if not S:
        return skipped(f"only {T} common trading days; CSCV needs about 200", params=[p.name for p in params])
    res = cscv(R.to_numpy(), n_blocks=S, seed=seed)
    ann = math.sqrt(252)
    full_sr = (R.mean() / R.std(ddof=1).replace(0, np.nan) * ann).fillna(0.0).to_numpy()
    res.update({
        "status": "ok",
        "params": [{"name": p.name, "base": p.value, "kind": p.kind, "values": sorted({g[i] for g in grid})}
                   for i, p in enumerate(params)],
        "variants": [{"label": lab, "sharpe": round(float(s), 4)} for lab, s in zip(labels, full_sr)],
        "base_label": labels[0] if labels else "",
        "base_rank_full_sample": int(pd.Series(full_sr).rank(ascending=False).iloc[0]) if len(full_sr) else None,
        "n_failed_variants": failed,
        "common_start": str(pd.Timestamp(start).date()),
        "is_sharpe_best_annual": [round(x * ann, 4) for x in res["is_sharpe_best"]],
        "oos_sharpe_best_annual": [round(x * ann, 4) for x in res["oos_sharpe_best"]],
        "seconds": round(time.time() - t0, 2),
    })
    return res


def risk_label(pbo: float) -> str:
    return "high" if pbo > PBO_FAIL else ("moderate" if pbo > PBO_MODERATE else "low")

"""Deterministic audit of a strategy's signals.

Every test returns a structured result::

    {"name", "passed", "severity", "summary", "evidence", "suggested_fix"}

Severity: critical > high > medium > low > info. A strategy is **clean** when
no critical/high test fails. The LLM auditor never overrides these results.

Signals are produced through an *executor*: ``executor(list_of_datasets) ->
list_of_weight_frames``. In production it is the sandbox (one subprocess for all
variants); in unit tests it can call a Python function directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from audit.ast_scan import scan_code
from audit.overfitting import deflated_sharpe, split_sharpes
from data.synthetic import FIELDS
from engine.backtester import BacktestConfig, align_weights, run_backtest

Executor = Callable[[list[dict]], list[pd.DataFrame]]
BLOCKING = ("critical", "high")
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


@dataclass
class TestResult:
    name: str
    passed: bool
    severity: str
    summary: str
    evidence: dict = field(default_factory=dict)
    suggested_fix: str = ""

    @property
    def blocking(self) -> bool:
        return (not self.passed) and self.severity in BLOCKING

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": bool(self.passed), "severity": self.severity,
                "blocking": self.blocking, "summary": self.summary,
                "evidence": _jsonable(self.evidence), "suggested_fix": self.suggested_fix}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ----------------------------------------------------------------------------
# Data variants
# ----------------------------------------------------------------------------
def _copy(data: dict) -> dict:
    return {k: data[k].copy() for k in FIELDS}


def perturb_after(data: dict, cut_pos: int, rng: np.random.Generator, n_kill: int = 3) -> dict:
    """Return a copy whose rows <= cut_pos are bit-identical and rows after are scrambled.

    After the cut: returns are time-shuffled, sign-flipped per ticker and noised;
    OHLC are rescaled consistently; volume is noised; a few live tickers are
    'delisted' shortly after the cut (tests universe/survivorship dependence).
    """
    out = _copy(data)
    close = data["close"]
    T, N = close.shape
    m = T - cut_pos - 1
    if m <= 0:
        return out
    old_r = close.pct_change(fill_method=None).to_numpy()[cut_pos + 1:]
    old_r = np.nan_to_num(old_r, nan=0.0)
    perm = rng.permutation(m)
    signs = rng.choice([-1.0, 1.0], size=(1, N))
    new_r = np.clip(old_r[perm] * signs + 0.01 * rng.standard_normal((m, N)), -0.5, 0.5)
    factor = np.cumprod((1 + new_r) / (1 + old_r), axis=0)
    for f in ("open", "high", "low", "close"):
        arr = out[f].to_numpy(copy=True)
        arr[cut_pos + 1:] = arr[cut_pos + 1:] * factor
        out[f] = pd.DataFrame(arr, index=close.index, columns=close.columns)
    vol = out["volume"].to_numpy(copy=True)
    vol[cut_pos + 1:] = vol[cut_pos + 1:] * rng.lognormal(0, 0.5, size=(m, N))
    out["volume"] = pd.DataFrame(vol, index=close.index, columns=close.columns)
    alive = np.where(close.iloc[cut_pos].notna().to_numpy())[0]
    if len(alive) and n_kill > 0 and m > 3:
        kill = rng.choice(alive, size=min(n_kill, len(alive)), replace=False)
        for f in FIELDS:
            arr = out[f].to_numpy(copy=True)
            arr[cut_pos + 3:, kill] = np.nan
            out[f] = pd.DataFrame(arr, index=close.index, columns=close.columns)
    return out


def choose_cuts(n_rows: int, n_cuts: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    lo, hi = max(60, int(0.15 * n_rows)), int(0.9 * n_rows)
    if hi <= lo:
        return []
    return sorted(int(x) for x in rng.choice(np.arange(lo, hi), size=min(n_cuts, hi - lo), replace=False))


def revive_delisted(data: dict, seed: int, max_names: int = 3, min_gap: int = 5) -> tuple[dict | None, list, int]:
    """Counterfactual where the latest-delisted names keep trading (random walk).

    Returns (data_cf, revived_tickers, compare_until_pos). Rows strictly before
    the earliest revived delisting must be unchanged for a point-in-time strategy.
    """
    close = data["close"]
    T = len(close)
    last_pos = {c: close.index.get_loc(close[c].last_valid_index())
                for c in close.columns if close[c].notna().any()}
    delisted = [(p, c) for c, p in last_pos.items() if p < T - min_gap]
    if not delisted:
        return None, [], 0
    delisted.sort(reverse=True)
    chosen = delisted[:max_names]
    rng = np.random.default_rng(seed)
    out = _copy(data)
    for p, c in chosen:
        n = T - p - 1
        r = 0.0003 + 0.02 * rng.standard_normal(n)
        path = close[c].iloc[p] * np.cumprod(1 + r)
        j = close.columns.get_loc(c)
        for f in ("open", "high", "low", "close"):
            arr = out[f].to_numpy(copy=True)
            arr[p + 1:, j] = path
            out[f] = pd.DataFrame(arr, index=close.index, columns=close.columns)
        vol = out["volume"].to_numpy(copy=True)
        vol[p + 1:, j] = float(np.nanmedian(data["volume"][c]))
        out["volume"] = pd.DataFrame(vol, index=close.index, columns=close.columns)
    compare_until = min(p for p, _ in chosen)
    return out, [c for _, c in chosen], compare_until


# ----------------------------------------------------------------------------
# Individual tests
# ----------------------------------------------------------------------------
def _diff_rows(a: pd.DataFrame, b: pd.DataFrame, upto: int) -> np.ndarray:
    x = a.iloc[: upto + 1].to_numpy()
    y = b.iloc[: upto + 1].to_numpy()
    return np.abs(x - y) > (1e-9 + 1e-7 * np.abs(x))


def evaluate_perturbation(data: dict, base_w: pd.DataFrame, pert_ws: list[pd.DataFrame], cuts: list[int]) -> TestResult:
    idx = data["close"].index
    per_cut, first_leak, worst = [], None, 0.0
    for cut, pw in zip(cuts, pert_ws):
        changed = _diff_rows(base_w, pw, cut)
        rows = np.where(changed.any(axis=1))[0]
        mag = float(np.abs(base_w.iloc[: cut + 1].to_numpy() - pw.iloc[: cut + 1].to_numpy()).max()) if cut >= 0 else 0.0
        entry = {"cut_date": idx[cut], "leaked": bool(len(rows)), "rows_changed": int(len(rows)),
                 "max_abs_weight_change": round(mag, 6)}
        if len(rows):
            entry["earliest_changed_date"] = idx[rows[0]]
            entry["latest_changed_date"] = idx[rows[-1]]
            tick = data["close"].columns[np.where(changed[rows[-1]])[0][:5]].tolist()
            entry["example_tickers"] = tick
            if first_leak is None or idx[rows[0]] < first_leak:
                first_leak = idx[rows[0]]
            worst = max(worst, mag)
        per_cut.append(entry)
    n_leak = sum(e["leaked"] for e in per_cut)
    passed = n_leak == 0
    if passed:
        summary = f"Signals on or before {len(cuts)} random cut dates are unchanged when all later data is scrambled."
    else:
        full = any(e["rows_changed"] > 0 and e["earliest_changed_date"] <= idx[min(cuts) // 4] for e in per_cut if e["leaked"])
        summary = (f"LOOK-AHEAD: scrambling data after the cut changed past signals at {n_leak}/{len(cuts)} cut dates "
                   f"(first leaking date {first_leak:%Y-%m-%d}, max weight change {worst:.3f})."
                   + (" Changes reach the start of the sample -> full-sample statistic or survivor universe." if full else ""))
    return TestResult(
        "future_perturbation", passed, "critical", summary,
        {"n_cuts": len(cuts), "n_leaking_cuts": n_leak,
         "first_leaking_date": first_leak, "max_abs_weight_change": round(worst, 6), "cuts": per_cut},
        "" if passed else ("Every value used for row t must be computable from rows <= t. Remove negative shifts, "
                           "centred windows, backward fills, full-sample normalisation and universes defined by "
                           "future availability (e.g. dropna(axis=1)). Use rolling/expanding estimators."))


def evaluate_shift(s1: float, s2: float, min_sharpe: float = 1.0, max_drop: float = 0.25) -> TestResult:
    drop = (s1 - s2) / abs(s1) if abs(s1) > 1e-9 else 0.0
    passed = not (s1 > min_sharpe and drop > max_drop)
    return TestResult(
        "execution_delay", passed, "high",
        (f"Sharpe {s1:.2f} at t+1 vs {s2:.2f} at t+2 (drop {drop:.0%}); robust to one extra bar of delay."
         if passed else
         f"Sharpe collapses from {s1:.2f} (t+1) to {s2:.2f} (t+2), a {drop:.0%} drop: the edge lives in the "
         f"bar right after the signal, typical of same-bar or look-ahead execution."),
        {"sharpe_lag1": round(s1, 4), "sharpe_lag2": round(s2, 4), "relative_drop": round(drop, 4),
         "threshold_drop": max_drop, "min_sharpe_to_check": min_sharpe},
        "" if passed else "Make sure the signal for row t uses only closes up to t and that no future bar leaks in.")


def evaluate_plausibility(metrics: dict, max_sharpe: float = 3.0) -> TestResult:
    s = metrics.get("sharpe", 0.0)
    passed = s <= max_sharpe
    return TestResult(
        "plausibility", passed, "medium",
        (f"Sharpe {s:.2f} is within a plausible range for a daily equity strategy."
         if passed else f"Sharpe {s:.2f} (CAGR {metrics.get('cagr', 0):.0%}) after costs is implausibly high "
                        f"for daily equities; treat as a bug until proven otherwise."),
        {"sharpe": round(s, 4), "cagr": round(metrics.get("cagr", 0.0), 4), "max_sharpe": max_sharpe},
        "" if passed else "Re-check timing and data alignment; real edges rarely exceed Sharpe 2-3.")


def evaluate_survivorship(data: dict, base_w: pd.DataFrame, diag: dict,
                          cf_w: pd.DataFrame | None, revived: list, compare_until: int) -> TestResult:
    close = data["close"]
    T = len(close)
    last_valid = close.apply(lambda s: s.last_valid_index())
    delisted = [c for c in close.columns if last_valid[c] is not None and close.index.get_loc(last_valid[c]) < T - 5]
    ev: dict = {"n_tickers": int(close.shape[1]), "n_delisted_in_data": len(delisted)}
    problems, severity, fixes = [], "info", []

    wbl = float(diag.get("weight_before_listing", 0.0))
    ev["weight_before_listing"] = round(wbl, 6)
    if wbl > 1e-9:
        problems.append("weights assigned to tickers before they were listed")
        severity = "critical"
        fixes.append("Mask weights with close.notna(): a ticker cannot be held before it exists.")

    if cf_w is not None:
        changed = _diff_rows(base_w, cf_w, compare_until - 1)
        rows = np.where(changed.any(axis=1))[0]
        ev["counterfactual_revived"] = revived
        ev["counterfactual_compare_until"] = close.index[compare_until - 1]
        ev["counterfactual_rows_changed"] = int(len(rows))
        if len(rows):
            ev["counterfactual_first_change"] = close.index[rows[0]]
            problems.append("past signals change when delisted names are made to survive "
                            "(universe depends on future survival)")
            if severity != "critical":
                severity = "high"
            fixes.append("Define the universe point-in-time (eligible = close.notna() on each date); "
                         "never drop tickers because they have missing data later.")

    if delisted:
        held = base_w[delisted].abs().sum(axis=1)
        alive_d = close[delisted].notna().sum(axis=1)
        alive_all = close.notna().sum(axis=1).replace(0, np.nan)
        ev["avg_weight_on_later_delisted"] = round(float(held.mean()), 4)
        ev["avg_universe_share_of_later_delisted"] = round(float((alive_d / alive_all).mean()), 4)
    else:
        ev["data_warning"] = "Dataset contains no delisted tickers: results are survivorship-biased by construction."
        if not problems:
            severity = "medium"
            problems.append("dataset has no delisted tickers")
            fixes.append("Use a point-in-time dataset that includes delisted securities.")

    passed = not problems
    if passed:
        severity = "high"
        summary = (f"Point-in-time universe: no weight before listing; signals unchanged when "
                   f"{len(revived)} delisted names are revived.")
    else:
        summary = "SURVIVORSHIP: " + "; ".join(problems) + "."
    return TestResult("survivorship", passed, severity, summary, ev, " ".join(fixes))


def evaluate_overfitting(returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None,
                         dsr_threshold: float = 0.95) -> list[TestResult]:
    d = deflated_sharpe(returns.to_numpy(), n_trials, trial_sharpes)
    dsr_pass = d["dsr"] >= dsr_threshold
    dsr = TestResult(
        "deflated_sharpe", dsr_pass, "medium",
        (f"Deflated Sharpe {d['dsr']:.2f} >= {dsr_threshold} after {n_trials} trial(s): edge survives selection bias."
         if dsr_pass else
         f"Deflated Sharpe {d['dsr']:.2f} < {dsr_threshold} after {n_trials} trial(s): observed Sharpe "
         f"{d.get('sr_annual', 0):.2f} is not clearly above the {d.get('sr0_annual', 0):.2f} expected from luck."),
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} | {"threshold": dsr_threshold},
        "" if dsr_pass else "Fewer variants, longer history, or out-of-sample confirmation before trusting this edge.")
    tr, te = split_sharpes(returns.to_numpy(), 0.7)
    degrade = (tr - te) / abs(tr) if abs(tr) > 1e-9 else 0.0
    tt_pass = not (tr > 0.5 and (te < 0 or degrade > 0.75))
    tt = TestResult(
        "train_test_split", tt_pass, "medium",
        (f"Sharpe train {tr:.2f} vs test {te:.2f} (70/30 split): no severe degradation."
         if tt_pass else f"Sharpe falls from {tr:.2f} in-sample to {te:.2f} out-of-sample (70/30 split)."),
        {"sharpe_train": round(tr, 4), "sharpe_test": round(te, 4), "relative_degradation": round(degrade, 4)},
        "" if tt_pass else "Simplify the rule or reduce tuned parameters; validate on unseen data.")
    return [dsr, tt]


def evaluate_static(code: str | None) -> TestResult:
    if not code:
        return TestResult("static_scan", True, "info", "No source code available for static scan.", {})
    findings = scan_code(code)
    worst = min((f.severity for f in findings), key=lambda s: SEVERITY_ORDER[s], default="info")
    blocking = [f for f in findings if f.severity in BLOCKING]
    passed = not blocking
    summary = (f"No blocking red flags ({len(findings)} soft warning(s))." if passed else
               f"{len(blocking)} red flag(s): " + ", ".join(sorted({f.rule for f in blocking})) + ".")
    return TestResult("static_scan", passed, worst if findings else "info", summary,
                      {"findings": [f.to_dict() for f in findings]},
                      " ".join(dict.fromkeys(f.fix for f in findings)))


# ----------------------------------------------------------------------------
# Full audit
# ----------------------------------------------------------------------------
@dataclass
class AuditReport:
    tests: list[TestResult]
    metrics: dict
    metrics_lag2: dict
    equity: pd.Series
    returns: pd.Series
    diagnostics: dict

    @property
    def clean(self) -> bool:
        return not any(t.blocking for t in self.tests)

    @property
    def failing(self) -> list[TestResult]:
        return [t for t in self.tests if not t.passed]

    def to_dict(self) -> dict:
        return {"clean": self.clean,
                "tests": [t.to_dict() for t in sorted(self.tests, key=lambda t: (t.passed, SEVERITY_ORDER[t.severity]))],
                "metrics": _jsonable(self.metrics), "metrics_lag2": _jsonable(self.metrics_lag2),
                "diagnostics": _jsonable(self.diagnostics)}


def in_process_executor(fn: Callable[[dict], pd.DataFrame]) -> Executor:
    return lambda datasets: [fn(_copy(d)) for d in datasets]


def run_audit(data: dict, executor: Executor, code: str | None = None, *, n_trials: int = 1,
              trial_sharpes: list[float] | None = None, cost_bps: float = 10.0, execution: str = "open",
              n_cuts: int = 6, seed: int = 7) -> AuditReport:
    close = data["close"]
    cuts = choose_cuts(len(close), n_cuts, seed)
    rng = np.random.default_rng(seed + 1)
    variants = [perturb_after(data, c, rng) for c in cuts]
    cf, revived, until = revive_delisted(data, seed + 2)
    datasets = [data] + variants + ([cf] if cf is not None else [])
    outs = executor(datasets)
    if len(outs) != len(datasets):
        raise RuntimeError(f"executor returned {len(outs)} outputs for {len(datasets)} datasets")

    base_w, diag = align_weights(outs[0], data)
    pert_ws = [align_weights(o, d)[0] for o, d in zip(outs[1:1 + len(cuts)], variants)]
    cf_w = align_weights(outs[-1], cf)[0] if cf is not None else None

    bt1 = run_backtest(base_w, data, BacktestConfig(lag=1, execution=execution, cost_bps=cost_bps))
    bt2 = run_backtest(base_w, data, BacktestConfig(lag=2, execution=execution, cost_bps=cost_bps))

    tests = [
        evaluate_perturbation(data, base_w, pert_ws, cuts),
        evaluate_survivorship(data, base_w, diag, cf_w, revived, until),
        evaluate_shift(bt1.metrics["sharpe"], bt2.metrics["sharpe"]),
        evaluate_static(code),
        evaluate_plausibility(bt1.metrics),
        *evaluate_overfitting(bt1.returns.loc[bt1.holdings.abs().sum(axis=1).gt(0).idxmax():]
                              if bt1.holdings.abs().sum().sum() > 0 else bt1.returns,
                              n_trials, trial_sharpes),
    ]
    if bt1.metrics["avg_gross"] == 0:
        tests.append(TestResult("non_trivial", False, "high", "Strategy never takes a position.",
                                {}, "Return non-zero target weights."))
    return AuditReport(tests, bt1.metrics, bt2.metrics, bt1.equity, bt1.returns, diag)

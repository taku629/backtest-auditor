"""Repair benchmark: can the agentic loop turn the 24 leaky LeakBench strategies into honest ones?

For every leaky strategy (``bench/strategies/L*.py``, LeakBench header stripped) the full
``agents.loop.run_audit_loop`` is run with the stripped code as ``initial_code`` (iteration 1 audits the
user's code unchanged; the Auditor diagnoses, the Coder repairs, the deterministic tests re-check), under
two conditions:

* ``A`` full system: the Auditor sees the test evidence including the dynamic leak localization
  (``evidence["localized"]`` in the compacted test JSON) and the ``localized_hint`` line.
* ``B`` ablation: identical, but ``agents.prompts.localized_hint`` returns ``""`` and ``evidence["localized"]``
  is removed from the compacted tests. Implemented by thread-local monkeypatching in this runner only;
  ``agents/`` and ``audit/`` are not edited.

Per run it records: repaired (``final_status == "clean"``), iterations used, leaky vs final Sharpe, LLM cost,
over-correction (final ``non_trivial`` failing or almost no exposure), and whether the first Auditor diagnosis
names the localized line.

Every fresh paid call is appended to ``bench/repair_spend.jsonl``; every response is cached in
``bench/repair_cache.json`` keyed by a hash of (model, sampling settings, messages), so reruns are free.
A hard budget cap (``--budget``, default $2.00, counted over the whole spend log) stops new paid calls.

Usage::

    NEBIUS_API_KEY=... python -m bench.run_repair --only L01 L02 L03     # pilot, to estimate cost
    NEBIUS_API_KEY=... python -m bench.run_repair                        # all 24, both conditions
    NEBIUS_API_KEY=... python -m bench.run_repair --conditions A         # full system only
    python -m bench.run_repair --mock --only L01                         # offline code path (MockLLM)
    python -m bench.run_repair --report-only                             # rebuild REPAIR.md from the json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_JSON = BENCH_DIR / "repair_results.json"
REPAIR_MD = BENCH_DIR / "REPAIR.md"
CACHE_PATH = BENCH_DIR / "repair_cache.json"
SPEND_LOG = BENCH_DIR / "repair_spend.jsonl"
DEFAULT_BUDGET = 2.00
MAX_ITERS = 3
CACHE_VERSION = "r1"
TRIVIAL_GROSS = 0.05        # mean gross exposure below this = the "fix" mostly sits in cash

# Neutral one-line intent per strategy: what the strategy is meant to do, never the bug.
HYPOTHESES = {
    "L01_shift_minus_one": "20-day momentum: hold the top quintile of names, equal weight.",
    "L02_centred_rolling": "Mean reversion: buy the 8 names trading furthest below their ~20-day moving average, equal weight.",
    "L03_bfill_prices": "60-day momentum: long the 10 names with the highest 60-day return, equal weight.",
    "L04_fullsample_zscore": "Volatility-adjusted 40-day momentum, standardised per ticker; long the top 10, equal weight.",
    "L05_dropna_universe": "Low volatility: hold the 10 least volatile names by 60-day volatility, rebalanced monthly.",
    "L06_select_by_final_return": "Trend following on market leaders (strongest names over the past year): long leaders trading above their 50-day average.",
    "L07_time_rank_full_history": "Buy stocks whose 20-day momentum is high (top 20%) relative to their own history, equal weight.",
    "L08_reverse_cummax": "Drawdown reversal: buy the 10 names trading at the largest discount to their running peak, equal weight.",
    "L09_weekly_resample_bfill": "20-day momentum, top decile, equal weight, weekly rebalance.",
    "L10_reversed_expanding_vol": "Inverse-volatility weights on the 25 lowest-volatility names, using a stable long-window volatility estimate.",
    "L11_groupby_quantile_threshold": "Breakout: hold a stock when its 20-day momentum is above its own usual top-quintile level, equal weight.",
    "L12_helper_hidden_shift": "5-day momentum: hold the top quintile of names, equal weight.",
    "L13_alive_at_end": "120-day momentum: equal-weight the top third of tradable names.",
    "L14_polyfit_detrend": "Mean reversion around each stock's log-price trend line: buy the 8 names furthest below trend.",
    "L15_min_history_filter": "Low volatility among seasoned stocks (long trading record): hold the 10 least volatile names, equal weight.",
    "L16_liquidity_by_recent_volume": "20-day reversal among the 25 most liquid stocks: buy the 5 biggest losers, equal weight.",
    "L17_rank_default_axis": "Cross-sectional momentum: hold names in the top quintile of 20-day return, equal weight.",
    "L18_sub_mean_axis_trap": "Market-neutral-style momentum: long the 10 names with the highest demeaned 60-day return, equal weight.",
    "L19_resample_ms_last": "3-month momentum: long the top 10 names, rebalanced monthly.",
    "L20_groupby_transform_last": "Low volatility: 10 least volatile names (60-day vol), one portfolio per calendar month.",
    "L21_trailing_ic_forward_return": "Adaptive 5-day reversal: trade reversal only on stocks where it has recently worked (positive trailing information coefficient); top 8.",
    "L22_descending_sort_rolling": "Mean reversion: buy the 8 names furthest below their 20-day moving average, equal weight.",
    "L23_fillna_full_mean": "60-day momentum: long the top 10, with new listings included in the ranking.",
    "L24_pooled_qcut": "Quintile sort on 20-day momentum: hold stocks in the top quintile, equal weight.",
}


# ----------------------------------------------------------------------------
# Condition switch (thread-local, so A and B runs can share one process)
# ----------------------------------------------------------------------------
_COND = threading.local()
_PATCH_LOCK = threading.Lock()
_PATCHED = {"done": False}


def _install_condition_patches() -> None:
    """Wrap agents.prompts.localized_hint / compact_tests once; behaviour depends on the thread's condition."""
    from agents import prompts as P

    with _PATCH_LOCK:
        if _PATCHED["done"]:
            return
        orig_hint, orig_compact = P.localized_hint, P.compact_tests

        def localized_hint(tests):
            if getattr(_COND, "no_localization", False):
                return ""
            return orig_hint(tests)

        def compact_tests(tests, include_passed=True):
            if getattr(_COND, "no_localization", False):
                tests = [dict(t, evidence={k: v for k, v in (t.get("evidence") or {}).items() if k != "localized"})
                         for t in tests]
            return orig_compact(tests, include_passed=include_passed)

        P.localized_hint, P.compact_tests = localized_hint, compact_tests
        _PATCHED.update(done=True, orig_hint=orig_hint, orig_compact=orig_compact)


@contextmanager
def condition(name: str):
    _install_condition_patches()
    prev = getattr(_COND, "no_localization", False)
    _COND.no_localization = (name == "B")
    try:
        yield
    finally:
        _COND.no_localization = prev


# ----------------------------------------------------------------------------
# Cached, budgeted LLM wrapper
# ----------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return default
    return default


def logged_spend(path: Path = SPEND_LOG) -> float:
    if not path.exists():
        return 0.0
    tot = 0.0
    for ln in path.read_text().splitlines():
        if ln.strip():
            try:
                tot += float(json.loads(ln).get("cost_usd") or 0.0)
            except json.JSONDecodeError:
                pass
    return tot


class CachedBudgetLLM:
    """Wraps NebiusLLM/MockLLM: cache by prompt hash, log every fresh paid call, stop at the budget."""

    def __init__(self, inner, *, cache_path: Path = CACHE_PATH, spend_path: Path = SPEND_LOG,
                 budget: float = DEFAULT_BUDGET, use_cache: bool = True):
        self.inner, self.config = inner, inner.config
        self.is_mock = bool(getattr(inner, "is_mock", False))
        self.cache_path, self.spend_path, self.budget = cache_path, spend_path, budget
        self.use_cache = use_cache and not self.is_mock
        self.cache = load_json(cache_path, {}) if self.use_cache else {}
        self.spent = logged_spend(spend_path) if not self.is_mock else 0.0
        self.lock = threading.Lock()
        self.tag = threading.local()   # per-thread (strategy, condition) for the spend log

    def key(self, role: str, messages: list[dict]) -> str:
        r = self.config.route(role)
        d = {"v": CACHE_VERSION, "model": r.model, "t": r.temperature, "mt": r.max_tokens,
             "think": r.thinking, "messages": messages}
        rep = int(getattr(self.tag, "replicate", 0) or 0)
        if rep:  # independent sample of the same prompt (replicate 0 keeps the original keys)
            d["replicate"] = rep
        blob = json.dumps(d, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def chat(self, role: str, messages: list[dict], context: dict | None = None):
        from agents.config import estimate_cost
        from agents.llm import LLMResponse

        k = self.key(role, messages)
        if self.use_cache:
            with self.lock:
                hit = self.cache.get(k)
            if hit:
                self.tag.cached = getattr(self.tag, "cached", 0) + 1
                return LLMResponse(hit["text"], role, hit["model"], hit["prompt_tokens"], hit["completion_tokens"],
                                   hit["latency_s"], False)
        if not self.is_mock:
            with self.lock:
                if self.spent >= self.budget:
                    raise BudgetExceeded(f"budget ${self.budget:.2f} reached (spent ${self.spent:.4f})")
        resp = self.inner.chat(role, messages, context=context)
        if self.is_mock:
            return resp
        cost = estimate_cost(resp.model, resp.prompt_tokens, resp.completion_tokens) or 0.0
        with self.lock:
            self.spent += cost
            self.tag.fresh = getattr(self.tag, "fresh", 0.0) + cost
            with open(self.spend_path, "a") as fh:
                fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "strategy": getattr(self.tag, "strategy", None),
                                     "condition": getattr(self.tag, "condition", None),
                                     "replicate": int(getattr(self.tag, "replicate", 0) or 0),
                                     "role": role, "model": resp.model, "prompt_tokens": resp.prompt_tokens,
                                     "completion_tokens": resp.completion_tokens,
                                     "latency_s": round(resp.latency_s, 2), "cost_usd": round(cost, 6),
                                     "cache_key": k}) + "\n")
            if self.use_cache and resp.text.strip():
                self.cache[k] = {"text": resp.text, "model": resp.model, "prompt_tokens": resp.prompt_tokens,
                                 "completion_tokens": resp.completion_tokens,
                                 "latency_s": round(resp.latency_s, 2), "role": role}
                self.cache_path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))
        return resp


# ----------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------
def _sharpe(m: dict) -> float | None:
    return None if not m or "sharpe" not in m else round(float(m["sharpe"]), 3)


def mentions_line(text: str, loc: dict | None) -> bool | None:
    """Does the diagnosis name the localized statement (by line number or by its source text)?"""
    if not loc or not loc.get("found") or not loc.get("entry") or not text:
        return None
    e = loc["entry"]
    n = int(e["line"])
    if re.search(rf"\b[Ll]ines?\s*(?:\d+\s*(?:[-–,]|and)\s*)*{n}\b", text) or re.search(rf"\bL{n}\b", text):
        return True
    src = (e.get("source_line") or "").split("#")[0].strip()
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    if src and norm(src) and norm(src) in norm(text):
        return True
    rhs = src.split("=", 1)[1].strip() if "=" in src and not src.startswith("return") else src.replace("return", "", 1).strip()
    return bool(rhs) and len(norm(rhs)) >= 10 and norm(rhs) in norm(text)


def run_one(strategy: dict, cond: str, data: dict, llm: CachedBudgetLLM, max_iters: int = MAX_ITERS,
            replicate: int = 0) -> dict:
    from agents.loop import run_audit_loop

    hyp = HYPOTHESES.get(strategy["id"]) or strategy.get("hypothesis") or strategy["id"]
    llm.tag.strategy, llm.tag.condition, llm.tag.fresh, llm.tag.cached = strategy["id"], cond, 0.0, 0
    llm.tag.replicate = replicate
    rec = {"id": strategy["id"], "condition": cond, "replicate": replicate, "bug": strategy.get("bug", ""),
           "hypothesis": hyp, "max_iters": max_iters}
    t0 = time.time()
    try:
        with condition(cond):
            tr = run_audit_loop(hyp, data, llm, max_iters=max_iters, summarize=False,
                                initial_code=strategy["stripped"])
    except BudgetExceeded as e:
        return rec | {"status": "skipped_budget", "error": str(e), "fresh_cost_usd": round(llm.tag.fresh, 6)}
    except Exception as e:  # noqa: BLE001 - an API failure is recorded, never hidden
        return rec | {"status": "run_error", "error": f"{type(e).__name__}: {e}"[:500],
                      "fresh_cost_usd": round(llm.tag.fresh, 6)}
    its = tr.iterations
    first, last = its[0], its[-1]
    fp0 = next((t for t in first.tests if t["name"] == "future_perturbation"), None)
    loc = (fp0 or {}).get("evidence", {}).get("localized") if fp0 else None
    diag0 = first.auditor_feedback
    fails_last = [t["name"] for t in last.tests if not t["passed"]]
    blocking_last = [t["name"] for t in last.tests if t.get("blocking")]
    gross = last.metrics.get("avg_gross") if last.metrics else None
    non_trivial_failed = "non_trivial" in fails_last
    over = bool(last.status != "error" and (non_trivial_failed or (gross is not None and gross < TRIVIAL_GROSS)))
    entry = (loc or {}).get("entry") or {}
    return rec | {
        "status": "done", "final_status": tr.final_status, "repaired": tr.final_status == "clean",
        "repaired_non_trivial": tr.final_status == "clean" and not over,
        "iterations": len(its), "repair_attempts": len(its) - 1,
        "leaky_sharpe": _sharpe(first.metrics), "final_sharpe": _sharpe(last.metrics),
        "final_cagr": None if not last.metrics else round(float(last.metrics.get("cagr", 0.0)), 4),
        "final_avg_gross": None if gross is None else round(float(gross), 4),
        "over_corrected": over, "non_trivial_failed": non_trivial_failed,
        "first_blocking": [t["name"] for t in first.tests if t.get("blocking")],
        "final_failing": fails_last, "final_blocking": blocking_last,
        "statuses": [it.status for it in its], "errors": [it.error[:300] for it in its if it.error],
        "localized_line": entry.get("line"), "localized_source": entry.get("source_line"),
        "localized_found": bool((loc or {}).get("found")),
        "diag1_mentions_localized_line": mentions_line(diag0, loc),
        "diag1_excerpt": diag0[:700],
        "final_code": last.code,
        "cost_usd": round(sum(c.get("cost_usd") or 0.0 for c in tr.calls), 6),
        "fresh_cost_usd": round(llm.tag.fresh, 6), "cached_calls": llm.tag.cached,
        "n_calls": len(tr.calls),
        "tokens": {"prompt": sum(c["prompt_tokens"] for c in tr.calls),
                   "completion": sum(c["completion_tokens"] for c in tr.calls)},
        "seconds": round(time.time() - t0, 1),
        "models": tr.models,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def leaky_strategies(only: list[str] | None = None) -> list[dict]:
    from bench.run_bench import load_strategies

    out = [s for s in load_strategies() if s["leaky"]]
    if only:
        out = [s for s in out if any(o in s["id"] for o in only)]
    return out


def run_repair(*, only: list[str] | None = None, conditions: tuple[str, ...] = ("A", "B"), mock: bool = False,
               data: dict | None = None, budget: float = DEFAULT_BUDGET, workers: int = 4,
               max_iters: int = MAX_ITERS, results_path: Path | None = RESULTS_JSON,
               cache_path: Path = CACHE_PATH, spend_path: Path = SPEND_LOG, llm=None, rerun: bool = False,
               replicates: tuple[int, ...] = (0,), verbose: bool = True) -> dict:
    """Run (strategy x condition) repairs; merges into ``results_path`` (keyed by id+condition) when given."""
    from concurrent.futures import ThreadPoolExecutor

    from agents.llm import make_llm

    log = print if verbose else (lambda *a, **k: None)
    if data is None:
        from data import generate_panel
        data = generate_panel(seed=51)
    inner = llm or make_llm(mock=mock)
    wrapped = CachedBudgetLLM(inner, cache_path=cache_path, spend_path=spend_path, budget=budget)
    strategies = leaky_strategies(only)
    prev = load_json(results_path, {}) if results_path else {}
    done = {(r["id"], r["condition"], r.get("replicate", 0)): r for r in prev.get("runs", [])}
    todo = [(s, c, k) for k in replicates for c in conditions for s in strategies
            if rerun or (s["id"], c, k) not in done or done[(s["id"], c, k)].get("status") != "done"]
    spent0 = wrapped.spent
    log(f"[repair] {len(todo)} runs ({'mock' if wrapped.is_mock else 'live'}), budget ${budget:.2f}, "
        f"already logged ${spent0:.4f}")

    def one(sc):
        s, c, k = sc
        r = run_one(s, c, data, wrapped, max_iters=max_iters, replicate=k)
        log(f"[repair] {s['id']:<32} {c}{k}  {r.get('final_status', r['status']):<10} "
            f"iters={r.get('iterations')}  sharpe {r.get('leaky_sharpe')} -> {r.get('final_sharpe')}  "
            f"over={r.get('over_corrected')}  diag@line={r.get('diag1_mentions_localized_line')}  "
            f"fresh ${r.get('fresh_cost_usd', 0):.4f}  ({r.get('seconds')}s)")
        return r

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        new = list(ex.map(one, todo))
    for r in new:
        done[(r["id"], r["condition"], r.get("replicate", 0))] = r
    order = {s["id"]: i for i, s in enumerate(leaky_strategies())}
    runs = sorted(done.values(), key=lambda r: (r.get("replicate", 0), r["condition"], order.get(r["id"], 99)))
    res = {"benchmark": "LeakBench repair", "version": 1,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "mock": wrapped.is_mock, "max_iters": max_iters, "summarize": False,
           "data": {"source": "data.generate_panel(seed=51)", "seed": 51,
                    "n_tickers": int(data["close"].shape[1]), "n_dates": int(data["close"].shape[0])},
           "models": {r: wrapped.config.route(r).model for r in ("coder", "auditor")},
           "spend_logged_usd": round(logged_spend(spend_path), 6) if not wrapped.is_mock else 0.0,
           "spend_this_invocation_usd": round(wrapped.spent - spent0, 6),
           "runs": runs}
    res["summary"] = summarize(runs)
    if results_path:
        results_path.write_text(json.dumps(res, indent=1, default=str))
    return res


def summarize(runs: list[dict]) -> dict:
    out = {}
    for c in ("A", "B"):
        rs = [r for r in runs if r["condition"] == c and r.get("status") == "done"]
        if not rs:
            continue
        rep = [r for r in rs if r["repaired"]]
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        diag = [r["diag1_mentions_localized_line"] for r in rs if r.get("diag1_mentions_localized_line") is not None]
        out[c] = {"n": len(rs), "repaired": len(rep), "repair_rate": mean([1.0 if r["repaired"] else 0.0 for r in rs]),
                  "repaired_non_trivial": sum(1 for r in rs if r["repaired_non_trivial"]),
                  "over_corrected": sum(1 for r in rs if r["over_corrected"]),
                  "mean_iterations_repaired": mean([r["iterations"] for r in rep]),
                  "mean_repair_attempts_repaired": mean([r["repair_attempts"] for r in rep]),
                  "repaired_first_attempt": sum(1 for r in rep if r["iterations"] == 2),
                  "total_cost_usd": round(sum(r["cost_usd"] for r in rs), 6),
                  "mean_cost_per_run_usd": mean([r["cost_usd"] for r in rs]),
                  "cost_per_repair_usd": round(sum(r["cost_usd"] for r in rs) / len(rep), 6) if rep else None,
                  "diag1_mentions_localized_line": sum(1 for d in diag if d), "diag1_checked": len(diag),
                  "median_leaky_sharpe": _median([r["leaky_sharpe"] for r in rs]),
                  "median_final_sharpe_repaired": _median([r["final_sharpe"] for r in rep])}
        reps = sorted({r.get("replicate", 0) for r in rs})
        out[c]["replicates"] = reps
        out[c]["by_replicate"] = {str(k): {"n": sum(1 for r in rs if r.get("replicate", 0) == k),
                                           "repaired": sum(1 for r in rs if r.get("replicate", 0) == k and r["repaired"]),
                                           "first_attempt": sum(1 for r in rs if r.get("replicate", 0) == k
                                                                and r["repaired"] and r["iterations"] == 2)}
                                  for k in reps}
        if len(reps) > 1:  # non-determinism across replicates of the same strategy
            groups: dict[str, list[dict]] = {}
            for r in rs:
                groups.setdefault(r["id"], []).append(r)
            multi = {i: g for i, g in groups.items() if len(g) > 1}
            out[c]["strategies_with_replicates"] = len(multi)
            out[c]["outcome_flips"] = sorted(i for i, g in multi.items() if len({x["repaired"] for x in g}) > 1)
            out[c]["final_code_differs"] = sum(1 for g in multi.values() if len({x["final_code"] for x in g}) > 1)
            out[c]["final_sharpe_differs"] = sum(1 for g in multi.values() if len({x["final_sharpe"] for x in g}) > 1)
    return out


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return round(xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2, 3)


def write_markdown(res: dict, path: Path = REPAIR_MD, notes: list[str] | None = None) -> None:
    S, runs = res["summary"], res["runs"]
    by = {(r["id"], r["condition"]): r for r in runs if r.get("replicate", 0) == 0}
    allreps: dict = {}
    for r in runs:
        if r.get("status") == "done":
            allreps.setdefault((r["id"], r["condition"]), []).append(r)
    n_reps = max((len(s_.get("replicates", [0])) for s_ in S.values()), default=1)
    ids = list(dict.fromkeys(r["id"] for r in runs))
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"  # noqa: E731
    money = lambda x: "n/a" if x is None else f"${x:.4f}"  # noqa: E731
    L = ["# LeakBench repair benchmark", "",
         f"Generated {res['generated_at']} · {'MOCK (offline code path, not a result)' if res['mock'] else 'live Nemotron on Nebius Token Factory'}"
         f" · Coder `{res['models']['coder']}`, Auditor `{res['models']['auditor']}` · max_iters={res['max_iters']} "
         f"(iteration 1 = the leaky code as given, so at most 2 repair attempts) · data: synthetic panel "
         f"{res['data']['n_tickers']} tickers × {res['data']['n_dates']} days, seed {res['data']['seed']}.", "",
         "Each of the 24 leaky LeakBench strategies (header stripped) is fed to `run_audit_loop` as `initial_code`, with "
         "a neutral one-line hypothesis that states the intent but not the bug. **Repaired** = the loop's own "
         "stopping rule fired (no critical/high deterministic test fails). **A** = full system (Auditor sees "
         "the dynamic leak localization in the test JSON plus the 'Leak enters at line N' hint). **B** = same, with "
         "the localization removed from the Auditor prompt.", "", "## Headline", "",
         "| Condition | Repaired | Repaired & non-trivial | Repaired on 1st attempt | Mean iterations (repaired) | Over-corrected | Cost per repair | Total cost | 1st diagnosis names the leak line |",
         "|---|---|---|---|---|---|---|---|---|"]
    names = {"A": "A: full (with localization)", "B": "B: ablation (no localization)"}
    for c, s in S.items():
        L.append(f"| {names[c]} | {pct(s['repaired'], s['n'])} | {pct(s['repaired_non_trivial'], s['n'])} | "
                 f"{pct(s['repaired_first_attempt'], s['n'])} | {s['mean_iterations_repaired']} | {s['over_corrected']} | "
                 f"{money(s['cost_per_repair_usd'])} | {money(s['total_cost_usd'])} | "
                 f"{pct(s['diag1_mentions_localized_line'], s['diag1_checked'])} |")
    if n_reps > 1:
        L += ["", f"Pooled over {n_reps} independent replicates per strategy (same prompts sampled again at the "
                  "configured temperatures; replicate 0 = first run). Per replicate:", ""]
        for c, s in S.items():
            L.append(f"- {names[c]}: " + "; ".join(
                f"rep {k}: repaired {v['repaired']}/{v['n']}, first attempt {v['first_attempt']}/{v['n']}"
                for k, v in s["by_replicate"].items())
                + f". Outcome flips across replicates: {len(s.get('outcome_flips', []))}"
                + (f" ({', '.join('`' + x + '`' for x in s['outcome_flips'])})" if s.get("outcome_flips") else "")
                + f"; final code differs across replicates for {s.get('final_code_differs')}/{s.get('strategies_with_replicates')} "
                  f"strategies, final Sharpe for {s.get('final_sharpe_differs')}.")
    L += ["", f"Median Sharpe: leaky draft {S.get('A', S.get('B', {})).get('median_leaky_sharpe')} → repaired "
              + ", ".join(f"{c}: {s['median_final_sharpe_repaired']}" for c, s in S.items()) + ".",
          f"Live spend logged in `bench/repair_spend.jsonl`: {money(res['spend_logged_usd'])} (cached reruns cost nothing). "
          "Cost per run is the list-price estimate from `agents/config.py` token counts.", ""]
    skipped = [r for r in runs if r.get("status") != "done"]
    if skipped:
        L += [f"Runs not completed: " + ", ".join(f"`{r['id']}`/{r['condition']} ({r['status']})" for r in skipped), ""]
    L += ["## Per strategy", "",
          "Iterations counts the leaky draft as iteration 1. Sharpe = after 10 bps costs, t+1 open execution. "
          "'Line?' = the first Auditor diagnosis names the line the localizer found.", "",
          "Columns per condition are replicate 0; 'reps' = repaired across all replicates.", "",
          "| Strategy | Bug | Leaky Sharpe | A: result | A: iters | A: final Sharpe | A: line? | A: reps | B: result | B: iters | B: final Sharpe | B: line? | B: reps |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(r):
        if r is None:
            return ["–"] * 4
        if r.get("status") != "done":
            return [r["status"], "–", "–", "–"]
        res_ = "✅ clean" if r["repaired"] else ("❌ " + (",".join(r["final_blocking"]) or r["final_status"]))
        if r["over_corrected"]:
            res_ += " ⚠ trivial"
        ln = {True: "yes", False: "no", None: "–"}[r.get("diag1_mentions_localized_line")]
        return [res_, str(r["iterations"]), str(r["final_sharpe"]), ln]

    for i in ids:
        a, b = by.get((i, "A")), by.get((i, "B"))
        base = a or b
        lk = base.get("leaky_sharpe") if base else None
        def reps(c):
            g = allreps.get((i, c), [])
            return f"{sum(1 for x in g if x['repaired'])}/{len(g)}" if g else "–"
        L.append(f"| `{i}` | {base.get('bug', '')} | {lk} | " + " | ".join(cell(a)) + f" | {reps('A')} | "
                 + " | ".join(cell(b)) + f" | {reps('B')} |")
    done_runs = [r for r in runs if r.get("status") == "done"]
    fails = [r for r in done_runs if not r["repaired"]]
    crashed = [r for r in done_runs if r["errors"]]
    timeouts = sum(1 for r in done_runs for e in r["errors"] if "Timeout" in e)
    L += ["", "## Failure modes (all replicates)", "",
          f"- Not repaired: {len(fails)} of {len(done_runs)} runs; "
          f"{sum(1 for r in fails if r['final_status'] == 'error')} of them ended because the Coder's last rewrite "
          f"crashed or timed out in the sandbox, {sum(1 for r in fails if r['final_status'] == 'not_clean')} with a "
          f"blocking test still failing.",
          f"- Runs where at least one Coder rewrite crashed in the sandbox: {len(crashed)} "
          f"({timeouts} sandbox timeouts at 90 s, the rest pandas/numpy API errors). The loop feeds the traceback "
          "back to the Coder without an Auditor call.",
          f"- Over-corrected (repaired by going to cash: `non_trivial` failing or mean gross exposure < "
          f"{TRIVIAL_GROSS:.0%}): {sum(1 for r in done_runs if r['over_corrected'])}.", ""]
    for r in fails:
        last_err = r["errors"][-1][:160] if r["errors"] else ", ".join(r["final_blocking"])
        L.append(f"  - `{r['id']}` {r['condition']} rep {r.get('replicate', 0)}: {' → '.join(r['statuses'])}; last: {last_err}")
    L += ["", "## Caveats", "",
          "- n = 24 strategies per condition per replicate; one strategy is ~4 points of repair rate. Differences of "
          "one or two runs between A and B are within sampling noise.",
          "- Ceiling effect: every LeakBench bug is a single statement in a ~10-20 line function, the leak also shows "
          "in the static scan or the perturbation evidence, and code comments remain in the stripped file. The "
          "Auditor finds the line without the localization hint most of the time, so this benchmark can only show "
          "the hint's effect on the margin (wrong line numbers, misattributed causes), not on easy-vs-impossible.",
          "- 'Repaired' is the loop's own deterministic criterion (no critical/high test fails). It does not check "
          "that the rewrite still implements the hypothesis faithfully; the Coder sometimes changes parameters "
          "(e.g. a trailing 252-day window instead of the full history). Over-correction is only checked as "
          "going (almost) to cash.",
          "- The LLM calls are sampled (Coder temperature 0.2, Auditor 0.1, reasoning on), so reruns differ; "
          "replicates > 0 sample the same prompts again. Cached replays are exact.",
          "- Hypotheses are hand-written one-liners that state the intent without the bug (see `HYPOTHESES` in "
          "`bench/run_repair.py`); the strategies' own docstrings and comments are still in the code.",
          "- The Summarizer is skipped (`summarize=False`) to save cost; it does not affect the loop's outcome.",
          "- Costs are list-price estimates from token counts (`agents/config.py`), not invoices."]
    if notes:
        L += ["", "## Observations from this run", ""] + [f"- {n}" for n in notes]
    L += ["", "## Reproduce", "", "```bash",
          "NEBIUS_API_KEY=... python -m bench.run_repair --replicates 0 1 2   # responses cached in bench/repair_cache.json",
          "python -m bench.run_repair --report-only [--notes-file notes.md]   # rebuild this file from repair_results.json",
          "```", ""]
    path.write_text("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", default=None, help="substring filter on strategy ids")
    ap.add_argument("--conditions", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--mock", action="store_true", help="MockLLM (offline); writes nothing unless --out is given")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="hard cap on total logged spend (USD)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--replicates", nargs="+", type=int, default=[0],
                    help="replicate indices; >0 samples the same prompts again (separate cache keys)")
    ap.add_argument("--rerun", action="store_true", help="rerun runs already in the results file (cache still applies)")
    ap.add_argument("--report-only", action="store_true", help="rebuild REPAIR.md from repair_results.json")
    ap.add_argument("--notes-file", default=None, help="markdown bullets to embed as caveats")
    ap.add_argument("--out", default=None, help="results json path (default bench/repair_results.json)")
    a = ap.parse_args(argv)
    notes = []
    if a.notes_file and Path(a.notes_file).exists():
        notes = [ln.lstrip("- ").rstrip() for ln in Path(a.notes_file).read_text().splitlines() if ln.strip()]
    out = Path(a.out) if a.out else (None if a.mock else RESULTS_JSON)
    if a.report_only:
        res = load_json(RESULTS_JSON, None)
        if not res:
            print("no results file")
            return 1
        res["summary"] = summarize(res["runs"])
        write_markdown(res, notes=notes)
        print(f"wrote {REPAIR_MD.relative_to(ROOT)}")
        return 0
    res = run_repair(only=a.only, conditions=tuple(a.conditions), mock=a.mock, budget=a.budget, workers=a.workers,
                     max_iters=a.max_iters, results_path=out, rerun=a.rerun,
                     replicates=tuple(a.replicates))
    print(json.dumps(res["summary"], indent=1))
    print(f"spend this invocation ${res['spend_this_invocation_usd']:.4f}; logged total ${res['spend_logged_usd']:.4f}")
    if out == RESULTS_JSON:
        write_markdown(res, notes=notes)
        print(f"wrote {RESULTS_JSON.relative_to(ROOT)} and {REPAIR_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

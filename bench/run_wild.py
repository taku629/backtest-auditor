"""In-the-wild study: how often do Nemotron-written backtests leak when nobody asks them to?

For each of the 30 ordinary strategy requests in ``bench/wild_prompts.py`` a Nemotron model writes a first draft of
``generate_signals(data)`` (one call, no repair loop). Every draft then goes through the same deterministic pipeline
as the app: sandboxed audit on the seeded synthetic panel (future-perturbation, survivorship, execution delay, static
scan, plausibility, DSR, train/test, PBO), the dynamic leak localizer, and the LLM-free Quick fix.

Conditions (model x system prompt):

* models: ``super`` = ``nvidia/nemotron-3-super-120b-a12b`` (the app's Coder), ``lightning`` =
  ``nvidia/Nemotron-3_5-Lightning``.
* system prompts: ``app`` = the app's Coder system prompt (``agents.prompts.CODER_SYSTEM``, which states that row t
  is a decision after the close of t and is executed at the next bar); ``chat`` = a minimal request a user would type
  into a chat assistant (same function contract, no timing guidance).

Every response is cached in ``bench/wild_cache.json`` (LLM outputs, keyed by prompt hash) and every fresh paid call is
logged in ``bench/wild_spend.jsonl``. A hard budget (``--budget``, default $1.00, counted over the whole spend log)
stops new paid calls. Reruns with a full cache cost nothing.

Usage::

    NEBIUS_API_KEY=... python -m bench.run_wild                       # all conditions
    NEBIUS_API_KEY=... python -m bench.run_wild --only W01 W02 --models super --systems app   # pilot
    python -m bench.run_wild --offline                                # cache only, no API calls
    python -m bench.run_wild --report-only                            # rebuild WILD.md from wild_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_repair import BudgetExceeded, CachedBudgetLLM, load_json, logged_spend  # noqa: E402
from bench.wild_prompts import WILD_PROMPTS  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
CACHE_PATH = BENCH_DIR / "wild_cache.json"
SPEND_LOG = BENCH_DIR / "wild_spend.jsonl"
RESULTS_JSON = BENCH_DIR / "wild_results.json"
WILD_MD = BENCH_DIR / "WILD.md"
PUBLIC_JSON = BENCH_DIR / "wild_public.json"   # slim copy the app reads (no code, no diffs)
DEFAULT_BUDGET = 1.00

MODELS = {"super": "nvidia/nemotron-3-super-120b-a12b", "lightning": "nvidia/Nemotron-3_5-Lightning"}
SYSTEMS = ("app", "chat")

CHAT_SYSTEM = """Write a Python function `generate_signals(data)` for a stock backtest.
`data` is a dict of pandas DataFrames with keys "open", "high", "low", "close", "volume";
index = dates, columns = tickers, NaN where a ticker has no data.
Return a DataFrame of portfolio weights with the same index and columns.
Use only numpy, pandas and math. Reply with one ```python code block."""

# test name -> bug class reported in the study
LEAK_CLASSES = {
    "future_perturbation": "look_ahead",
    "survivorship": "survivorship",
    "execution_delay": "same_bar_execution",
    "static_scan": "static_red_flag",
    "plausibility": "implausible_sharpe",
    "non_trivial": "never_trades",
}
TRIVIAL_GROSS = 0.05


def classify(r: dict) -> list[str]:
    """Bug classes of one audited draft, from its blocking tests (the survivorship test is split by its reason)."""
    out = set()
    summ = r.get("test_summaries") or {}
    for n in r.get("blocking") or []:
        if n == "survivorship":
            txt = summ.get(n, "")
            if "before they were listed" in txt:
                out.add("pre_listing_weight")
            if "made to survive" in txt:
                out.add("survivor_universe")
            if not ({"pre_listing_weight", "survivor_universe"} & out):
                out.add("survivorship")
        elif n in LEAK_CLASSES and n != "non_trivial":
            out.add(LEAK_CLASSES[n])
    return sorted(out)


class OfflineLLM:
    """Stand-in for --offline: every call must be a cache hit (the wrapper answers it before reaching here)."""

    is_mock = False

    def __init__(self, config):
        self.config = config

    def chat(self, role, messages, context=None):
        raise BudgetExceeded("offline: response not in bench/wild_cache.json")


def make_config():
    """The repo's routing plus one coder route per studied model (same sampling settings as the app's Coder)."""
    from dataclasses import replace

    from agents.config import load_config

    cfg = load_config()
    base = cfg.route("coder")
    for name, model in MODELS.items():
        # Lightning with reasoning on spent its whole 8k budget thinking in the pilot (no code came back), so it runs
        # with thinking off, as the app runs it for the Summarizer.
        cfg.routes[f"coder_{name}"] = replace(base, role=f"coder_{name}", model=model,
                                              thinking=(name != "lightning"))
    return cfg


def messages_for(prompt: dict, system: str) -> list[dict]:
    from agents import prompts as P

    if system == "app":
        return [{"role": "system", "content": P.CODER_SYSTEM},
                {"role": "user", "content": P.CODER_USER.format(iteration=1, hypothesis=prompt["text"],
                                                                feedback_block="")}]
    return [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": prompt["text"]}]


def run_key(prompt_id: str, model: str, system: str, replicate: int = 0) -> str:
    return f"{prompt_id}|{model}|{system}" + (f"|r{replicate}" if replicate else "")


# ----------------------------------------------------------------------------
# Phase 1: drafts (LLM, cached)
# ----------------------------------------------------------------------------
RETRY_USER = """Running your function in the backtest raised this error:
{error}
Return the corrected complete generate_signals function in one ```python code block."""


def draft(prompt: dict, model: str, system: str, llm: CachedBudgetLLM, replicate: int = 0,
          retry_of: dict | None = None) -> dict:
    """One Coder call. ``retry_of`` = the crashed first draft: the error is fed back once (as the app's loop does)."""
    from agents.llm import extract_code

    llm.tag.strategy, llm.tag.replicate = prompt["id"], replicate
    llm.tag.condition = f"{model}/{system}" + ("/retry" if retry_of else "")
    llm.tag.fresh, llm.tag.cached = 0.0, 0
    msgs = messages_for(prompt, system)
    if retry_of:
        msgs = msgs + [{"role": "assistant", "content": "```python\n" + retry_of["code"] + "\n```"},
                       {"role": "user", "content": RETRY_USER.format(error=retry_of["error"][-1500:])}]
    try:
        resp = llm.chat(f"coder_{model}", msgs)
    except BudgetExceeded as e:
        return {"status": "skipped_budget", "error": str(e)}
    except Exception as e:  # noqa: BLE001 - an API failure is recorded, never hidden
        return {"status": "api_error", "error": f"{type(e).__name__}: {e}"[:400]}
    from agents.config import estimate_cost
    return {"status": "drafted", "code": extract_code(resp.text), "model_id": resp.model,
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
            "cost_usd": round(estimate_cost(resp.model, resp.prompt_tokens, resp.completion_tokens) or 0.0, 6),
            "fresh_cost_usd": round(llm.tag.fresh, 6), "cached": bool(llm.tag.cached)}


# ----------------------------------------------------------------------------
# Phase 2: deterministic audit + localizer + Quick fix (no LLM)
# ----------------------------------------------------------------------------
def _sh(m: dict | None) -> float | None:
    return None if not m or "sharpe" not in m else round(float(m["sharpe"]), 3)


def audit_draft(code: str, data: dict, *, timeout: float = 90.0, pbo: bool = True) -> dict:
    from audit.leak_tests import run_audit
    from audit.quickfix import quick_fix
    from sandbox.runner import SandboxError, SandboxExecutor, check_policy

    t0 = time.time()
    v = check_policy(code)
    if v:
        return {"audit": "policy_violation", "error": "; ".join(v)[:400], "seconds": round(time.time() - t0, 1)}
    try:
        rep = run_audit(data, SandboxExecutor(code, timeout), code, pbo=pbo, pbo_timeout=timeout)
    except (SandboxError, RuntimeError, ValueError) as e:
        return {"audit": "crashed", "error": f"{type(e).__name__}: {e}"[-400:], "seconds": round(time.time() - t0, 1)}
    rd = rep.to_dict()
    tests = rd["tests"]
    by = {t["name"]: t for t in tests}
    blocking = [t["name"] for t in tests if t.get("blocking")]
    warnings = [t["name"] for t in tests if not t["passed"] and not t.get("blocking")]
    m = rd.get("metrics") or {}
    fp = by.get("future_perturbation") or {}
    loc = (fp.get("evidence") or {}).get("localized") or {}
    entry = loc.get("entry") or {}
    st = by.get("static_scan") or {}
    findings = (st.get("evidence") or {}).get("findings") or []
    pbo_t = by.get("pbo") or {}
    pbo_v = (pbo_t.get("evidence") or {}).get("pbo")
    gross = m.get("avg_gross")
    trivial = "non_trivial" in blocking or (gross is not None and float(gross) < TRIVIAL_GROSS)
    out = {
        "audit": "ok", "clean": not blocking, "blocking": blocking, "warnings": warnings,
        "never_trades": "non_trivial" in blocking, "trivial_exposure": bool(trivial),
        "sharpe": _sh(m), "cagr": None if not m else round(float(m.get("cagr", 0.0)), 4),
        "avg_gross": None if gross is None else round(float(gross), 4),
        "first_leak_date": (fp.get("evidence") or {}).get("first_leaking_date"),
        "localized_found": bool(loc.get("found")), "localized_line": entry.get("line"),
        "localized_source": entry.get("source_line"),
        "static_findings": [{k: f.get(k) for k in ("rule", "line", "severity", "snippet") if k in f}
                            for f in findings][:6],
        "pbo": None if pbo_v is None else round(float(pbo_v), 3), "pbo_failed": pbo_t.get("passed") is False,
        "dsr_failed": (by.get("deflated_sharpe") or {}).get("passed") is False,
        "test_summaries": {t["name"]: t["summary"][:240] for t in tests if not t["passed"]},
    }
    if blocking and not out["never_trades"]:
        qf = quick_fix(code, data, report=rep, timeout=timeout)
        out["quickfix"] = {"status": qf["status"], "rule": qf.get("rule"), "description": qf.get("description"),
                           "lines": qf.get("lines"), "diff": (qf.get("diff") or "")[:3000],
                           "after_sharpe": (qf.get("after") or {}).get("sharpe"),
                           "n_candidates": qf.get("n_candidates")}
    out["leak_classes"] = classify(out)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_wild(*, only: list[str] | None = None, models: tuple[str, ...] = tuple(MODELS),
             systems: tuple[str, ...] = SYSTEMS, budget: float = DEFAULT_BUDGET, offline: bool = False,
             workers: int = 4, data: dict | None = None, llm=None,
             results_path: Path | None = RESULTS_JSON, cache_path: Path = CACHE_PATH, spend_path: Path = SPEND_LOG,
             reaudit: bool = False, pbo: bool = True, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    cfg = make_config()
    if llm is None:
        if offline:
            llm = OfflineLLM(cfg)
        else:
            from agents.llm import NebiusLLM
            llm = NebiusLLM(cfg)
    wrapped = CachedBudgetLLM(llm, cache_path=cache_path, spend_path=spend_path, budget=budget)
    if data is None:
        from data import generate_panel
        data = generate_panel(seed=51)
    prompts = [p for p in WILD_PROMPTS if not only or p["id"] in only]
    jobs = [(p, m, s) for m in models for s in systems for p in prompts]
    prev = load_json(results_path, {}) if results_path else {}
    runs = dict(prev.get("runs", {}))

    def one(job):
        p, m, s = job
        key = run_key(p["id"], m, s)
        old = runs.get(key)
        d = draft(p, m, s, wrapped)
        rec = {"key": key, "prompt_id": p["id"], "family": p["family"], "style": p["style"], "model": m,
               "model_id": MODELS[m], "system": s} | d
        if d["status"] != "drafted":
            return key, (old if old and old.get("status") == "drafted" else rec)
        if old and not reaudit and old.get("first_code", old.get("code")) == d["code"] and old.get("audit"):
            return key, old   # same cached draft: the audit is deterministic, keep the stored result
        first = audit_draft(d["code"], data, pbo=pbo)
        rec |= first
        if first.get("audit") in ("crashed", "policy_violation"):
            # one free retry with the error, like the app's loop; the study reports both drafts
            rec["first_code"], rec["first_error"], rec["first_audit"] = d["code"], first.get("error"), first["audit"]
            d2 = draft(p, m, s, wrapped, retry_of={"code": d["code"], "error": first.get("error") or ""})
            rec["retried"] = True
            if d2["status"] == "drafted":
                rec["cost_usd"] = round((d.get("cost_usd") or 0) + (d2.get("cost_usd") or 0), 6)
                rec["code"] = d2["code"]
                for k in ("error", "audit"):
                    rec.pop(k, None)
                rec |= audit_draft(d2["code"], data, pbo=pbo)
            else:
                rec["retry_status"] = d2["status"]
        log(f"{key}: {rec.get('audit')}{' (retry)' if rec.get('retried') else ''} blocking={rec.get('blocking')} "
            f"qf={(rec.get('quickfix') or {}).get('status')} ({rec.get('seconds')}s)")
        return key, rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, rec in ex.map(one, jobs):
            runs[key] = rec
    log(f"spend log ${logged_spend(spend_path):.4f}")
    res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data": "synthetic panel, 50 tickers x 1512 days, seed 51 (8 delisted, 6 IPOs)",
           "models": MODELS, "spend_usd": round(logged_spend(spend_path), 4), "runs": runs}
    res["summary"] = summarize(res)
    if results_path:
        results_path.write_text(json.dumps(res, indent=1, sort_keys=True))
    return res


# ----------------------------------------------------------------------------
# Follow-ups on leaky drafts: re-run Quick fix with the current rules; full agentic repair loop
# ----------------------------------------------------------------------------
def leaky_runs(res: dict) -> list[dict]:
    return [r for k, r in sorted(res["runs"].items())
            if "|r" not in k and r.get("audit") == "ok" and r.get("blocking") and not r.get("never_trades")]


def requickfix(res: dict, data: dict, *, verbose: bool = True) -> dict:
    """Re-audit every leaky draft with the current Quick fix rules; the first result is kept as ``quickfix_base``."""
    for r in leaky_runs(res):
        if "quickfix_base" not in r:
            r["quickfix_base"] = r.get("quickfix") or {"status": "not_run"}
        new = audit_draft(r["code"], data, pbo=False)
        for k in ("quickfix", "leak_classes", "blocking", "localized_line", "localized_source", "localized_found"):
            if k in new:
                r[k] = new[k]
        if verbose:
            print(f"{r['key']}: quick fix {r['quickfix_base'].get('status')} -> {(r.get('quickfix') or {}).get('status')}")
    return res


def repair_with_loop(res: dict, data: dict, llm, *, max_iters: int = 3, verbose: bool = True) -> dict:
    """Leaky drafts that Quick fix could not repair go through the app's full loop (Quick fix -> Nemotron Ultra
    Auditor -> Nemotron Super Coder), starting from the draft as user code. Summarizer and PBO off to save cost."""
    from agents.loop import run_audit_loop

    prompts = {p["id"]: p for p in WILD_PROMPTS}
    for r in leaky_runs(res):
        if (r.get("quickfix") or {}).get("status") == "fixed" or (r.get("loop") or {}).get("status") == "done":
            continue
        llm.tag.strategy, llm.tag.condition, llm.tag.replicate = r["prompt_id"], f"{r['model']}/{r['system']}/loop", 0
        llm.tag.fresh, llm.tag.cached = 0.0, 0
        try:
            tr = run_audit_loop(prompts[r["prompt_id"]]["text"], data, llm, max_iters=max_iters, summarize=False,
                                initial_code=r["code"], quick_fix=True, pbo=False)
        except BudgetExceeded as e:
            r["loop"] = {"status": "skipped_budget", "error": str(e)}
            break
        except Exception as e:  # noqa: BLE001
            r["loop"] = {"status": "run_error", "error": f"{type(e).__name__}: {e}"[:300]}
            continue
        last = tr.iterations[-1]
        gross = (last.metrics or {}).get("avg_gross")
        fails = [t["name"] for t in last.tests if not t["passed"]]
        over = bool(last.status != "error" and ("non_trivial" in fails or (gross is not None and gross < TRIVIAL_GROSS)))
        r["loop"] = {"status": "done", "final_status": tr.final_status,
                     "repaired": tr.final_status == "clean" and not over, "over_corrected": over,
                     "iterations": len(tr.iterations), "statuses": [it.status for it in tr.iterations],
                     "sources": [it.source for it in tr.iterations],
                     "final_sharpe": _sh(last.metrics), "final_code": last.code,
                     "errors": [it.error[-200:] for it in tr.iterations if it.error],
                     "cost_usd": round(sum(c.get("cost_usd") or 0.0 for c in tr.calls), 6),
                     "fresh_cost_usd": round(llm.tag.fresh, 6), "n_calls": len(tr.calls)}
        if verbose:
            print(f"{r['key']}: loop {tr.final_status} after {len(tr.iterations)} iterations "
                  f"(${r['loop']['cost_usd']:.4f}, spend log ${llm.spent:.4f})")
    return res


# ----------------------------------------------------------------------------
# Summary + report
# ----------------------------------------------------------------------------
def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def summarize_group(rs: list[dict]) -> dict:
    drafted = [r for r in rs if r.get("status") == "drafted"]
    ran = [r for r in drafted if r.get("audit") == "ok"]
    crashed = [r for r in drafted if r.get("audit") in ("crashed", "policy_violation")]
    leaky = [r for r in ran if r.get("blocking") and not r.get("never_trades")]
    never = [r for r in ran if r.get("never_trades")]
    by_class: dict[str, int] = {}
    for r in leaky:
        r["leak_classes"] = classify(r)
        for c in r["leak_classes"]:
            by_class[c] = by_class.get(c, 0) + 1
    la = [r for r in leaky if "look_ahead" in (r.get("leak_classes") or [])]
    fixed = [r for r in leaky if (r.get("quickfix") or {}).get("status") == "fixed"]
    fixed_base = [r for r in leaky if ((r.get("quickfix_base") or r.get("quickfix")) or {}).get("status") == "fixed"]
    looped = [r for r in leaky if (r.get("loop") or {}).get("status") == "done"]
    loop_ok = [r for r in looped if r["loop"].get("repaired")]
    repaired_any = fixed + loop_ok
    sys_pairs = [(r["sharpe"], (r["quickfix"]["after_sharpe"] if r in fixed else r["loop"]["final_sharpe"]))
                 for r in repaired_any if r.get("sharpe") is not None]
    pairs = [(r["sharpe"], r["quickfix"]["after_sharpe"]) for r in fixed
             if r.get("sharpe") is not None and r["quickfix"].get("after_sharpe") is not None]
    return {
        "n": len(rs), "drafted": len(drafted), "ran": len(ran), "crashed": len(crashed), "never_trades": len(never),
        "leaky": len(leaky), "clean": len([r for r in ran if r.get("clean")]),
        "by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "look_ahead": len(la), "look_ahead_localized": len([r for r in la if r.get("localized_found")]),
        "quickfix_fixed": len(fixed), "quickfix_fixed_base": len(fixed_base),
        "loop_tried": len(looped), "loop_repaired": len(loop_ok),
        "loop_cost_usd": round(sum(r["loop"].get("cost_usd") or 0.0 for r in looped), 4),
        "system_repaired": len(repaired_any),
        "median_sharpe_leaky_all": _median([r.get("sharpe") for r in leaky]),
        "median_sharpe_leaky_repaired": _median([p[0] for p in sys_pairs if p[1] is not None]),
        "median_sharpe_after_repair": _median([p[1] for p in sys_pairs]),
        "median_sharpe_leaky": _median([p[0] for p in pairs]), "median_sharpe_fixed": _median([p[1] for p in pairs]),
        "median_sharpe_clean": _median([r.get("sharpe") for r in ran if r.get("clean")]),
        "pbo_warn": len([r for r in ran if r.get("pbo_failed")]),
        "pbo_measured": len([r for r in ran if r.get("pbo") is not None]),
        "cost_usd": round(sum(r.get("cost_usd") or 0.0 for r in drafted), 4),
    }


def summarize(res: dict) -> dict:
    runs = [r for k, r in res["runs"].items() if "|r" not in k]
    groups = {"all": runs}
    for m in MODELS:
        for s in SYSTEMS:
            g = [r for r in runs if r["model"] == m and r["system"] == s]
            if g:
                groups[f"{m}/{s}"] = g
    for st in ("plain", "notebook"):
        groups[f"style:{st}"] = [r for r in runs if r.get("style") == st]
    return {k: summarize_group(v) for k, v in groups.items() if v}


def _cond_label(k: str) -> str:
    if k == "all":
        return "**All drafts**"
    if k.startswith("style:"):
        return f"All models, {k.split(':')[1]}-style prompts"
    m, s = k.split("/")
    return f"{'Super' if m == 'super' else 'Lightning'}, {'app Coder prompt' if s == 'app' else 'chat prompt'}"


def _loop_cell(r: dict) -> str:
    lp = r.get("loop") or {}
    if lp.get("status") != "done":
        return "—" if not lp else lp.get("status", "—")
    tag = "✅ clean" if lp.get("repaired") else ("clean, but < 5% avg gross exposure (not counted)" if lp.get("over_corrected") else f"❌ {lp.get('final_status')}")
    return f"{tag}, {lp.get('iterations')} iters, Sharpe {lp.get('final_sharpe')}, ${lp.get('cost_usd', 0):.4f}"


def write_report(res: dict, path: Path = WILD_MD) -> str:
    S = res.get("summary") or summarize(res)
    runs = res["runs"]
    a = S["all"]
    lines = [
        "# In the wild: do Nemotron-written backtests leak when nobody asks them to?",
        "",
        f"Generated {res['generated_at'][:16].replace('T', ' ')} UTC · live Nemotron on Nebius Token Factory · "
        f"{', '.join('`' + v + '`' for v in res['models'].values())} · {res['data']}.",
        "",
        "30 ordinary strategy requests (`bench/wild_prompts.py`) were each sent once to each model under two system "
        "prompts: the app's own Coder prompt (`app`, which says row t is a decision after the close of t, executed at "
        "the next bar) and a minimal request a user would type into a chat assistant (`chat`, same function contract, "
        "no timing guidance). Each first draft (no repair loop, no retries) was run through the deterministic audit, "
        "the leak localizer and the LLM-free Quick fix exactly as the app does. **Leaky** = at least one blocking "
        "(critical/high) test fails among drafts that run and trade.",
        "",
        "## Headline",
        "",
        f"- **{_pct(a['leaky'], a['ran'])} of runnable drafts leak** ({a['ran']} of {a['drafted']} drafts ran; "
        f"{a['crashed']} crashed in the sandbox, {a['never_trades']} never took a position).",
        f"- Look-ahead (future-perturbation test failed): {a['look_ahead']} drafts; the localizer reported a line in "
        f"{a['look_ahead_localized']}/{a['look_ahead']} (see the manual review below for how many are the root-cause line).",
        f"- Quick fix (no LLM, $0) repaired {_pct(a['quickfix_fixed'], a['leaky'])} of leaky drafts with the current rules "
        f"({a['quickfix_fixed_base']}/{a['leaky']} with the rules at the start of the study; the added rules were written "
        "after seeing these drafts, so the higher number is in-sample).",
        f"- The full agentic loop (Quick fix → Nemotron Ultra Auditor → Nemotron Super Coder, at most 2 repair attempts) "
        f"repaired {a['loop_repaired']}/{a['loop_tried']} of the drafts Quick fix could not "
        f"(${a['loop_cost_usd']:.3f} in total), so the system repaired {_pct(a['system_repaired'], a['leaky'])} of leaky drafts.",
        f"- Leaks were mostly not spectacular: median Sharpe of the leaky drafts {a['median_sharpe_leaky_all']} vs "
        f"{a['median_sharpe_clean']} for clean drafts; after repair, the repaired ones moved from a median "
        f"{a['median_sharpe_leaky_repaired']} to {a['median_sharpe_after_repair']}. The equity curve would not have "
        "given these bugs away.",
        f"- PBO above 0.5 (overfitting warning, non-blocking): {a['pbo_warn']} of {a['pbo_measured']} drafts where PBO "
        "could be measured.",
        f"- Live spend for the whole study: ${res.get('spend_usd', 0):.4f} (first drafts and crash retries "
        f"${a['cost_usd']:.4f}, repair loop ${a['loop_cost_usd']:.4f}, the rest pilot calls; logged call by call in "
        "`bench/wild_spend.jsonl`; cached reruns are free).",
        "",
        "## By condition",
        "",
        "| Condition | Drafts | Ran | Crashed | Leaky (of ran) | Look-ahead | Survivor universe | Weight before listing | Line localized | "
        "Quick fix repaired | + loop repaired | Median Sharpe leaky → fixed (Quick fix) | PBO > 0.5 | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, g in S.items():
        lines.append(
            f"| {_cond_label(k)} | {g['drafted']} | {g['ran']} | {g['crashed']} | {_pct(g['leaky'], g['ran'])} | "
            f"{g['look_ahead']} | {g['by_class'].get('survivor_universe', 0)} | {g['by_class'].get('pre_listing_weight', 0)} | {g['look_ahead_localized']}/{g['look_ahead']} | "
            f"{g['quickfix_fixed']}/{g['leaky']} | {g['loop_repaired']}/{g['loop_tried']} | {g['median_sharpe_leaky']} → {g['median_sharpe_fixed']} | "
            f"{g['pbo_warn']}/{g['pbo_measured']} | ${g['cost_usd']:.4f} |")
    lines += ["", "Blocking failures by test, all drafts: " +
              (", ".join(f"{c} {n}" for c, n in a["by_class"].items()) or "none") +
              " (a draft can fail several).", ""]

    leaky_rows = [r for r in runs.values() if r.get("audit") == "ok" and r.get("blocking") and not r.get("never_trades")]
    lines += ["## Every leaky draft", "",
              "| Draft | Request | Failing blocking tests | Leak line (localizer) | Sharpe | Quick fix | Sharpe after | Loop (if Quick fix failed) |",
              "|---|---|---|---|---|---|---|---|"]
    prompts = {p["id"]: p for p in WILD_PROMPTS}
    for r in sorted(leaky_rows, key=lambda r: r["key"]):
        qf = r.get("quickfix") or {}
        src = (r.get("localized_source") or "").strip().replace("|", "\\|")
        loc = f"L{r['localized_line']}: `{src[:70]}`" if r.get("localized_found") else "—"
        lines.append(f"| `{r['key']}` | {prompts[r['prompt_id']]['text'][:70]}… | {', '.join(r['blocking'])} | {loc} | "
                     f"{r.get('sharpe')} | {qf.get('status', '—')}{(' (' + str(qf.get('rule')) + ')') if qf.get('rule') else ''} | "
                     f"{qf.get('after_sharpe', '—')} | {_loop_cell(r)} |")
    crashed = [r for r in runs.values() if r.get("audit") in ("crashed", "policy_violation")]
    if crashed:
        lines += ["", "## Drafts that did not run", ""]
        for r in sorted(crashed, key=lambda r: r["key"]):
            err = (r.get("error") or "").strip().splitlines()
            lines.append(f"- `{r['key']}` ({r['audit']}): {(err[-1] if err else '')[:200]}")
    notes = res.get("notes")
    if notes:
        lines += ["", "## Observations", "", notes.strip()]
    lines += [
        "",
        "## Caveats",
        "",
        "- Small sample: 30 prompts × 2 models × 2 system prompts, one sample each (temperature 0.2; reasoning on for "
        "Super, off for Lightning, which spent its whole 8k-token budget thinking in a pilot and returned no code). "
        "One prompt is ~3 points of a condition's leak rate; treat differences between conditions as indicative only.",
        "- The prompts were written by us, not collected from real users. A third of them (`notebook` style) mention a "
        "notebook habit (fill the gaps, drop tickers with missing data, z-score, `qcut`, pick the most correlated pair, "
        "pick the best parameters) that is easy to implement with full-sample information; they are ordinary requests, "
        "but they are not a random sample of what people ask. Results are reported per style.",
        "- 'Leaky' is the auditor's own deterministic verdict on the bundled synthetic panel. The dynamic tests have no "
        "false alarms on LeakBench's 15 clean strategies, but LeakBench is also ours.",
        "- Survivorship failures here include drafts that put weight on a ticker before it lists or that change past "
        "weights when delisted names keep trading; on real data with delistings the effect size can differ.",
        "- The first draft only: in the app, the loop would go on to repair the draft (Quick fix, then Auditor → Coder).",
        "- Costs are list-price estimates from token counts (`agents/config.py`), not invoices.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "NEBIUS_API_KEY=... python -m bench.run_wild        # drafts cached in bench/wild_cache.json",
        "python -m bench.run_wild --offline                  # replay from cache, no API calls",
        "python -m bench.run_wild --report-only              # rebuild this file from bench/wild_results.json",
        "```",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def public_summary(res: dict | None = None) -> dict:
    """A compact view for the app (no code, no diffs): summary + leaky rows. Reads wild_results.json by default."""
    res = res if res is not None else load_json(RESULTS_JSON, {})
    if not res or not res.get("runs"):
        return {}
    prompts = {p["id"]: p for p in WILD_PROMPTS}
    rows = []
    for r in sorted(res["runs"].values(), key=lambda r: r["key"]):
        if r.get("status") != "drafted" or "|r" in r["key"]:
            continue
        qf = r.get("quickfix") or {}
        rows.append({"key": r["key"], "prompt": prompts.get(r["prompt_id"], {}).get("text", ""),
                     "model": r["model"], "system": r["system"], "style": r.get("style"),
                     "outcome": ("crashed" if r.get("audit") != "ok" else "never trades" if r.get("never_trades")
                                 else "leaky" if r.get("blocking") else "clean"),
                     "blocking": r.get("blocking") or [], "sharpe": r.get("sharpe"),
                     "leak_line": r.get("localized_line"), "leak_source": r.get("localized_source"),
                     "quickfix": qf.get("status"), "sharpe_after": qf.get("after_sharpe"),
                     "pbo": r.get("pbo"),
                     "loop": (None if (r.get("loop") or {}).get("status") != "done" else
                              ("clean" if r["loop"].get("repaired") else
                               "clean, low exposure" if r["loop"].get("over_corrected") else r["loop"].get("final_status"))),
                     "sharpe_after_loop": (r.get("loop") or {}).get("final_sharpe")})
    return {"generated_at": res.get("generated_at"), "spend_usd": res.get("spend_usd"),
            "summary": res.get("summary") or summarize(res), "rows": rows, "models": res.get("models")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="prompt ids, e.g. W01 W02")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--offline", action="store_true", help="cache only; never call the API")
    ap.add_argument("--reaudit", action="store_true", help="re-run the audit even when the code is unchanged")
    ap.add_argument("--no-pbo", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--requickfix", action="store_true", help="re-run Quick fix (current rules) on leaky drafts")
    ap.add_argument("--repair-loop", action="store_true",
                    help="run the full agentic loop on leaky drafts Quick fix did not repair (paid, cached)")
    ap.add_argument("--notes-file", help="markdown observations to embed in WILD.md")
    args = ap.parse_args(argv)
    if args.report_only:
        res = load_json(RESULTS_JSON, {})
    else:
        res = run_wild(only=args.only, models=tuple(args.models), systems=tuple(args.systems), budget=args.budget,
                       offline=args.offline, reaudit=args.reaudit, pbo=not args.no_pbo)
    if args.requickfix or args.repair_loop:
        from data import generate_panel
        data = generate_panel(seed=51)
        if args.requickfix:
            requickfix(res, data)
        if args.repair_loop:
            from agents.config import load_config
            from agents.llm import NebiusLLM
            inner = OfflineLLM(load_config()) if args.offline else NebiusLLM(load_config())
            wrapped = CachedBudgetLLM(inner, cache_path=CACHE_PATH, spend_path=SPEND_LOG, budget=args.budget)
            repair_with_loop(res, data, wrapped)
            res["spend_usd"] = round(logged_spend(SPEND_LOG), 4)
        RESULTS_JSON.write_text(json.dumps(res, indent=1, sort_keys=True))
    if args.notes_file:
        res["notes"] = Path(args.notes_file).read_text()
        RESULTS_JSON.write_text(json.dumps(res, indent=1, sort_keys=True))
    res["summary"] = summarize(res)
    write_report(res)
    PUBLIC_JSON.write_text(json.dumps(public_summary(res), indent=1, sort_keys=True))
    print(json.dumps(res["summary"]["all"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

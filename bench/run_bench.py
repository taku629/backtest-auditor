"""LeakBench: deterministic audit vs LLM-only judge on labelled strategies.

Every file in ``bench/strategies/`` follows the strategy contract
(``generate_signals(data) -> weights``) and opens with a header::

    # LEAKBENCH-LABEL: leaky|clean
    # LEAKBENCH-BUG: <bug type>
    # LEAKBENCH-NOTE: <one line>

Methods compared
----------------
* ``deterministic``: the Backtest Auditor suite (future-perturbation, survivorship
  counterfactual, execution delay, AST scan, ...) run through the sandbox exactly
  as ``agents/loop.py`` does. Flagged = at least one blocking (critical/high) test fails.
* ``llm:<model>``: the LLM-only judge. The model sees the code with the LeakBench
  header stripped, plus the engine contract, and answers ``{"leaky": bool, "reason": str}``.

Usage::

    python -m bench.run_bench --no-llm                 # offline, deterministic only
    NEBIUS_API_KEY=... python -m bench.run_bench       # + Nemotron Ultra and Super judges
    python -m bench.run_bench --models nvidia/nemotron-3-super-120b-a12b

LLM answers are cached in ``bench/llm_cache.json`` (keyed by model, prompt version
and code hash), so reruns are free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(__file__).resolve().parent
STRAT_DIR = BENCH_DIR / "strategies"
RESULTS_JSON = BENCH_DIR / "results.json"
RESULTS_MD = BENCH_DIR / "RESULTS.md"
CACHE_PATH = BENCH_DIR / "llm_cache.json"
SPEND_LOG = BENCH_DIR / "llm_spend.jsonl"   # every paid call, including unparsed ones that are not cached

DEFAULT_MODELS = ["nvidia/Nemotron-3-Ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b"]
SHORT_NAMES = {"nvidia/Nemotron-3-Ultra-550b-a55b": "Nemotron Ultra",
               "nvidia/nemotron-3-super-120b-a12b": "Nemotron Super",
               "nvidia/Nemotron-3_5-Lightning": "Nemotron Lightning"}
PROMPT_VERSION = "v1"
JUDGE_MAX_TOKENS = 3000      # caps reasoning + answer (Ultra can think for a long time)
FALLBACK_MAX_TOKENS = 400    # retry with thinking off if the first answer had no parsable JSON
HEADER_RE = re.compile(r"^#\s*LEAKBENCH-([A-Z]+):\s*(.*)$")

JUDGE_SYSTEM = ("You are a senior quantitative researcher reviewing backtest code for correctness. "
                "You answer with a single JSON object and nothing else.")
JUDGE_USER = """Review this trading strategy written for a daily equity backtester.

Engine contract:
- data["open"|"high"|"low"|"close"|"volume"] are DataFrames (rows = trading dates in ascending order, columns = tickers). NaN means the ticker is not listed on that date (IPOs start late, some tickers are delisted and stop early).
- generate_signals(data) returns target weights with the same shape. Row t is a decision made after the close of day t; the engine executes it at the next bar (t+1), so using data up to and including the close of day t in row t is legitimate.

Question: Does this backtest strategy code use future information (look-ahead bias) or have survivorship bias?

```python
{code}
```

Answer with JSON only: {{"leaky": true or false, "reason": "<one or two sentences>"}}"""


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------
def strip_header(code: str) -> str:
    """Remove the LeakBench label lines so the judge sees only the strategy."""
    lines = code.splitlines()
    i = 0
    while i < len(lines) and HEADER_RE.match(lines[i].strip()):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:]).rstrip() + "\n"


def load_strategies(strat_dir: Path = STRAT_DIR) -> list[dict]:
    out = []
    for p in sorted(strat_dir.glob("*.py")):
        code = p.read_text()
        meta = {}
        for ln in code.splitlines():
            m = HEADER_RE.match(ln.strip())
            if not m:
                break
            meta[m.group(1).lower()] = m.group(2).strip()
        label = meta.get("label", "").lower()
        if label not in ("leaky", "clean"):
            raise ValueError(f"{p.name}: missing or invalid '# LEAKBENCH-LABEL: leaky|clean' header")
        stripped = strip_header(code)
        assert "LEAKBENCH" not in stripped, p.name
        out.append({"id": p.stem, "file": f"bench/strategies/{p.name}", "label": label,
                    "leaky": label == "leaky", "bug": meta.get("bug", ""), "note": meta.get("note", ""),
                    "code": code, "stripped": stripped})
    return out


# ----------------------------------------------------------------------------
# Deterministic audit (same path as agents/loop.py: policy check -> sandbox -> run_audit)
# ----------------------------------------------------------------------------
def deterministic_audit(code: str, data: dict, *, sandbox: bool = True, timeout: float = 90.0) -> dict:
    from audit.leak_tests import in_process_executor, run_audit
    from sandbox.runner import SandboxError, SandboxExecutor, check_policy

    t0 = time.time()
    try:
        if sandbox:
            violations = check_policy(code)
            if violations:
                raise SandboxError("Sandbox policy violation:\n" + "\n".join(violations))
            executor = SandboxExecutor(code, timeout)
        else:
            g: dict = {}
            exec(compile(code, "<strategy>", "exec"), g)  # trusted bench files only
            executor = in_process_executor(g["generate_signals"])
        rep = run_audit(data, executor, code, n_trials=1)
    except Exception as e:  # noqa: BLE001 - a crash is reported, never hidden
        return {"flagged": True, "error": f"{type(e).__name__}: {e}"[:600], "blocking": [], "failing": [],
                "tests": {}, "sharpe": None, "seconds": round(time.time() - t0, 2)}
    tests = {t.name: {"passed": bool(t.passed), "severity": t.severity, "blocking": t.blocking,
                      "summary": t.summary} for t in rep.tests}
    return {"flagged": not rep.clean, "error": "",
            "blocking": [t.name for t in rep.tests if t.blocking],
            "failing": [t.name for t in rep.tests if not t.passed],
            "tests": tests, "sharpe": round(float(rep.metrics.get("sharpe", 0.0)), 3),
            "static_findings": sorted({(f["rule"], f["severity"]) for f in
                                       next((t.evidence.get("findings", []) for t in rep.tests
                                             if t.name == "static_scan"), [])}),
            "seconds": round(time.time() - t0, 2)}


# ----------------------------------------------------------------------------
# LLM-only judge
# ----------------------------------------------------------------------------
def _cache_key(model: str, code: str) -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}\n{model}\n{code}".encode()).hexdigest()[:24]


def load_cache(path: Path = CACHE_PATH) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict, path: Path = CACHE_PATH) -> None:
    path.write_text(json.dumps(cache, indent=1, sort_keys=True))


def parse_verdict(text: str) -> dict | None:
    """Find the last JSON object with a boolean 'leaky' field."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    cands = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for c in reversed(cands):
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "leaky" in obj:
            v = obj["leaky"]
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "yes", "1")
            return {"leaky": bool(v), "reason": str(obj.get("reason", ""))[:500]}
    m = re.search(r'"leaky"\s*:\s*(true|false)', text, flags=re.I)
    if m:
        return {"leaky": m.group(1).lower() == "true", "reason": ""}
    return None


def make_judge_client():
    """Reuse the project's Nebius client (agents/llm.py) with a dedicated 'judge' route per model."""
    from agents.config import ModelRoute, load_config
    from agents.llm import NebiusLLM

    cfg = load_config()
    llm = NebiusLLM(cfg)

    def call(model: str, code: str, thinking: bool = True, max_tokens: int = JUDGE_MAX_TOKENS):
        role = f"judge::{model}::{int(thinking)}"
        cfg.routes[role] = ModelRoute(role, model, "LeakBench LLM-only judge", 0.0, max_tokens, thinking=thinking)
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(code=code.rstrip())}]
        return llm.chat(role, msgs)

    return call


def llm_judge(model: str, code: str, cache: dict, call=None) -> dict:
    from agents.config import estimate_cost

    key = _cache_key(model, code)
    if key in cache:
        return cache[key] | {"cached": True}
    if call is None:
        raise RuntimeError("no LLM client and no cached answer")
    attempts, verdict, last_text = [], None, ""
    for thinking, mt in ((True, JUDGE_MAX_TOKENS), (False, FALLBACK_MAX_TOKENS)):
        try:
            r = call(model, code, thinking=thinking, max_tokens=mt)
        except Exception as e:  # noqa: BLE001
            attempts.append({"thinking": thinking, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        cost = estimate_cost(model, r.prompt_tokens, r.completion_tokens) or 0.0
        attempts.append({"thinking": thinking, "prompt_tokens": r.prompt_tokens,
                         "completion_tokens": r.completion_tokens, "latency_s": round(r.latency_s, 2),
                         "cost_usd": round(cost, 6)})
        last_text = r.text
        verdict = parse_verdict(r.text)
        if verdict is not None:
            break
    entry = {"model": model, "leaky": verdict["leaky"] if verdict else None,
             "reason": verdict["reason"] if verdict else "", "parsed": verdict is not None,
             "raw_tail": last_text[-400:] if verdict is None else "",
             "attempts": attempts,
             "cost_usd": round(sum(a.get("cost_usd", 0.0) for a in attempts), 6),
             "prompt_tokens": sum(a.get("prompt_tokens", 0) for a in attempts),
             "completion_tokens": sum(a.get("completion_tokens", 0) for a in attempts),
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if verdict is not None:  # do not cache failures, so a rerun retries them
        cache[key] = entry
    return entry | {"cached": False}


# ----------------------------------------------------------------------------
# Metrics and reporting
# ----------------------------------------------------------------------------
def score(labels: list[bool], preds: list[bool | None]) -> dict:
    """Binary metrics with 'leaky' as the positive class. Unparsed (None) answers count as 'not flagged'."""
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    n = len(labels)
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "unanswered": sum(1 for p in preds if p is None),
            "recall": div(tp, tp + fn), "false_positive_rate": div(fp, fp + tn),
            "precision": div(tp, tp + fp), "accuracy": div(tp + tn, n),
            "f1": div(2 * tp, 2 * tp + fp + fn)}


def method_label(m: str) -> str:
    if m == "deterministic":
        return "Deterministic audit"
    return "LLM-only: " + SHORT_NAMES.get(m.split(":", 1)[1], m.split(":", 1)[1])


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def write_markdown(res: dict, path: Path = RESULTS_MD) -> None:
    methods = res["methods"]
    L = ["# LeakBench results", "",
         f"Generated {res['generated_at']} · {res['n_strategies']} strategies "
         f"({res['n_leaky']} leaky, {res['n_clean']} clean) · data: synthetic panel "
         f"{res['data']['n_tickers']} tickers × {res['data']['n_dates']} days, seed {res['data']['seed']}.", "",
         "Positive class = **leaky**. *Recall* = share of leaky strategies flagged; *FPR* = share of clean "
         "strategies wrongly flagged. The LLM judges saw only the code (LeakBench header stripped) and the "
         "engine contract (row t is decided after the close of t and executed at t+1).", "",
         "## Headline", "",
         "| Method | Recall | False-positive rate | Precision | Accuracy | F1 | TP/FN/FP/TN |",
         "|---|---|---|---|---|---|---|"]
    for m, s in res["metrics"].items():
        L.append(f"| {method_label(m)} | {_pct(s['recall'])} | {_pct(s['false_positive_rate'])} | "
                 f"{_pct(s['precision'])} | {_pct(s['accuracy'])} | {_pct(s['f1'])} | "
                 f"{s['tp']}/{s['fn']}/{s['fp']}/{s['tn']}" + (f" ({s['unanswered']} unanswered)" if s["unanswered"] else "") + " |")
    if res.get("llm_cost_usd") is not None and len(methods) > 1:
        L += ["", f"LLM judge cost for this results file: ${res['llm_cost_usd']:.4f} "
                  f"(new calls this run: ${res.get('llm_cost_usd_this_run', 0):.4f}; cached answers are free)."]

    abl = res.get("ablations") or {}
    if abl:
        names = {"static_scan_only": "Static AST scan alone", "dynamic_tests_only": "Dynamic tests alone (no AST scan)",
                 "static_scan_any_finding": "Static AST scan, any finding incl. non-blocking hints",
                 "future_perturbation_only": "Future-perturbation test alone"}
        L += ["", "### Ablation: which deterministic tests do the work", "",
              "| Subset of the suite | Recall | FPR |", "|---|---|---|"]
        for k, s in abl.items():
            L.append(f"| {names.get(k, k)} | {_pct(s['recall'])} | {_pct(s['false_positive_rate'])} |")
    L += ["", "## Per strategy", "",
          "✅ = correct verdict, ❌ = wrong. Deterministic column lists the blocking tests that fired.", "",
          "| Strategy | Label | Bug type | Deterministic | " + " | ".join(method_label(m).replace("LLM-only: ", "") for m in methods[1:]) + " |",
          "|---|---|---|---|" + "---|" * (len(methods) - 1)]
    for r in res["strategies"]:
        d = r["deterministic"]
        det = ("✅ " if d["flagged"] == r["leaky"] else "❌ ") + ("flagged: " + ", ".join(d["blocking"]) if d["flagged"] and d["blocking"] else ("ERROR" if d.get("error") else "pass"))
        cells = []
        for m in methods[1:]:
            j = r["llm"].get(m.split(":", 1)[1])
            if not j or j.get("leaky") is None:
                cells.append("–")
            else:
                cells.append(("✅ " if j["leaky"] == r["leaky"] else "❌ ") + ("leaky" if j["leaky"] else "clean"))
        L.append(f"| `{r['id']}` | {r['label']} | {r['bug']} | {det} | " + " | ".join(cells) + " |")

    L += ["", "## Misses and false positives", ""]
    for m in methods:
        miss = [r for r in res["strategies"] if r["leaky"] and not r["pred"][m]]
        fps = [r for r in res["strategies"] if not r["leaky"] and r["pred"][m]]
        L.append(f"**{method_label(m)}**: missed {len(miss)} leaky"
                 + (": " + ", ".join(f"`{r['id']}`" for r in miss) if miss else "")
                 + f"; false positives {len(fps)}" + (": " + ", ".join(f"`{r['id']}`" for r in fps) if fps else "") + ".")
        L.append("")
    if res.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in res["notes"]] + [""]
    if len(methods) > 1:
        L += ["## LLM reasons on the cases they got wrong", ""]
        any_wrong = False
        for r in res["strategies"]:
            for m in methods[1:]:
                j = r["llm"].get(m.split(":", 1)[1])
                if j and j.get("leaky") is not None and j["leaky"] != r["leaky"]:
                    any_wrong = True
                    L.append(f"- `{r['id']}` ({r['label']}, {r['bug']}) — {method_label(m).replace('LLM-only: ', '')} "
                             f"said **{'leaky' if j['leaky'] else 'clean'}**: {j['reason']}")
        if not any_wrong:
            L.append("- none")
        L.append("")
    L += ["## Reproduce", "", "```bash",
          "python -m bench.run_bench --no-llm      # deterministic only, offline",
          "NEBIUS_API_KEY=... python -m bench.run_bench   # + LLM judges (answers cached in bench/llm_cache.json)",
          "```", ""]
    path.write_text("\n".join(L))


def prefill_llm(strategies: list[dict], models: list[str], cache: dict, call, workers: int = 6, log=print) -> float:
    """Ask every (model, strategy) pair not yet cached, in parallel. Returns the new spend in USD."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    todo = [(m, s) for s in strategies for m in models if _cache_key(m, s["stripped"]) not in cache]
    if not todo:
        return 0.0
    lock, spent = threading.Lock(), [0.0]

    def one(ms):
        m, s = ms
        local: dict = {}
        j = llm_judge(m, s["stripped"], local, call)
        with lock:
            cache.update(local)
            spent[0] += j.get("cost_usd", 0.0)
            save_cache(cache)
            with open(SPEND_LOG, "a") as fh:
                fh.write(json.dumps({"strategy": s["id"], "model": m, "cost_usd": j.get("cost_usd", 0.0),
                                     "parsed": j.get("parsed"), "attempts": j.get("attempts", [])}) + "\n")
        log(f"[llm] {s['id']:<34} {SHORT_NAMES.get(m, m):<15} -> {j.get('leaky')}  ${j.get('cost_usd', 0):.4f}"
            f"  ({sum(a.get('latency_s', 0) for a in j.get('attempts', []))}s)")

    log(f"[llm] {len(todo)} judge calls to make ({workers} in parallel)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    return round(spent[0], 6)


def ablations(rows: list[dict]) -> dict:
    """Which parts of the deterministic suite catch what (from the same audit run)."""
    labels = [r["leaky"] for r in rows]
    static_only = [("static_scan" in r["deterministic"]["blocking"]) for r in rows]
    dynamic = [bool(set(r["deterministic"]["blocking"]) - {"static_scan"}) or bool(r["deterministic"]["error"])
               for r in rows]
    pert = [("future_perturbation" in r["deterministic"]["blocking"]) for r in rows]
    hints = [bool(r["deterministic"].get("static_findings")) for r in rows]
    return {"static_scan_only": score(labels, static_only),
            "static_scan_any_finding": score(labels, hints),
            "dynamic_tests_only": score(labels, dynamic),
            "future_perturbation_only": score(labels, pert)}


def run(*, use_llm: bool = True, models: list[str] | None = None, sandbox: bool = True,
        only: list[str] | None = None, notes: list[str] | None = None, verbose: bool = True,
        workers: int = 6) -> dict:
    from data import generate_panel
    from data.synthetic import SyntheticConfig

    models = models or DEFAULT_MODELS
    strategies = load_strategies()
    if only:
        strategies = [s for s in strategies if any(o in s["id"] for o in only)]
    cfg = SyntheticConfig()
    data = generate_panel()
    log = print if verbose else (lambda *a, **k: None)

    cache = load_cache()
    call = None
    if use_llm:
        try:
            call = make_judge_client()
        except Exception as e:  # noqa: BLE001
            log(f"[llm] client unavailable ({e}); using cached answers only")

    run_cost = 0.0
    if use_llm and call is not None:
        run_cost = prefill_llm(strategies, models, cache, call, workers=workers, log=log)

    rows = []
    for s in strategies:
        d = deterministic_audit(s["code"], data, sandbox=sandbox)
        row = {k: s[k] for k in ("id", "file", "label", "leaky", "bug", "note")}
        row["deterministic"] = d
        row["llm"] = {}
        log(f"{s['id']:<34} {s['label']:<6} det={'FLAG' if d['flagged'] else 'pass'} "
            f"{','.join(d['blocking']) or d['error'][:60]}  ({d['seconds']}s)")
        if use_llm:
            for m in models:
                try:
                    j = llm_judge(m, s["stripped"], cache, call)
                except RuntimeError as e:
                    j = {"model": m, "leaky": None, "reason": str(e), "parsed": False, "cost_usd": 0.0, "cached": False}
                row["llm"][m] = j
                log(f"    {SHORT_NAMES.get(m, m):<16} -> {j.get('leaky')}  ${j.get('cost_usd', 0):.4f}"
                    f"{' (cached)' if j.get('cached') else ''}  {j.get('reason', '')[:90]}")
        rows.append(row)

    methods = ["deterministic"] + ([f"llm:{m}" for m in models] if use_llm else [])
    for r in rows:
        r["pred"] = {"deterministic": bool(r["deterministic"]["flagged"])}
        for m in models if use_llm else []:
            r["pred"][f"llm:{m}"] = r["llm"].get(m, {}).get("leaky")
    labels = [r["leaky"] for r in rows]
    metrics = {m: score(labels, [r["pred"][m] for r in rows]) for m in methods}
    llm_cost = round(sum(j.get("cost_usd", 0.0) for r in rows for j in r["llm"].values()), 6) if use_llm else None
    res = {"benchmark": "LeakBench", "version": 1, "prompt_version": PROMPT_VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "data": {"source": "data.synthetic.generate_panel()", "seed": cfg.seed,
                    "n_tickers": int(data["close"].shape[1]), "n_dates": int(data["close"].shape[0])},
           "executor": "sandbox" if sandbox else "in-process",
           "n_strategies": len(rows), "n_leaky": sum(labels), "n_clean": len(labels) - sum(labels),
           "methods": methods, "method_labels": {m: method_label(m) for m in methods},
           "metrics": metrics, "ablations": ablations(rows), "llm_cost_usd": llm_cost, "llm_cost_usd_this_run": round(run_cost, 6),
           "judge_prompt": {"system": JUDGE_SYSTEM, "user_template": JUDGE_USER,
                            "max_tokens": JUDGE_MAX_TOKENS, "temperature": 0.0},
           "notes": notes or [], "strategies": rows}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-llm", action="store_true", help="deterministic audit only (offline)")
    ap.add_argument("--models", nargs="+", default=None, help=f"judge model IDs (default: {' '.join(DEFAULT_MODELS)})")
    ap.add_argument("--in-process", action="store_true", help="run strategies in-process instead of the sandbox")
    ap.add_argument("--only", nargs="+", default=None, help="substring filter on strategy ids")
    ap.add_argument("--notes-file", default=None, help="markdown bullet notes to embed (one per line)")
    ap.add_argument("--workers", type=int, default=6, help="parallel LLM calls")
    ap.add_argument("--no-write", action="store_true", help="do not write results.json / RESULTS.md")
    a = ap.parse_args(argv)
    notes = []
    if a.notes_file and os.path.exists(a.notes_file):
        notes = [ln.lstrip("- ").rstrip() for ln in Path(a.notes_file).read_text().splitlines() if ln.strip()]
    res = run(use_llm=not a.no_llm, models=a.models, sandbox=not a.in_process, only=a.only, notes=notes,
              workers=a.workers)
    print()
    for m, s in res["metrics"].items():
        print(f"{method_label(m):<34} recall={_pct(s['recall']):>5}  FPR={_pct(s['false_positive_rate']):>5}  "
              f"precision={_pct(s['precision']):>5}  accuracy={_pct(s['accuracy']):>5}")
    if res["llm_cost_usd"] is not None:
        print(f"LLM cost: ${res['llm_cost_usd']:.4f} total in results, ${res['llm_cost_usd_this_run']:.4f} new this run")
    if not a.no_write:
        RESULTS_JSON.write_text(json.dumps(res, indent=1, default=str))
        write_markdown(res)
        print(f"wrote {RESULTS_JSON.relative_to(ROOT)} and {RESULTS_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

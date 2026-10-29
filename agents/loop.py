"""Coder -> sandbox -> deterministic audit -> Auditor -> Coder ... until clean.

The loop's stopping rule is purely numeric: stop when no critical/high test
fails. The LLMs write code and explanations; they never decide "clean".
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from agents import prompts as P
from agents.config import estimate_cost
from agents.llm import LLMResponse, extract_code
from audit.leak_tests import run_audit
from sandbox.runner import SandboxError, SandboxExecutor, check_policy

Event = dict
OnEvent = Callable[[Event], None]


@dataclass
class Iteration:
    index: int
    code: str
    status: str = "pending"          # clean | failing | error
    error: str = ""
    tests: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    metrics_lag2: dict = field(default_factory=dict)
    equity: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    auditor_feedback: str = ""
    llm_calls: list[dict] = field(default_factory=list)
    audit_seconds: float = 0.0
    user_code: bool = False

    @property
    def clean(self) -> bool:
        return self.status == "clean"


@dataclass
class RunTrace:
    hypothesis: str
    mock: bool
    models: dict
    iterations: list[Iteration] = field(default_factory=list)
    final_status: str = "running"
    summary: str = ""
    calls: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for it, src in zip(d["iterations"], self.iterations):
            it["clean"] = src.clean
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), default=str, **kw)

    def token_totals(self) -> dict:
        tot: dict[str, dict] = {}
        for c in self.calls:
            t = tot.setdefault(c["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            t["calls"] += 1
            t["prompt_tokens"] += c["prompt_tokens"]
            t["completion_tokens"] += c["completion_tokens"]
            t["cost_usd"] += c.get("cost_usd") or 0.0
        return tot


def _record(trace: RunTrace, it: Iteration | None, resp: LLMResponse, step: str) -> dict:
    call = resp.to_dict() | {"step": step, "iteration": it.index if it else None,
                             "cost_usd": estimate_cost(resp.model, resp.prompt_tokens, resp.completion_tokens)}
    trace.calls.append(call)
    if it is not None:
        it.llm_calls.append(call)
    return call


def run_audit_loop(hypothesis: str, data: dict, llm, *, max_iters: int = 4, cost_bps: float = 10.0,
                   sandbox_timeout: float = 90.0, on_event: OnEvent | None = None,
                   summarize: bool = True, initial_code: str | None = None) -> RunTrace:
    """Run the Coder -> deterministic audit -> Auditor -> Coder loop.

    ``initial_code``: audit an existing strategy (e.g. pasted from a notebook) instead of asking the
    Coder for a first draft. Iteration 1 then runs the user's code unchanged.
    """
    emit = on_event or (lambda e: None)
    cfg = llm.config
    trace = RunTrace(hypothesis, bool(getattr(llm, "is_mock", False)),
                     {r: cfg.route(r).model for r in ("coder", "auditor", "summarizer")},
                     settings={"max_iters": max_iters, "cost_bps": cost_bps, "user_code": bool(initial_code),
                               "n_dates": int(len(data["close"])), "n_tickers": int(data["close"].shape[1])})
    prev_code, feedback, trial_sharpes = "", "", []
    last_audit_fb = ""            # the Auditor's latest leak diagnosis, kept across sandbox crashes
    crash_retries_left = 1        # one free retry when a rewrite crashes (does not consume an audit iteration)
    audited = 0                   # iterations that actually ran the audit

    for i in range(max_iters + 1):
        if audited >= max_iters:
            break
        # ---- Coder
        emit({"type": "coder_start", "iteration": i, "model": cfg.route("coder").model})
        fb = P.CODER_FEEDBACK_BLOCK.format(code=prev_code, feedback=feedback) if prev_code else ""
        msgs = [{"role": "system", "content": P.CODER_SYSTEM},
                {"role": "user", "content": P.CODER_USER.format(iteration=i + 1, hypothesis=hypothesis,
                                                                feedback_block=fb)}]
        if i == 0 and initial_code:
            code = initial_code if initial_code.endswith("\n") else initial_code + "\n"
            it = Iteration(i, code, user_code=True)
            trace.iterations.append(it)
            emit({"type": "coder_done", "iteration": i, "code": code, "call": None, "user_code": True})
        else:
            resp = llm.chat("coder", msgs, context={"hypothesis": hypothesis, "iteration": i, "feedback": feedback})
            code = extract_code(resp.text)
            it = Iteration(i, code)
            trace.iterations.append(it)
            _record(trace, it, resp, "write_code" if i == 0 else "fix_code")
            emit({"type": "coder_done", "iteration": i, "code": code, "call": it.llm_calls[-1]})

        # ---- Sandbox + deterministic audit
        t0 = time.time()
        violations = check_policy(code)
        try:
            if violations:
                raise SandboxError("Sandbox policy violation:\n" + "\n".join(violations))
            rep = run_audit(data, SandboxExecutor(code, sandbox_timeout), code, n_trials=i + 1,
                            trial_sharpes=None, cost_bps=cost_bps)
        except SandboxError as e:
            it.status, it.error = "error", str(e)
            it.audit_seconds = time.time() - t0
            emit({"type": "error", "iteration": i, "error": it.error})
            feedback = (f"The code could not run in the sandbox:\n{it.error[-1500:]}\n"
                        "Return a corrected generate_signals(data) using only numpy/pandas/math. Keep it vectorised "
                        "(no per-date Python loops).")
            if last_audit_fb:
                feedback += ("\n\nThe fix must still address the original audit findings:\n" + last_audit_fb)
            prev_code = code
            if crash_retries_left > 0:
                crash_retries_left -= 1
            else:
                audited += 1
            continue
        audited += 1
        it.audit_seconds = time.time() - t0
        rd = rep.to_dict()
        it.tests, it.metrics, it.metrics_lag2 = rd["tests"], rd["metrics"], rd["metrics_lag2"]
        it.equity = [round(float(x), 6) for x in rep.equity.to_numpy()]
        it.dates = [d.strftime("%Y-%m-%d") for d in rep.equity.index]
        it.status = "clean" if rep.clean else "failing"
        trial_sharpes.append(rep.metrics.get("sharpe_daily", 0.0))
        emit({"type": "tests_done", "iteration": i, "clean": rep.clean, "tests": it.tests, "metrics": it.metrics})
        if rep.clean:
            break

        # ---- Auditor (explains the numbers, writes fix instructions)
        if audited >= max_iters:
            break
        emit({"type": "auditor_start", "iteration": i, "model": cfg.route("auditor").model})
        msgs = [{"role": "system", "content": P.AUDITOR_SYSTEM},
                {"role": "user", "content": P.AUDITOR_USER.format(
                    hypothesis=hypothesis, iteration=i + 1, code=code, cost_bps=cost_bps,
                    metrics=P.fmt_metrics(it.metrics), tests=P.compact_tests(it.tests)) + P.localized_hint(it.tests)}]
        aresp = llm.chat("auditor", msgs, context={"hypothesis": hypothesis, "iteration": i, "code": code,
                                                   "tests": it.tests, "metrics": it.metrics})
        it.auditor_feedback = aresp.text
        _record(trace, it, aresp, "diagnose")
        emit({"type": "auditor_done", "iteration": i, "feedback": aresp.text, "call": it.llm_calls[-1]})
        prev_code, feedback = code, aresp.text
        last_audit_fb = aresp.text

    last = trace.iterations[-1] if trace.iterations else None
    trace.final_status = "clean" if last and last.clean else ("error" if last and last.status == "error" else "not_clean")

    if summarize and trace.iterations:
        emit({"type": "summary_start", "model": cfg.route("summarizer").model})
        lines = []
        for it in trace.iterations:
            fails = [t["name"] for t in it.tests if not t["passed"]]
            lines.append(f"- Iteration {it.index + 1}: status={it.status}; " +
                         (f"metrics: {P.fmt_metrics(it.metrics)}; " if it.metrics else "") +
                         f"failing tests: {', '.join(fails) or 'none'}" + (f"; error: {it.error[:200]}" if it.error else ""))
        msgs = [{"role": "system", "content": P.SUMMARIZER_SYSTEM},
                {"role": "user", "content": P.SUMMARIZER_USER.format(hypothesis=hypothesis, status=trace.final_status,
                                                                     cost_bps=cost_bps,
                                                                     origin="pasted by the user" if initial_code else "written by the Coder model",
                                                                     iterations="\n".join(lines))}]
        sresp = llm.chat("summarizer", msgs, context={"hypothesis": hypothesis,
                                                      "iterations": [asdict(x) | {"clean": x.clean}
                                                                     for x in trace.iterations]})
        trace.summary = sresp.text
        _record(trace, None, sresp, "summarize")
    trace.finished_at = time.time()
    emit({"type": "finished", "status": trace.final_status})
    return trace

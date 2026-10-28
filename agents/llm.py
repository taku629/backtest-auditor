"""LLM clients: Nebius Token Factory (OpenAI-compatible) and an offline MockLLM."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

from agents.config import AgentConfig, load_config, use_mock
from agents.mock_strategies import SCENARIOS, pick_scenario

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


@dataclass
class LLMResponse:
    text: str
    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    mock: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def extract_code(text: str) -> str:
    text = THINK_RE.sub("", text)
    blocks = CODE_RE.findall(text)
    if blocks:
        with_fn = [b for b in blocks if "def generate_signals" in b]
        return (with_fn or blocks)[-1].strip() + "\n"
    return text.strip() + "\n"


class NebiusLLM:
    """Calls Nebius Token Factory through the OpenAI SDK."""

    def __init__(self, config: AgentConfig | None = None):
        from openai import OpenAI

        self.config = config or load_config()
        key = self.config.api_key()
        if not key:
            raise RuntimeError(f"{self.config.api_key_env} is not set (or set USE_MOCK=1 to run offline)")
        self.client = OpenAI(base_url=self.config.base_url, api_key=key,
                             timeout=self.config.request_timeout, max_retries=self.config.max_retries)
        self.is_mock = False

    def chat(self, role: str, messages: list[dict], context: dict | None = None) -> LLMResponse:
        route = self.config.route(role)
        t0 = time.time()
        extra = None if route.thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self.client.chat.completions.create(model=route.model, messages=messages,
                                                   temperature=route.temperature, max_tokens=route.max_tokens,
                                                   extra_body=extra)
        msg = resp.choices[0].message
        text = msg.content or ""
        if not text.strip():  # reasoning consumed the budget: fall back to the reasoning trace
            text = getattr(msg, "reasoning_content", None) or ""
        text = THINK_RE.sub("", text).strip()
        usage = getattr(resp, "usage", None)
        return LLMResponse(text, role, route.model,
                           int(getattr(usage, "prompt_tokens", 0) or 0),
                           int(getattr(usage, "completion_tokens", 0) or 0),
                           time.time() - t0, False)


def _tok(s: str) -> int:
    return max(1, len(s) // 4)


class MockLLM:
    """Deterministic stand-in. Reports the configured Nemotron model IDs, flagged mock=True.

    * coder: returns canned drafts for the scenario matched from the hypothesis
      (first draft leaky, later drafts fixed). Iteration comes from the context.
    * auditor: writes a diagnosis from the *actual* failing test JSON.
    * summarizer: templated summary from the run's real numbers.
    """

    def __init__(self, config: AgentConfig | None = None, latency_s: float = 0.0):
        self.config = config or load_config()
        self.latency_s = latency_s
        self.is_mock = True

    def chat(self, role: str, messages: list[dict], context: dict | None = None) -> LLMResponse:
        ctx = context or {}
        t0 = time.time()
        if self.latency_s:
            time.sleep(self.latency_s)
        if role == "coder":
            versions = SCENARIOS[pick_scenario(ctx.get("hypothesis", ""))]
            i = min(int(ctx.get("iteration", 0)), len(versions) - 1)
            text = f"```python\n{versions[i]}```"
        elif role == "auditor":
            text = self._audit(ctx)
        else:
            text = self._summary(ctx)
        prompt = "\n".join(m["content"] for m in messages)
        return LLMResponse(text, role, self.config.route(role).model, _tok(prompt), _tok(text),
                           time.time() - t0, True)

    @staticmethod
    def _audit(ctx: dict) -> str:
        failing = [t for t in ctx.get("tests", []) if not t["passed"]]
        code = ctx.get("code", "")
        lines = code.splitlines()
        cause, fixes = [], []

        def find(pattern):
            for n, ln in enumerate(lines, 1):
                if re.search(pattern, ln):
                    return n, ln.strip()
            return None

        checks = [
            (r"\.shift\(\s*-", "uses `.shift(-n)`, so row t contains tomorrow's value: the signal is computed "
                               "from the future.", "Delete the negative shift; the engine already trades at t+1."),
            (r"bfill|backfill", "back-fills weights/data, copying end-of-week (future) values into earlier days.",
             "Rebalance on the first trading day of each period and forward-fill (`ffill`) until the next one."),
            (r"dropna\(axis=1", "drops every ticker with a missing value, i.e. keeps only names that survive "
                                "to the end of the sample (survivorship bias).",
             "Use a point-in-time universe: `listed = close.notna()`, and mask ranks/weights with it."),
            (r"center\s*=\s*True", "uses a centred rolling window, which averages over future bars.",
             "Use a trailing window (`rolling(n)` without `center=True`)."),
            (r"\.mean\(\)\s*\)\s*/.*\.std\(\)", "standardises with full-sample mean/std, which include future data.",
             "Use trailing statistics (`rolling(n).std()`) or cross-sectional ones (`axis=1`)."),
        ]
        for pat, why, fix in checks:
            hit = find(pat)
            if hit:
                cause.append(f"- Line {hit[0]} `{hit[1]}` {why}")
                fixes.append(fix)
        for t in failing:
            if t["name"] == "future_perturbation" and not any("future" in c for c in cause):
                cause.append(f"- {t['summary']}")
            if t["name"] == "deflated_sharpe":
                fixes.append("(Non-blocking) Deflated Sharpe is below threshold: keep the number of variants low "
                             "and confirm out of sample.")
        if not cause:
            cause = [f"- {t['name']}: {t['summary']}" for t in failing] or ["- No failing tests."]
        for t in failing:
            if t["name"] == "future_perturbation":
                ev = t["evidence"]
                cause.append(f"- Proof: scrambling data after the cut changed past weights at "
                             f"{ev.get('n_leaking_cuts')}/{ev.get('n_cuts')} cut dates (first leak "
                             f"{ev.get('first_leaking_date')}).")
        m = ctx.get("metrics", {})
        fixes_txt = "\n".join(f"{i}. {f}" for i, f in enumerate(dict.fromkeys(fixes), 1)) or "1. No changes required."
        return ("### Root cause\n" + "\n".join(cause) + "\n\n### Fix instructions\n" + fixes_txt +
                f"\n\n### Risk if ignored\nThe reported Sharpe of {m.get('sharpe', 0):.2f} is produced by "
                f"information an investor could not have had, so live results would fall far short.")

    @staticmethod
    def _summary(ctx: dict) -> str:
        its = ctx.get("iterations", [])
        if not its:
            return "- No iterations were run."
        first, last = its[0], its[-1]
        fm, lm = first.get("metrics") or {}, last.get("metrics") or {}
        first_fail = [t["name"] for t in first.get("tests", []) if t.get("blocking")]
        soft = [t["name"] for t in last.get("tests", []) if not t["passed"] and not t.get("blocking")]
        status = "CLEAN" if last.get("clean") else "NOT CLEAN"
        out = [f"- First draft reported Sharpe {fm.get('sharpe', 0):.2f} / CAGR {fm.get('cagr', 0):.0%}, but failed "
               f"{len(first_fail)} blocking test(s): {', '.join(first_fail) or 'none'}."]
        if len(its) > 1:
            out.append(f"- After {len(its)} iterations the strategy is {status}: Sharpe {lm.get('sharpe', 0):.2f}, "
                       f"CAGR {lm.get('cagr', 0):.0%}, max drawdown {lm.get('max_drawdown', 0):.0%}.")
            fs, ls_ = fm.get("sharpe", 0.0), lm.get("sharpe", 0.0)
            if fs > 0 and ls_ <= 0:
                out.append("- All of the first draft's edge was an artefact: the honest version does not make money.")
            elif fs > 0:
                out.append(f"- {min(1.0, max(0.0, 1 - ls_ / fs)):.0%} of the first draft's Sharpe came from "
                           f"look-ahead / survivorship artefacts.")
        out.append("- Every verdict comes from deterministic tests (future-data perturbation, survivorship "
                   "counterfactual, execution delay, AST scan); the LLM only explained and fixed.")
        if soft:
            out.append(f"- Remaining non-blocking caveats: {', '.join(soft)}. Results are on synthetic data.")
        return "\n".join(out)


def make_llm(mock: bool | None = None, config: AgentConfig | None = None, mock_latency_s: float = 0.0):
    if mock is None:
        mock = use_mock()
    return MockLLM(config, mock_latency_s) if mock else NebiusLLM(config)

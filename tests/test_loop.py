import json

from agents.config import load_config
from agents.llm import MockLLM, extract_code
from agents.loop import run_audit_loop


def test_momentum_loop_converges(panel):
    events = []
    tr = run_audit_loop("Buy top-decile 20-day momentum, rebalance weekly", panel, MockLLM(),
                        max_iters=4, on_event=events.append)
    assert tr.final_status == "clean"
    assert len(tr.iterations) == 3
    first, last = tr.iterations[0], tr.iterations[-1]
    assert first.status == "failing" and any(t["blocking"] for t in first.tests)
    assert last.clean
    assert first.metrics["sharpe"] > 3 * last.metrics["sharpe"]
    roles = [c["role"] for c in tr.calls]
    assert roles == ["coder", "auditor", "coder", "auditor", "coder", "summarizer"]
    cfg = load_config()
    assert {c["model"] for c in tr.calls} == {cfg.route(r).model for r in ("coder", "auditor", "summarizer")}
    assert all(c["prompt_tokens"] > 0 for c in tr.calls)
    assert tr.summary
    assert first.auditor_feedback.startswith("### Root cause")
    assert {"coder_done", "tests_done", "auditor_done", "finished"} <= {e["type"] for e in events}
    json.loads(tr.to_json())


def test_other_scenarios_converge(panel):
    for hyp in ("Mean reversion: buy names furthest below their moving average",
                "Low volatility: hold the least volatile names, monthly"):
        tr = run_audit_loop(hyp, panel, MockLLM(), max_iters=4, summarize=False)
        assert tr.final_status == "clean" and len(tr.iterations) == 2


def test_max_iters_respected(panel):
    tr = run_audit_loop("momentum", panel, MockLLM(), max_iters=1, summarize=False)
    assert len(tr.iterations) == 1 and tr.final_status == "not_clean"


def test_sandbox_error_is_fed_back(panel):
    class BadThenGood(MockLLM):
        n = 0

        def chat(self, role, messages, context=None):
            r = super().chat(role, messages, context)
            if role == "coder" and context["iteration"] == 0:
                r.text = "```python\nimport os\ndef generate_signals(data):\n    return data['close']\n```"
            return r

    tr = run_audit_loop("momentum", panel, BadThenGood(), max_iters=2, summarize=False)
    assert tr.iterations[0].status == "error" and "import 'os'" in tr.iterations[0].error


def test_extract_code_handles_think_and_prose():
    txt = "<think>hmm ```python\nbad\n```</think>Here:\n```python\ndef generate_signals(data):\n    return 1\n```\nDone."
    assert extract_code(txt).startswith("def generate_signals")


def test_nebius_client_plumbing(monkeypatch, panel):
    """NebiusLLM wiring (base_url, model routing, usage, <think> stripping) without network."""
    from types import SimpleNamespace

    from agents.llm import NebiusLLM
    from agents.mock_strategies import MOMENTUM_V3

    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    llm = NebiusLLM()
    assert str(llm.client.base_url).startswith("https://api.tokenfactory.nebius.com/v1")
    seen = []

    def fake_create(model, messages, temperature, max_tokens):
        seen.append(model)
        text = "<think>plan</think>```python\n" + MOMENTUM_V3 + "```" if "Iteration" in messages[-1]["content"] \
            else "- summary"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    tr = run_audit_loop("momentum", panel, llm, max_iters=2)
    assert tr.final_status == "clean" and not tr.mock
    assert seen == [llm.config.route("coder").model, llm.config.route("summarizer").model]
    assert tr.calls[0]["prompt_tokens"] == 100 and tr.calls[0]["cost_usd"] > 0

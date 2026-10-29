"""Offline smoke test of bench/run_repair.py: MockLLM, 2 leaky strategies, conditions A and B."""
import json

from agents.llm import MockLLM


class PromptSpy(MockLLM):
    """MockLLM that records the Auditor prompts per thread condition."""

    def __init__(self):
        super().__init__()
        self.auditor_prompts = []

    def chat(self, role, messages, context=None):
        if role == "auditor":
            from bench.run_repair import _COND
            self.auditor_prompts.append((getattr(_COND, "no_localization", False), messages[-1]["content"]))
        return super().chat(role, messages, context)


def test_run_repair_offline(small_panel, tmp_path):
    from bench import run_repair as R

    spy = PromptSpy()
    out = tmp_path / "repair.json"
    res = R.run_repair(only=["L01_", "L05_"], conditions=("A", "B"), llm=spy, data=small_panel,
                       results_path=out, cache_path=tmp_path / "c.json", spend_path=tmp_path / "s.jsonl",
                       workers=4, verbose=False)
    runs = res["runs"]
    assert {(r["id"], r["condition"]) for r in runs} == {
        (s, c) for s in ("L01_shift_minus_one", "L05_dropna_universe") for c in ("A", "B")}
    for r in runs:
        assert r["status"] == "done", r
        assert r["iterations"] >= 1 and r["leaky_sharpe"] is not None
        assert isinstance(r["over_corrected"], bool) and r["first_blocking"]
        assert r["localized_found"] and r["localized_line"]
    # Condition B must hide the localization from the Auditor; A must show it.
    a_prompts = [p for nl, p in spy.auditor_prompts if not nl]
    b_prompts = [p for nl, p in spy.auditor_prompts if nl]
    assert a_prompts and b_prompts
    assert all("Dynamic leak localization" in p and '"localized"' in p for p in a_prompts)
    assert not any("Dynamic leak localization" in p or '"localized"' in p for p in b_prompts)
    # Patches are thread-local: outside a condition, the original behaviour is intact.
    from agents import prompts as P
    assert P.localized_hint([]) == ""
    assert set(res["summary"]) == {"A", "B"} and res["mock"] is True
    assert json.loads(out.read_text())["runs"]
    R.write_markdown(res, path=tmp_path / "REPAIR.md")
    assert "LeakBench repair" in (tmp_path / "REPAIR.md").read_text()
    assert not (tmp_path / "s.jsonl").exists()  # mock calls are never logged as spend


def test_mentions_line():
    from bench.run_repair import mentions_line
    loc = {"found": True, "entry": {"line": 11, "source_line": "mom = mom.shift(-1)"}}
    assert mentions_line("Line 11 `mom = mom.shift(-1)` reads tomorrow", loc)
    assert mentions_line("the statement mom = mom.shift(-1) leaks", loc)
    assert not mentions_line("line 3 is wrong", loc)
    assert mentions_line("anything", {"found": False}) is None

"""Offline tests of the in-the-wild study (bench/run_wild.py) and its app panel (bench/ui.render_wild)."""
import json

from agents.llm import LLMResponse

LEAKY = '''import pandas as pd

def generate_signals(data):
    close = data["close"]
    mom = close.pct_change(20).shift(-1)
    top = mom.rank(axis=1, pct=True) > 0.8
    w = top.astype(float)
    return w.div(w.sum(axis=1).replace(0, float("nan")), axis=0).fillna(0.0)
'''

CLEAN = '''import pandas as pd

def generate_signals(data):
    close = data["close"]
    mom = close.pct_change(20)
    top = mom.rank(axis=1, pct=True) > 0.8
    w = top.astype(float)
    return w.div(w.sum(axis=1).replace(0, float("nan")), axis=0).fillna(0.0)
'''

CRASH = '''import pandas as pd

def generate_signals(data):
    close = data["close"]
    return close.not_a_method()
'''


class FakeCoder:
    """Answers by prompt id: W01 leaky, W02 clean, W03 crashes first and is fixed on the retry."""

    is_mock = False

    def __init__(self, config):
        self.config = config
        self.calls = []

    def chat(self, role, messages, context=None):
        user = messages[-1]["content"]
        self.calls.append((role, user[:60]))
        if "raised this error" in user:
            code = CLEAN
        elif "12-1 month" in user:       # W01
            code = LEAKY
        elif "Time-series momentum" in user:   # W02
            code = CLEAN
        else:                              # W03
            code = CRASH
        model = self.config.route(role).model
        return LLMResponse(f"```python\n{code}```", role, model, 100, 200, 0.01, False)


def _run(tmp_path, small_panel, llm=None, offline=False):
    from bench import run_wild as W

    cfg = W.make_config()
    fake = llm or FakeCoder(cfg)
    res = W.run_wild(only=["W01", "W02", "W03"], models=("super",), systems=("app", "chat"), data=small_panel,
                     llm=None if offline else fake, offline=offline, results_path=tmp_path / "r.json",
                     cache_path=tmp_path / "c.json", spend_path=tmp_path / "s.jsonl", pbo=False, verbose=False)
    return W, res, fake


def test_run_wild_offline(tmp_path, small_panel):
    W, res, fake = _run(tmp_path, small_panel)
    runs = res["runs"]
    assert set(runs) == {f"W0{i}|super|{s}" for i in (1, 2, 3) for s in ("app", "chat")}
    leaky = runs["W01|super|app"]
    assert leaky["audit"] == "ok" and "future_perturbation" in leaky["blocking"]
    assert "look_ahead" in leaky["leak_classes"]
    assert leaky["localized_found"] and "shift(-1)" in leaky["localized_source"]
    assert leaky["quickfix"]["status"] == "fixed"
    assert runs["W02|super|chat"]["clean"] is True
    crashed = runs["W03|super|app"]
    assert crashed["retried"] and crashed["first_audit"] == "crashed" and crashed["clean"] is True
    s = res["summary"]["all"]
    assert s["drafted"] == 6 and s["ran"] == 6 and s["leaky"] == 2 and s["quickfix_fixed"] == 2
    assert s["look_ahead"] == 2 and s["look_ahead_localized"] == 2
    assert "super/app" in res["summary"] and "style:plain" in res["summary"]
    # every distinct prompt was cached; the spend log has one line per fresh call (cost 0 for unknown pricing)
    cache = json.loads((tmp_path / "c.json").read_text())
    assert len(cache) == len(fake.calls) == 8   # 6 drafts + 2 crash retries
    # report + slim public json
    md = W.write_report(res, path=tmp_path / "WILD.md")
    assert "In the wild" in md and "Caveats" in md and "W01|super|app" in md
    pub = W.public_summary(res)
    assert pub["summary"]["all"]["leaky"] == 2 and len(pub["rows"]) == 6
    assert all("code" not in r for r in pub["rows"])


def test_run_wild_replays_from_cache(tmp_path, small_panel):
    W, res1, _ = _run(tmp_path, small_panel)
    (tmp_path / "r.json").unlink()        # force re-audit, drafts must come from the cache only
    W2, res2, _ = _run(tmp_path, small_panel, offline=True)
    assert res2["summary"]["all"] == res1["summary"]["all"]


def test_classify_splits_survivorship():
    from bench.run_wild import classify
    r = {"blocking": ["survivorship", "future_perturbation"],
         "test_summaries": {"survivorship": "SURVIVORSHIP: weights assigned to tickers before they were listed."}}
    assert classify(r) == ["look_ahead", "pre_listing_weight"]
    r["test_summaries"]["survivorship"] = "SURVIVORSHIP: past signals change when delisted names are made to survive"
    assert classify(r) == ["look_ahead", "survivor_universe"]
    assert classify({"blocking": ["non_trivial"]}) == []


def test_prompts_are_thirty_and_neutral():
    from bench.wild_prompts import WILD_PROMPTS
    assert len(WILD_PROMPTS) == 30
    assert {p["style"] for p in WILD_PROMPTS} == {"plain", "notebook"}
    for p in WILD_PROMPTS:   # ordinary requests: never ask for future data
        t = p["text"].lower()
        assert "shift(-" not in t and "future" not in t and "tomorrow" not in t


class FakeSt:
    def __init__(self):
        self.out = []

    def __getattr__(self, name):
        def f(*a, **k):
            self.out.append((name, a))
            if name == "columns":
                return [FakeSt() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))]
            return None
        return f


def test_render_wild(tmp_path, small_panel):
    from bench import ui
    W, res, _ = _run(tmp_path, small_panel)
    p = tmp_path / "pub.json"
    p.write_text(json.dumps(W.public_summary(res)))
    st = FakeSt()
    ui.render_wild(st, p)
    names = [n for n, _ in st.out]
    assert "subheader" in names and names.count("dataframe") == 2
    st2 = FakeSt()
    ui.render_wild(st2, tmp_path / "missing.json")
    assert ("info" in [n for n, _ in st2.out])


def test_bundled_wild_public_json_loads():
    from bench import ui
    w = ui.load_wild()
    if w is None:   # not generated in this checkout
        return
    a = w["summary"]["all"]
    assert a["ran"] >= a["leaky"] >= a["quickfix_fixed"] >= 0
    assert ui.wild_frame(w).shape[0] == sum(r["outcome"] == "leaky" for r in w["rows"])

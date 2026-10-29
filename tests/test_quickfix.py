"""Deterministic Quick fix (audit/quickfix.py): candidate generation, audit gate, loop and CLI wiring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit.quickfix import _over_corrected, generate_candidates, quick_fix, unified_diff

ROOT = Path(__file__).resolve().parent.parent
STRATS = ROOT / "bench" / "strategies"


def _body(*lines: str) -> str:
    return ("import numpy as np\nimport pandas as pd\n\n\ndef generate_signals(data):\n"
            + "".join(f"    {ln}\n" for ln in lines))


def _first(code: str, line: int) -> str:
    c = generate_candidates(code, [line])
    assert c, "no candidate generated"
    return c[0].code.splitlines()[line - 1].strip()


# ---------------------------------------------------------------- candidate generation (pure AST, fast)
@pytest.mark.parametrize("src, expected", [
    ("x = c.shift(-1)", "x = c.shift(1)"),
    ("x = c.pct_change(periods=-3)", "x = c.pct_change(periods=3)"),
    ("x = c.shift(-lag)", "x = c.shift(lag)"),
    ("x = c.rolling(21, center=True, min_periods=5).mean()", "x = c.rolling(21, min_periods=5).mean()"),
    ("x = c.bfill()", "x = c.ffill()"),
    ("x = c.interpolate(limit=3)", "x = c.ffill()"),
    ("x = w.reindex(c.index, method='bfill')", "x = w.reindex(c.index, method='ffill')"),
    ("x = m.rank(pct=True)", "x = m.rank(axis=1, pct=True)"),
    ("x = m.rank(axis=0, pct=True)", "x = m.expanding(min_periods=20).rank(pct=True)"),
    ("x = (m - m.mean()) / m.std()",
     "x = (m - m.expanding(min_periods=20).mean()) / m.expanding(min_periods=20).std()"),
    ("x = m.sub(m.mean(), axis=1)", "x = m.sub(m.mean(axis=1), axis=0)"),
    ("x = m.fillna(m.mean())", "x = m.fillna(m.expanding(min_periods=20).mean())"),
    ("x = c.resample('MS').last()", "x = c.resample('ME').last()"),
    ("x = w.groupby(month).transform('last')", "x = w.groupby(month).transform('first')"),
    ("x = c.iloc[::-1].cummax().iloc[::-1]", "x = c.cummax()"),
    ("x = c[::-1].ewm(span=5).std()[::-1]", "x = c.ewm(span=5).std()"),
    ("x = c.dropna(axis=1)", "x = c"),
])
def test_rule_rewrites(src, expected):
    code = _body("c = data['close']", "month = c.index.to_period('M')", "m = c.pct_change()", "w = m", "lag = 1",
                 src, "return x")
    assert _first(code, 11) == expected


def test_reversed_frame_defined_on_another_line():
    code = _body("c = data['close']", "rf = c.sort_index(ascending=False)",
                 "ma = rf.rolling(20).mean().sort_index()", "return ma")
    new = generate_candidates(code, [8])[0].code
    assert "rf = c\n" in new and "ascending=False" not in new


def test_pooled_quantile_rewrites():
    code = _body("c = data['close']", "m = c.pct_change(20)", "s = m.stack()",
                 "lvl = s.groupby(level=1).quantile(0.8)", "b = pd.qcut(s, 5, labels=False).unstack()", "return b")
    assert "m.expanding(min_periods=20).quantile(0.8)" in generate_candidates(code, [9])[0].code
    assert "s.groupby(level=0).transform(lambda _s: pd.qcut(_s, 5, labels=False))" in \
        generate_candidates(code, [10])[0].code


def test_forward_target_in_trailing_window_lags_the_estimate():
    code = _body("c = data['close']", "sig = -c.pct_change(5)", "fwd = c.pct_change().shift(-1)",
                 "ic = sig.rolling(60).corr(fwd)", "return sig.where(ic > 0)")
    cands = [c.code for c in generate_candidates(code, [8])]
    assert any("ic = sig.rolling(60).corr(fwd).shift(1)" in c and "shift(-1)" in c for c in cands)


def test_point_in_time_selection_rewrites():
    code = _body("c = data['close']", "v = data['volume']", "listed = c.notna()",
                 "perf = c.iloc[-1] / c.iloc[-252] - 1", "best = perf.nlargest(10).index",
                 "adv = v.iloc[-60:].median()", "seasoned = c.count() >= 500", "alive = c.columns[c.iloc[-1].notna()]",
                 "x = c.pct_change(5)", "x.loc[:, ~c.columns.isin(best)] = np.nan", "return x")
    lines = generate_candidates(code, [9])[0].code.splitlines()
    assert lines[8].strip() == "perf = c / c.shift(251) - 1"
    assert lines[9].strip() == "best = perf.rank(axis=1, ascending=False, method='first') <= 10"
    assert lines[14].strip() == "x = x.where(best, np.nan)"
    assert generate_candidates(code, [11])[0].code.splitlines()[10].strip() == "adv = v.rolling(60).median()"
    assert generate_candidates(code, [12])[0].code.splitlines()[11].strip() == "seasoned = c.expanding().count() >= 500"
    assert generate_candidates(code, [13])[0].code.splitlines()[12].strip() == "alive = c.notna()"
    # consumers: X[mask] and X.loc[:, mask] become date-wise .where(mask)
    code2 = _body("c = data['close']", "alive = c.iloc[-1].notna()", "m = c.pct_change().loc[:, alive]",
                  "k = c.tail(20).mean()", "top = k.nsmallest(3).index", "return m[top]")
    new = generate_candidates(code2, [7])[0].code.splitlines()
    assert new[6].strip() == "alive = c.notna()" and new[7].strip() == "m = c.pct_change().where(alive)"
    new = generate_candidates(code2, [9])[0].code.splitlines()
    assert new[8].strip() == "k = c.rolling(20).mean()" and new[10].strip() == "return m.where(top)"


def test_full_sample_fit_rewrite_is_expanding_ols():
    code = _body("c = data['close']", "x = np.arange(len(c), dtype=float)", "y = np.log(c.iloc[:, 0]).to_numpy()",
                 "slope, icpt = np.polyfit(x, y, 1)", "coef = np.polyfit(x, y, deg=1)", "return y - (icpt + slope * x)")
    for ln in (9, 10):
        cand = generate_candidates(code, [ln])[0]
        assert cand.rule == "full_sample_fit" and cand.lines == [ln]
        assert cand.code.splitlines()[ln - 1].strip().endswith("_expanding_polyfit1(x, y)")
        assert "def _expanding_polyfit1(x, y):" in cand.code
    ns: dict = {}
    exec(compile(generate_candidates(code, [9])[0].code, "<c>", "exec"), ns)  # noqa: S102 - our own code
    xs = np.arange(50, dtype=float)
    ys = np.r_[2 + 0.5 * xs[:30], 100 + 3 * xs[30:]]
    b, a = ns["_expanding_polyfit1"](xs, ys)
    for t in (5, 29, 49):   # each point's fit uses only data up to it (no look-ahead)
        bb, aa = np.polyfit(xs[:t + 1], ys[:t + 1], 1)
        assert np.isclose(b[t], bb) and np.isclose(a[t], aa)
    assert np.isnan(b[0])


def test_fit_helper_goes_above_an_adapter_wrapper():
    from audit.adapters import WRAPPER_MARK
    code = _body("c = data['close']", "x = np.arange(len(c), dtype=float)", "b, a = np.polyfit(x, x, 1)",
                 "return c") + "\n\n" + WRAPPER_MARK + "\nimport bta_adapter\n"
    new = generate_candidates(code, [8])[0].code
    assert new.index("def _expanding_polyfit1") < new.index(WRAPPER_MARK)


def test_candidates_keep_comments_and_are_deterministic():
    code = _body("c = data['close']  # prices", "# a comment", "x = c.shift(-1)", "return x")
    a = [c.code for c in generate_candidates(code, [8])]
    b = [c.code for c in generate_candidates(code, [8])]
    assert a == b and "# a comment" in a[0] and "# prices" in a[0]
    d = unified_diff(code, a[0], "s.py")
    assert d.startswith("--- a/s.py") and "-    x = c.shift(-1)" in d and "+    x = c.shift(1)" in d


NO_RULE = _body("c = data['close']", "m = c.pct_change(20)", "thr = np.nanpercentile(m.to_numpy(), 90)",
                "p = (m > thr).astype(float)", "return p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)")


def test_no_rule_no_candidate_without_fallback():
    code = _body("c = data['close']", "thr = np.nanpercentile(c.to_numpy(), 90)", "return c > thr")
    assert generate_candidates(code, [7]) == []
    # the localized-line fallback still offers a one-bar lag
    assert [c.rule for c in generate_candidates(code, [7], fallback_lines=[7])] == ["lag_statement"]


# ---------------------------------------------------------------- audit gate (in-process executors, fast)
def _fn(code: str):
    ns: dict = {}
    exec(compile(code, "<strategy>", "exec"), ns)  # noqa: S102 - test-only, our own code
    return ns["generate_signals"]


def _factory(code):
    from audit.leak_tests import in_process_executor
    return in_process_executor(_fn(code))


LEAKY_SHIFT = _body("c = data['close']", "listed = c.notna()", "mom = c.pct_change(20).shift(-1)",
                    "r = mom.where(listed).rank(axis=1, pct=True)", "p = ((r >= 0.8) & listed).astype(float)",
                    "return p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)")
CLEAN = LEAKY_SHIFT.replace(".shift(-1)", ".shift(1)")


def test_clean_code_gets_no_change(small_panel):
    qf = quick_fix(CLEAN, small_panel, executor_factory=_factory)
    assert qf["status"] == "not_needed" and qf["diff"] == "" and qf["n_candidates"] == 0 and qf["cost_usd"] == 0.0


def test_fixes_negative_shift_in_process(small_panel):
    qf = quick_fix(LEAKY_SHIFT, small_panel, executor_factory=_factory)
    assert qf["status"] == "fixed" and qf["rule"] == "negative_period" and qf["lines"] == [8]
    assert qf["after"]["clean"] and qf["before"]["blocking"] and "shift(1)" in qf["code"]
    assert qf["report"].clean


def test_candidate_that_goes_to_cash_is_rejected(small_panel):
    from audit.leak_tests import in_process_executor

    def factory(code):
        if code == LEAKY_SHIFT:
            return _factory(code)
        return in_process_executor(lambda d: pd.DataFrame(0.0, index=d["close"].index, columns=d["close"].columns))

    qf = quick_fix(LEAKY_SHIFT, small_panel, executor_factory=factory)
    assert qf["status"] == "failed" and qf["code"] is None
    assert qf["attempts"] and all(a["result"] in ("still_failing", "over_corrected") for a in qf["attempts"])


def test_over_corrected_rule():
    class T:
        def __init__(self, name, passed):
            self.name, self.passed = name, passed

    class R:
        def __init__(self, gross, nt=True):
            self.metrics, self.tests = {"avg_gross": gross}, [T("non_trivial", nt)]
    assert _over_corrected(R(0.01)) and _over_corrected(R(1.0, nt=False)) and not _over_corrected(R(0.5))


# ---------------------------------------------------------------- LeakBench files through the sandbox
@pytest.mark.parametrize("name, rule", [("L17_rank_default_axis", "time_axis_rank"),
                                        ("L19_resample_ms_last", "resample_start_label")])
def test_leakbench_static_blind_cases_are_fixed(panel, name, rule):
    code = (STRATS / f"{name}.py").read_text()
    qf = quick_fix(code, panel)
    assert qf["status"] == "fixed" and qf["rule"] == rule and qf["after"]["avg_gross"] >= 0.05


def test_leak_without_a_rule_is_declined(panel):
    qf = quick_fix(NO_RULE, panel)   # a pooled full-panel percentile threshold: no rewrite rule, lag fallback fails
    assert qf["status"] in ("failed", "no_candidate") and qf["code"] is None and qf["diff"] == ""


@pytest.mark.parametrize("name, rule", [("L06_select_by_final_return", "point_in_time_selection"),
                                        ("L13_alive_at_end", "point_in_time_selection"),
                                        ("L14_polyfit_detrend", "full_sample_fit"),
                                        ("L15_min_history_filter", "point_in_time_selection"),
                                        ("L16_liquidity_by_recent_volume", "point_in_time_selection")])
def test_leakbench_selection_and_regression_leaks_are_fixed(panel, name, rule):
    code = (STRATS / f"{name}.py").read_text()
    qf = quick_fix(code, panel)
    assert qf["status"] == "fixed" and qf["rule"] == rule and qf["after"]["clean"]
    assert qf["after"]["avg_gross"] >= 0.05 and qf["report"].clean


# ---------------------------------------------------------------- loop wiring
def test_loop_uses_quick_fix_before_the_llm(small_panel):
    from agents.llm import MockLLM
    from agents.loop import run_audit_loop

    tr = run_audit_loop("momentum", small_panel, MockLLM(), max_iters=3, summarize=False, initial_code=LEAKY_SHIFT)
    assert tr.final_status == "clean" and len(tr.iterations) == 2
    assert tr.iterations[0].source == "user" and tr.iterations[0].quick_fix["status"] == "fixed"
    last = tr.iterations[-1]
    assert last.source == "quick_fix" and last.quick_fix["rule"] == "negative_period" and "shift(1)" in last.code
    assert tr.calls == []                                     # no Auditor, no Coder: $0
    assert "report" not in last.quick_fix                     # JSON-safe
    tr.to_json()


def test_loop_quick_fix_can_be_disabled(small_panel):
    from agents.llm import MockLLM
    from agents.loop import run_audit_loop

    tr = run_audit_loop("momentum", small_panel, MockLLM(), max_iters=3, summarize=False, initial_code=LEAKY_SHIFT,
                        quick_fix=False)
    assert all(it.source != "quick_fix" for it in tr.iterations)
    assert any(c["step"] == "diagnose" for c in tr.calls)


def test_loop_falls_back_to_llm_when_no_candidate_passes(small_panel):
    from agents.llm import MockLLM
    from agents.loop import run_audit_loop

    tr = run_audit_loop("momentum above a percentile threshold", small_panel, MockLLM(), max_iters=2,
                        summarize=False, initial_code=NO_RULE)
    first = tr.iterations[0]
    assert first.quick_fix.get("status") in ("failed", "no_candidate")
    assert any(c["step"] == "diagnose" for c in tr.calls)       # the LLM path ran
    assert tr.iterations[1].source == "coder"                 # the Coder's rewrite came next


def test_quickfix_results_file_is_consistent():
    import json
    p = ROOT / "bench" / "quickfix_results.json"
    rows = json.loads(p.read_text())["rows"]
    leaky = [r for r in rows if r["label"] == "leaky" and not r["holdout"]]
    clean = [r for r in rows if r["label"] == "clean" and not r["holdout"]]
    assert len(leaky) == 24 and len(clean) == 15
    assert sum(r["fixed"] for r in leaky) == 24
    hold = [r for r in rows if r["holdout"]]
    assert all(r["status"] == "not_needed" for r in hold if r["label"] == "clean")
    assert all(r["status"] == "not_needed" and not r["changed"] for r in clean)
    for r in leaky:
        if r["fixed"]:
            assert r["after"]["clean"] and r["after"]["avg_gross"] >= 0.05 and r["diff"]
    assert np.isclose(json.loads(p.read_text())["cost_usd"], 0.0)


def test_mock_demo_momentum_is_repaired_without_llm(panel):
    from agents.llm import MockLLM
    from agents.loop import run_audit_loop

    events = []
    tr = run_audit_loop("Buy top-decile 20-day momentum, rebalance weekly", panel, MockLLM(), max_iters=4,
                        on_event=events.append)
    assert tr.final_status == "clean" and [it.source for it in tr.iterations] == ["coder", "quick_fix"]
    assert [c["role"] for c in tr.calls] == ["coder", "summarizer"]      # no Auditor, no Coder rewrite
    assert tr.iterations[0].metrics["sharpe"] > 3 * tr.iterations[-1].metrics["sharpe"]
    assert {"quickfix_start", "quickfix_done"} <= {e["type"] for e in events}

"""Static AST rules for pandas default-axis / labelling traps (all medium = non-blocking hints)."""
import pytest

from audit.ast_scan import LOOKAHEAD_RULE_IDS, scan_code
from audit.leak_tests import evaluate_static
from bench.run_bench import load_strategies


def rules(code: str) -> set[str]:
    return {f.rule for f in scan_code(code)}


CASES = [
    # (rule, positive snippet, negative snippet)
    ("time_axis_rank",
     "p = mom.rank(pct=True)",
     "p = mom.rank(axis=1, pct=True)"),
    ("time_axis_rank",
     "p = mom.rank(axis=0, pct=True)",
     "p = mom.T.rank(pct=True).T"),
    ("time_axis_rank",
     "p = mom.rank(ascending=False)",
     "p = mom.rolling(252).rank(pct=True)"),
    ("time_axis_rank",
     "p = mom.rank(method='first')",
     "p = close.iloc[-1].rank()"),
    ("time_axis_demean",
     "d = mom.sub(mom.mean(), axis=1)",
     "d = mom.sub(mom.mean(axis=1), axis=0)"),
    ("time_axis_demean",
     "d = mom.div(mom.std())",
     "d = mom.div(mom.rolling(60).std())"),
    ("time_axis_demean",
     "z = (mom - mom.mean()) / mom.std()",
     "z = (mom - mom.expanding(min_periods=60).mean()) / mom.expanding(min_periods=60).std()"),
    ("time_axis_demean",
     "d = mom - mom.median()",
     "sharpe = book.mean() / book.std()"),
    ("full_sample_imputation",
     "m = mom.fillna(mom.mean())",
     "m = mom.fillna(0.0)"),
    ("full_sample_imputation",
     "m = mom.fillna(mom.median(axis=0))",
     "m = mom.fillna(mom.expanding().mean().shift(1))"),
    ("resample_start_label",
     "m = close.resample('MS').last()",
     "m = close.resample('ME').last()"),
    ("resample_start_label",
     "m = close.resample('QS').max()",
     "m = close.resample('MS').first()"),
    ("resample_start_label",
     "m = close.resample(rule='YS').agg('last')",
     "m = close.resample('MS', label='right', closed='right').last()"),
    ("resample_start_label",
     "m = close.resample('BMS').mean()",
     "m = close.resample('W-FRI').last()"),
    ("period_transform",
     "month = close.index.to_period('M')\nw = w.groupby(month).transform('last')",
     "month = close.index.to_period('M')\nw = w.groupby(month).transform('first')"),
    ("period_transform",
     "w = w.groupby(w.index.month).transform('max')",
     "w = w.groupby(level=0).transform('mean')"),
    ("period_transform",
     "w = w.groupby(pd.Grouper(freq='W')).transform('mean')",
     "month = close.index.to_period('M')\nw = w.groupby(month).cummax()"),
    ("reversed_time_window",
     "r = close.sort_index(ascending=False)\nma = r.rolling(20).mean().sort_index()",
     "r = close.sort_index()\nma = r.rolling(20).mean()"),
    ("reversed_time_window",
     "peak = close.iloc[::-1].cummax().iloc[::-1]",
     "back = close.iloc[::-1].iloc[::-1]\nx = back.rolling(5).mean()"),
    ("reversed_time_window",
     "vol = ret[::-1].expanding(min_periods=60).std()[::-1]",
     "vol = ret.expanding(min_periods=60).std()"),
    ("reversed_time_window",
     "x = close.sort_index(ascending=False).shift(1)",
     "top = scores.sort_values(ascending=False).cumsum()"),
    ("pooled_quantile",
     "long = mom.stack()\nb = pd.qcut(long, 5, labels=False)",
     "long = mom.stack()\nb = long.groupby(level=0).transform(lambda s: pd.qcut(s, 5, labels=False))"),
    ("pooled_quantile",
     "lvl = mom.stack().groupby(level=1).quantile(0.8)",
     "lvl = mom.stack().groupby(level=0).quantile(0.8)"),
    ("pooled_quantile",
     "thr = mom.stack().quantile(0.9)",
     "thr = mom.quantile(0.9, axis=1)"),
]


@pytest.mark.parametrize("rule,pos,neg", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_rule_positive_and_negative(rule, pos, neg):
    assert rule in rules(pos), (rule, pos, scan_code(pos))
    assert rule not in rules(neg), (rule, neg, scan_code(neg))


@pytest.mark.parametrize("rule,pos,neg", CASES[:1] + CASES[4:5] + CASES[10:11] + CASES[14:15] + CASES[17:18] + CASES[21:22])
def test_new_rules_are_medium_and_have_fix(rule, pos, neg):
    fs = [f for f in scan_code(pos) if f.rule == rule]
    assert fs and all(f.severity == "medium" and len(f.fix) > 30 for f in fs)
    assert rule in LOOKAHEAD_RULE_IDS
    assert evaluate_static(pos).passed            # hints never block on their own


def test_specific_rule_sits_next_to_generic_full_sample():
    fs = scan_code("z = (mom - mom.mean()) / mom.std()")
    assert {f.rule for f in fs} == {"time_axis_demean", "full_sample_statistic"}
    assert "full_sample_statistic" in rules("vol = ret.std()")
    assert "time_axis_demean" not in rules("vol = ret.std()")


STRATS = {s["id"]: s for s in load_strategies()}


@pytest.mark.parametrize("sid", [k for k, s in STRATS.items() if not s["leaky"]])
def test_clean_bench_strategies_get_no_new_hints(sid):
    new = {f.rule for f in scan_code(STRATS[sid]["code"])} & (LOOKAHEAD_RULE_IDS - {
        "negative_shift", "negative_pct_change", "negative_diff", "centered_window", "backward_fill",
        "interpolation", "full_sample_statistic", "survivor_universe"})
    assert not new, (sid, new)


@pytest.mark.parametrize("sid,rule", [
    ("L07_time_rank_full_history", "time_axis_rank"), ("L17_rank_default_axis", "time_axis_rank"),
    ("L04_fullsample_zscore", "time_axis_demean"), ("L18_sub_mean_axis_trap", "time_axis_demean"),
    ("L19_resample_ms_last", "resample_start_label"), ("L20_groupby_transform_last", "period_transform"),
    ("L08_reverse_cummax", "reversed_time_window"), ("L10_reversed_expanding_vol", "reversed_time_window"),
    ("L22_descending_sort_rolling", "reversed_time_window"), ("L23_fillna_full_mean", "full_sample_imputation"),
    ("L11_groupby_quantile_threshold", "pooled_quantile"), ("L24_pooled_qcut", "pooled_quantile"),
])
def test_leaky_bench_strategies_get_hint(sid, rule):
    assert rule in rules(STRATS[sid]["code"])

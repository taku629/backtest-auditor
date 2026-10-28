import pandas as pd
import pytest

from agents.mock_strategies import CLEAN, LEAKY
from audit.ast_scan import scan_code
from audit.leak_tests import in_process_executor, run_audit
from audit.overfitting import deflated_sharpe, expected_max_sharpe
from sandbox.runner import SandboxExecutor


def _fn(code):
    g = {}
    exec(code, g)
    return g["generate_signals"]


def _audit(code, data, sandbox=False, **kw):
    ex = SandboxExecutor(code) if sandbox else in_process_executor(_fn(code))
    return run_audit(data, ex, code, **kw)


def _by(rep):
    return {t.name: t for t in rep.tests}


@pytest.mark.parametrize("scenario", sorted(LEAKY))
def test_leaky_drafts_are_caught(panel, scenario):
    rep = _audit(LEAKY[scenario], panel)
    t = _by(rep)
    assert not rep.clean
    assert not t["future_perturbation"].passed
    assert t["future_perturbation"].severity == "critical"
    assert t["future_perturbation"].evidence["first_leaking_date"] is not None
    assert not t["static_scan"].passed


@pytest.mark.parametrize("scenario", sorted(CLEAN))
def test_clean_versions_pass(panel, scenario):
    rep = _audit(CLEAN[scenario], panel, n_trials=2)
    t = _by(rep)
    assert rep.clean, [x.to_dict() for x in rep.tests if x.blocking]
    for name in ("future_perturbation", "survivorship", "execution_delay", "static_scan"):
        assert t[name].passed, name


def test_leaky_momentum_looks_amazing_clean_realistic(panel):
    leaky = _audit(LEAKY["momentum"], panel).metrics["sharpe"]
    clean = _audit(CLEAN["momentum"], panel).metrics["sharpe"]
    assert leaky > 4.0 and 0.0 < clean < 2.0


def test_single_bar_lookahead_reports_cut_date(small_panel):
    code = '''
def generate_signals(data):
    c = data["close"]
    r = (c.shift(-1) / c - 1)
    top = (r.rank(axis=1, pct=True) >= 0.8).astype(float)
    return top.div(top.sum(axis=1), axis=0).fillna(0.0)
'''
    t = _by(_audit(code, small_panel))
    ev = t["future_perturbation"].evidence
    assert ev["n_leaking_cuts"] == ev["n_cuts"]
    for c in ev["cuts"]:
        assert c["latest_changed_date"] == c["cut_date"]   # leak shows up exactly at row t
    assert not t["execution_delay"].passed                  # edge collapses with an extra bar of delay
    assert not t["plausibility"].passed


def test_full_sample_normalisation_caught(small_panel):
    code = '''
def generate_signals(data):
    c = data["close"]
    z = (c - c.mean()) / c.std()
    top = (z.rank(axis=1, pct=True) <= 0.2).astype(float)
    return top.div(top.sum(axis=1), axis=0).fillna(0.0)
'''
    rep = _audit(code, small_panel)
    assert not _by(rep)["future_perturbation"].passed
    rules = {f["rule"] for f in _by(rep)["static_scan"].evidence["findings"]}
    assert "full_sample_statistic" in rules


def test_survivor_universe_caught(small_panel):
    code = '''
def generate_signals(data):
    c = data["close"].dropna(axis=1)
    w = c.notna().astype(float)
    return w.div(w.sum(axis=1), axis=0)
'''
    t = _by(_audit(code, small_panel))
    assert not t["survivorship"].passed
    assert t["survivorship"].evidence["counterfactual_rows_changed"] > 0


def test_weekly_resample_label_bug_caught(small_panel):
    code = '''
def generate_signals(data):
    c = data["close"]
    mom = c / c.shift(20) - 1
    top = (mom.rank(axis=1, pct=True) >= 0.8).astype(float)
    w = top.div(top.sum(axis=1), axis=0).fillna(0.0)
    wk = w.resample("W-FRI").last()
    return wk.reindex(c.index, method="bfill").fillna(0.0)
'''
    assert not _by(_audit(code, small_panel))["future_perturbation"].passed


def test_weight_before_listing_is_critical(small_panel):
    code = '''
def generate_signals(data):
    c = data["close"]
    w = c.notna().astype(float) * 0 + 1.0
    return w.div(w.sum(axis=1), axis=0)
'''
    t = _by(_audit(code, small_panel))["survivorship"]
    assert not t.passed and t.severity == "critical"


def test_sandbox_executor_matches_in_process(small_panel):
    a = _audit(CLEAN["momentum"], small_panel)
    b = _audit(CLEAN["momentum"], small_panel, sandbox=True)
    assert a.clean == b.clean
    assert abs(a.metrics["sharpe"] - b.metrics["sharpe"]) < 1e-9


def test_results_are_json_serialisable(small_panel):
    import json
    json.dumps(_audit(LEAKY["momentum"], small_panel).to_dict())


def test_ast_scan_rules():
    code = '''
def generate_signals(data):
    c = data["close"]
    a = c.shift(-2)
    b = c.pct_change(periods=-1)
    d = c.rolling(5, center=True).mean()
    e = c.bfill()
    f = c.fillna(method="backfill")
    g = c.dropna(axis="columns")
    h = c.rolling(5).mean() + c.mean(axis=1).values[:, None]
    return a
'''
    rules = {f.rule for f in scan_code(code)}
    assert {"negative_shift", "negative_pct_change", "centered_window", "backward_fill",
            "survivor_universe"} <= rules
    assert "full_sample_statistic" not in rules  # rolling + cross-sectional stats are fine


def test_dsr_properties():
    import numpy as np
    rng = np.random.default_rng(0)
    r = 0.001 + 0.01 * rng.standard_normal(1500)  # daily SR ~0.1 -> annual ~1.6
    one = deflated_sharpe(r, 1)
    many = deflated_sharpe(r, 100)
    assert one["dsr"] == pytest.approx(one["psr"])
    assert many["dsr"] < one["dsr"]
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01) > 0
    noise = deflated_sharpe(0.01 * rng.standard_normal(1500), 50)
    assert noise["dsr"] < 0.5

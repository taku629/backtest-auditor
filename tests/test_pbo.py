"""Probability of Backtest Overfitting (audit/pbo.py): CSCV maths, grid building, end to end in the sandbox."""
import numpy as np
import pytest

from agents.mock_strategies import CLEAN, LEAKY
from audit.leak_tests import evaluate_pbo, in_process_executor, run_audit
from audit.pbo import apply_values, build_grid, combined_module, compute_pbo, cscv, find_params

NOISE_MINED = '''import numpy as np
import pandas as pd

SALT = 40


def generate_signals(data):
    """Noise-mined: a pseudo-random long/short basket that reshuffles with SALT every 21 days."""
    close = data["close"]
    listed = close.notna()
    T, N = close.shape
    block = np.arange(T) // 21
    code = (block[:, None] * 7919 + np.arange(N)[None, :] * 104729 + SALT * 15485863) % 1000
    w = pd.DataFrame(np.where(code < 100, 1.0, np.where(code > 900, -1.0, 0.0)), index=close.index,
                     columns=close.columns)
    w = w.where(listed, 0.0)
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
'''


def _noise(seed, T=1500, N=16):
    return np.random.default_rng(seed).standard_normal((T, N)) * 0.01


def test_cscv_noise_grid_has_high_pbo():
    # 16 variants of pure noise: choosing the in-sample best is worthless out of sample
    pbos = [cscv(_noise(s))["pbo"] for s in range(8)]
    assert np.mean(pbos) > 0.55


def test_cscv_robust_signal_has_low_pbo():
    # variants of one genuine signal: highly correlated, with a persistent ordering of their edges
    pbos = []
    for s in range(8):
        rng = np.random.default_rng(100 + s)
        common = rng.standard_normal((1500, 1)) * 0.01
        idio = rng.standard_normal((1500, 16)) * 0.002
        pbos.append(cscv(common + idio + np.linspace(0.0, 0.0008, 16))["pbo"])
    assert max(pbos) < 0.1


def test_cscv_outputs_shape_and_ranges():
    r = cscv(_noise(1), n_blocks=8)
    assert r["n_splits"] == 70 and r["n_blocks"] == 8 and r["n_variants"] == 16
    assert len(r["logits"]) == 70 and 0 <= r["pbo"] <= 1 and 0 <= r["prob_oos_loss"] <= 1
    assert all(0 < w < 1 for w in r["oos_relative_rank"])
    with pytest.raises(ValueError):
        cscv(_noise(1), n_blocks=7)


def test_find_params_explicit_beats_inline():
    code = "LOOKBACK = 20\nTOP = 0.9\n\ndef generate_signals(data, z=2.0):\n    return data['close'].rolling(5).mean()\n"
    names = [p.name for p in find_params(code)]
    assert names == ["LOOKBACK", "TOP", "z"]


def test_find_params_inline_windows_and_thresholds():
    ps = find_params(CLEAN["reversal"])
    assert [p.name for p in ps] == ["window 10", "window 120", "threshold 5"]
    assert find_params("def generate_signals(data):\n    return data['close'].shift(1) * 0\n") == []


def test_grid_and_rewrite_keep_min_periods_valid():
    code = CLEAN["reversal"]
    ps = find_params(code)
    grid = build_grid(ps)
    assert 4 <= len(grid) <= 16 and grid[0] == [p.value for p in ps]
    v = apply_values(code, ps, [5.0, 240.0, 3.0])
    assert "rolling(5, min_periods=5)" in v and "rolling(240, min_periods=120)" in v and "rank <= 3" in v
    v2 = apply_values(code, ps, [10.0, 60.0, 5.0])
    assert "rolling(60, min_periods=30)" in v2
    # probability-like threshold: the distance to 1 is scaled ("top 10%" -> "top 5%..20%")
    p = find_params(CLEAN["momentum"])[1]
    assert p.vary(2.0) == pytest.approx(0.8) and p.vary(0.5) == pytest.approx(0.95)


def test_combined_module_compiles_and_passes_policy():
    from sandbox.runner import check_policy
    src = combined_module([CLEAN["momentum"], CLEAN["lowvol"]])
    assert src is not None and check_policy(src) == []


def test_noise_mined_strategy_flagged_end_to_end(panel):
    r = compute_pbo(NOISE_MINED, panel)
    assert r["status"] == "ok" and r["params"][0]["name"] == "SALT"
    assert r["pbo"] > 0.5
    t = evaluate_pbo(NOISE_MINED, panel)
    assert not t.passed and t.severity == "medium" and not t.blocking


def test_robust_momentum_low_pbo_end_to_end(panel):
    r = compute_pbo(CLEAN["momentum"], panel)
    assert r["status"] == "ok" and r["n_variants"] >= 8
    assert r["pbo"] < 0.3
    assert r["seconds"] < 20


def test_skipped_when_no_params(panel):
    code = ("import numpy as np\n\ndef generate_signals(data):\n    c = data['close']\n"
            "    w = c.notna().astype(float)\n    return w.div(w.sum(axis=1), axis=0).fillna(0.0)\n")
    t = evaluate_pbo(code, panel)
    assert t.passed and t.severity == "info" and t.evidence["status"] == "skipped"


def test_user_param_grid(panel):
    code = NOISE_MINED.replace("SALT = 40", "SALT = 40\nUNUSED = 3")
    r = compute_pbo(code, panel, param_grid={"SALT": [10, 20, 30, 40, 50, 60]})
    assert r["status"] == "ok" and r["params"][0]["values"] == [10, 20, 30, 40, 50, 60]
    assert compute_pbo(code, panel, param_grid={"NOPE": [1, 2]})["status"] == "skipped"


def test_run_audit_integration_skips_on_leak(panel):
    g = {}
    exec(LEAKY["momentum"], g)
    rep = run_audit(panel, in_process_executor(g["generate_signals"]), LEAKY["momentum"], pbo=True)
    t = {x.name: x for x in rep.tests}["pbo"]
    assert t.evidence["status"] == "skipped" and "blocking" in t.evidence["reason"]


def test_run_audit_integration_clean_runs_pbo(panel):
    g = {}
    exec(CLEAN["momentum"], g)
    rep = run_audit(panel, in_process_executor(g["generate_signals"]), CLEAN["momentum"], pbo=True)
    t = {x.name: x for x in rep.tests}["pbo"]
    assert t.evidence["status"] == "ok" and rep.clean
    assert "pbo" in [x["name"] for x in rep.to_dict()["tests"]]

"""Strategy format adapters (audit/adapters.py + sandbox/adapter_stubs.py)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit.adapters import WRAPPER_MARK, AdapterError, adapt, detect
from audit.leak_tests import run_audit
from sandbox import adapter_stubs as stubs
from sandbox.runner import SandboxExecutor, check_policy, run_strategy

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def _code(name):
    return (EX / name).read_text(encoding="utf-8")


def _by_name(rep):
    return {t.name: t for t in rep.tests}


# ------------------------------------------------------------------------------------------- detection
@pytest.mark.parametrize("code,fmt,entry", [
    ("import pandas as pd\ndef generate_signals(data):\n    return data['close'] * 0\n", "signals", "generate_signals"),
    ("def strategy(df):\n    return df.pct_change(20) > 0\n", "function", "strategy"),
    ("def signals(close):\n    return close * 0\n", "function", "signals"),
    ("def compute_weights(prices, n=20):\n    return prices * 0\n", "function", "compute_weights"),
    ("def helper(x):\n    return x\n\ndef my_rule(close, volume):\n    return helper(close) * 0\n", "function",
     "my_rule"),
    ("import vectorbt as vbt\ndef get_signals(close):\n    return close > 0, close < 0\n", "vectorbt", "get_signals"),
    ("import vectorbt as vbt\nentries = close > 1\nexits = close < 1\n"
     "pf = vbt.Portfolio.from_signals(close, entries, exits)\n", "vectorbt", "<module>"),
    ("import backtrader as bt\nclass A(bt.Strategy):\n    def next(self):\n        pass\n", "backtrader", "A"),
    ("from backtrader import Strategy\nclass B(Strategy):\n    pass\n", "backtrader", "B"),
])
def test_detect(code, fmt, entry):
    d = detect(code)
    assert (d["format"], d["entry"]) == (fmt, entry)


def test_backtrader_picks_addstrategy_class():
    code = ("import backtrader as bt\nclass A(bt.Strategy):\n    pass\nclass B(bt.Strategy):\n    pass\n"
            "cerebro = bt.Cerebro()\ncerebro.addstrategy(A)\n")
    assert detect(code)["entry"] == "A"


def test_unknown_and_forced_format_errors():
    with pytest.raises(AdapterError, match="Accepted formats"):
        adapt("x = 1\n")
    with pytest.raises(AdapterError):
        adapt(_code("vectorbt_clean.py"), "backtrader")
    assert adapt("def generate_signals(data):\n    return data['close']*0\n").format == "signals"


def test_wrapper_appended_keeps_user_lines_and_policy():
    src = _code("backtrader_clean.py")
    a = adapt(src)
    assert a.code.startswith(src.rstrip("\n")) and WRAPPER_MARK in a.code and a.line_offset == 0
    assert check_policy(a.code) == []
    assert a.strip(a.code).rstrip("\n") == src.rstrip("\n")


def test_policy_still_blocks_other_imports():
    a = adapt("import vectorbt as vbt\nimport yfinance as yf\ndef strategy(close):\n    return close*0\n")
    assert any("yfinance" in v for v in check_policy(a.code))


def test_module_wrap_without_blank_line_shifts_and_disables_localizer():
    a = adapt("import vectorbt as vbt\nentries = close > close.shift(1)\nexits = close < close.shift(1)\n")
    assert a.line_offset == 1 and not a.localizable


# ------------------------------------------------------------------------------------------- conversions
def _close(vals):
    idx = pd.bdate_range("2024-01-01", periods=len(vals))
    return pd.DataFrame(vals, index=idx, columns=[f"T{i}" for i in range(len(vals[0]))], dtype=float)


def test_signals_to_equal_weights():
    close = _close([[1, 1, 1], [1, 1, np.nan], [1, 1, 1]])
    s = pd.DataFrame([[1, -1, 1], [1, 0, 1], [0, 0, 0]], index=close.index, columns=close.columns)
    w = stubs.signals_to_weights(s, close)
    np.testing.assert_allclose(w.to_numpy(), [[1 / 3, -1 / 3, 1 / 3], [1, 0, 0], [0, 0, 0]])


def test_entries_exits_state_machine():
    close = _close([[1]] * 7)
    e = pd.DataFrame([0, 1, 1, 0, 0, 1, 1], index=close.index, columns=close.columns).astype(bool)
    x = pd.DataFrame([0, 0, 0, 1, 1, 1, 0], index=close.index, columns=close.columns).astype(bool)
    pos = stubs.positions_from_events(e, x, close)
    # entry, entry while long (no-op), exit, exit while flat (no-op), entry+exit same bar (cancel), entry
    assert pos.iloc[:, 0].tolist() == [0, 1, 1, 0, 0, 0, 1]
    both = stubs.entries_exits_to_weights(e, x, close, direction="both")
    assert both.iloc[:, 0].tolist() == [0, 1, 1, -1, -1, -1, 1]


def test_to_weights_kinds_in_sandbox(small_panel):
    close = small_panel["close"]
    cases = {
        "weights": "def compute_weights(prices):\n    return prices.notna() * 0.01\n",
        "bool": "def strategy(df):\n    return df.pct_change(20) > 0\n",
        "pm1": "import numpy as np\ndef signals(close):\n    return np.sign(close.pct_change(20)).fillna(0)\n",
        "tuple": "def get_signals(close):\n    m = close.rolling(20).mean()\n    return close > m, close < m\n",
    }
    for k, code in cases.items():
        r = run_strategy(adapt(code).code, [small_panel])
        assert r.ok, (k, r.error)
        w = r.outputs[0]
        gross = w.abs().sum(axis=1)
        if k == "weights":
            assert np.isclose(w.iloc[-1].sum(), 0.01 * close.iloc[-1].notna().sum())
        else:
            assert gross.max() <= 1 + 1e-9 and gross.iloc[-200:].gt(0.99).mean() > 0.5, k
        assert (w[close.isna()].fillna(0) == 0).all().all()


def test_vbt_stub_indicators_and_unsupported(small_panel):
    ok = ("import vectorbt as vbt\n\nclose = vbt.YFData.download('SPY').get('Close')\n"
          "fast = vbt.MA.run(close, 5)\nslow = vbt.MA.run(close, 20, ewm=True)\n"
          "entries = fast.ma_crossed_above(slow)\nexits = fast.ma_crossed_below(slow)\n"
          "pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=1000)\nprint(pf.stats())\n")
    r = run_strategy(adapt(ok).code, [small_panel])
    assert r.ok, r.error
    assert r.outputs[0].abs().sum(axis=1).max() > 0.99
    assert any("init_cash" in n for n in r.notes)
    bad = ok.replace("init_cash=1000", "sl_stop=0.05")
    r = run_strategy(adapt(bad).code, [small_panel])
    assert not r.ok and "sl_stop" in r.error and "not supported" in r.error
    sweep = ok.replace("vbt.MA.run(close, 5)", "vbt.MA.run(close, [5, 10])")
    r = run_strategy(adapt(sweep).code, [small_panel])
    assert not r.ok and "parameter sweep" in r.error


# ------------------------------------------------------------------------------------------- backtrader stub
BT_HEAD = "import backtrader as bt\n\n"


def _bt_weights(body, panel):
    r = run_strategy(adapt(BT_HEAD + body).code, [panel])
    return r


def test_backtrader_line_indexing_reads_next_bar(small_panel):
    # buy when tomorrow's close is higher: the target on day t must equal sign(close[t+1] - close[t])
    body = ("class Peek(bt.Strategy):\n    def next(self):\n"
            "        if self.data.close[1] > self.data.close[0]:\n            self.order_target_size(target=1)\n"
            "        else:\n            self.order_target_size(target=0)\n")
    r = _bt_weights(body, small_panel)
    assert r.ok, r.error
    c = small_panel["close"].iloc[:, 0].dropna()
    w = r.outputs[0].iloc[:, 0].reindex(c.index)
    expect = (c.shift(-1) > c)
    assert ((w > 0) == expect).iloc[:-1].all()


def test_backtrader_sma_minperiod_and_params(small_panel):
    body = ("class S(bt.Strategy):\n    params = dict(period=15)\n    def __init__(self):\n"
            "        self.sma = bt.ind.SMA(self.data, period=self.p.period)\n        self.n = 0\n"
            "    def next(self):\n        if self.n == 0:\n            assert len(self) == self.p.period\n"
            "            assert self.sma[0] == self.sma[0]\n        self.n += 1\n"
            "        if self.data.close[0] > self.sma[0]:\n            self.buy()\n")
    r = _bt_weights(body, small_panel)
    assert r.ok, r.error


def test_backtrader_notify_order_and_position(small_panel):
    body = ("class S(bt.Strategy):\n    def __init__(self):\n        self.order = None\n        self.fills = 0\n"
            "    def notify_order(self, order):\n        if order.status in [order.Completed]:\n"
            "            self.fills += 1\n            self.order = None\n"
            "    def next(self):\n        if self.order:\n            return\n"
            "        if not self.position:\n            self.order = self.buy()\n"
            "        elif len(self) % 10 == 0:\n            self.order = self.close()\n")
    r = _bt_weights(body, small_panel)
    assert r.ok, r.error
    w = r.outputs[0]
    assert (w > 0).any().any() and (w == 0).sum().sum() > 0      # positions open and close repeatedly


@pytest.mark.parametrize("call,msg", [
    ("self.broker.getvalue()", "broker"),
    ("bt.indicators.MACD(self.data)", "bt.indicators.MACD"),
    ("self.buy(exectype=bt.Order.Limit, price=1.0)", "exectype"),
    ("self.data.adjclose[0]", "adjclose"),
])
def test_backtrader_unsupported_fails_clearly(small_panel, call, msg):
    body = f"class S(bt.Strategy):\n    def next(self):\n        x = {call}\n"
    r = _bt_weights(body, small_panel)
    assert not r.ok and msg in r.error and "not supported" in r.error


def test_backtrader_static_rule_and_examples():
    from audit.ast_scan import scan_code
    leaky = [f for f in scan_code(_code("backtrader_leaky.py")) if f.rule == "future_bar_index"]
    assert leaky and leaky[0].severity == "critical"
    assert not [f for f in scan_code(_code("backtrader_clean.py")) if f.rule == "future_bar_index"]
    # self.datas[1] is a second feed, not the future
    assert not scan_code(BT_HEAD + "class S(bt.Strategy):\n    def next(self):\n        d = self.datas[1]\n")


# ------------------------------------------------------------------------------------------- full audits
@pytest.mark.parametrize("name,leaky", [("backtrader_leaky.py", True), ("backtrader_clean.py", False),
                                        ("vectorbt_leaky.py", True), ("vectorbt_clean.py", False)])
def test_examples_audit(panel, name, leaky):
    a = adapt(_code(name))
    t0 = time.time()
    rep = run_audit(panel, SandboxExecutor(a.code), a.code, pbo=False)
    by = _by_name(rep)
    assert by["future_perturbation"].passed is (not leaky)
    assert by["survivorship"].passed
    if leaky:
        lz = by["future_perturbation"].evidence["localized"]
        line = lz["entry"]["line"]
        assert lz["found"] and ("[1]" in a.code.splitlines()[line - 1] or "shift(-1)" in a.code.splitlines()[line - 1])
    else:
        assert rep.clean
    assert time.time() - t0 < 60


def test_backtrader_replay_speed_full_panel(panel):
    a = adapt(_code("backtrader_clean.py"))
    t0 = time.time()
    r = run_strategy(a.code, [panel] * 3)
    assert r.ok, r.error
    assert time.time() - t0 < 30      # ~0.3 s per 50 x 1512 dataset locally


def test_quick_fix_backtrader_leak(panel):
    from audit.quickfix import quick_fix
    a = adapt(_code("backtrader_leaky.py"))
    qf = quick_fix(a.code, panel)
    assert qf["status"] == "fixed" and qf["rule"] == "future_bar_index"
    fixed = a.strip(qf["code"])
    assert WRAPPER_MARK not in fixed and "self.data.close[0]" in fixed


# ------------------------------------------------------------------------------------------- CLI
def test_cli_auto_detect_json(tmp_path, capsys):
    import audit_cli
    rc = audit_cli.main([str(EX / "vectorbt_leaky.py"), str(EX / "backtrader_clean.py"), "--format", "json",
                         "--no-pbo"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    files = {Path(f["path"]).name: f for f in out["files"]}
    assert files["vectorbt_leaky.py"]["adapter"]["format"] == "vectorbt"
    assert not files["vectorbt_leaky.py"]["passed"] and files["backtrader_clean.py"]["passed"]
    assert files["backtrader_clean.py"]["adapter"]["format"] == "backtrader"


def test_cli_forced_wrong_format(capsys):
    import audit_cli
    rc = audit_cli.main([str(EX / "vectorbt_clean.py"), "--strategy-format", "backtrader", "--no-pbo"])
    assert rc == 1 and "Could not adapt" in capsys.readouterr().out


def test_agent_loop_on_adapted_backtrader(panel):
    """The app path: pasted backtrader code -> adapter wrapper -> loop; Quick fix repairs it without an LLM."""
    from agents.llm import make_llm
    from agents.loop import run_audit_loop
    a = adapt(_code("backtrader_leaky.py"))
    trace = run_audit_loop("SMA trend filter", panel, make_llm(mock=True), max_iters=3, initial_code=a.code,
                           summarize=False, quick_fix=True)
    assert trace.final_status == "clean"
    assert trace.iterations[0].user_code and trace.iterations[-1].source == "quick_fix"

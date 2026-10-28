import numpy as np
import pytest

from sandbox.runner import check_policy, run_strategy

OK = '''
import numpy as np
import pandas as pd

def generate_signals(data):
    c = data["close"]
    return c.notna().astype(float) / c.shape[1]
'''


@pytest.fixture(scope="module")
def tiny():
    from data import generate_panel
    return generate_panel(n_tickers=6, years=1, n_delisted=1, n_ipo=1)


def test_runs_and_returns_aligned_frames(tiny):
    r = run_strategy(OK, [tiny, tiny])
    assert r.ok, r.error
    assert len(r.outputs) == 2
    assert r.outputs[0].shape == tiny["close"].shape
    assert np.allclose(r.outputs[0].to_numpy(), (tiny["close"].notna() / 6).to_numpy())


@pytest.mark.parametrize("code,needle", [
    ("import os\ndef generate_signals(data):\n    return data['close']", "import 'os'"),
    ("import subprocess\ndef generate_signals(data):\n    return data['close']", "import 'subprocess'"),
    ("def generate_signals(data):\n    open('x','w')\n    return data['close']", "'open'"),
    ("def generate_signals(data):\n    return ().__class__", "__class__"),
    ("def generate_signals(data):\n    data['close'].to_csv('x.csv')\n    return data['close']", "to_csv"),
    ("def generate_signals(data):\n    return eval('1')", "'eval'"),
    ("def f(data):\n    return data['close']", "generate_signals"),
])
def test_policy_blocks(code, needle):
    v = check_policy(code)
    assert v and any(needle in x for x in v), v


@pytest.mark.parametrize("code,needle", [
    ("def generate_signals(data):\n    open('x','w').write('a')\n    return data['close']", "NameError"),
    ("def generate_signals(data):\n    import os\n    return data['close']", "ImportError"),
    ("def generate_signals(data):\n    data['close'].to_csv('x.csv')\n    return data['close']", "File too large"),
])
def test_runtime_layer_holds_without_static_policy(tiny, code, needle):
    r = run_strategy(code, [tiny], timeout=20, enforce_policy=False)
    assert not r.ok and needle in r.error, r.error


def test_api_key_not_visible_in_child(tiny, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "super-secret")
    code = ("import pandas as pd\ndef generate_signals(data):\n"
            "    print(pd.io.common.os.environ.get('NEBIUS_API_KEY'))\n    return data['close'] * 0")
    r = run_strategy(code, [tiny], timeout=20, enforce_policy=False)
    assert r.ok and "super-secret" not in r.stdout


def test_timeout(tiny):
    r = run_strategy("def generate_signals(data):\n    while True:\n        pass", [tiny], timeout=3)
    assert not r.ok and "Timeout" in r.error


def test_runtime_error_reports_line(tiny):
    r = run_strategy("def generate_signals(data):\n    return data['nope']", [tiny], timeout=20)
    assert not r.ok and "KeyError" in r.error and "line 2" in r.error


def test_series_output_rejected(tiny):
    r = run_strategy("def generate_signals(data):\n    return data['close'].iloc[:, 0]", [tiny], timeout=20)
    assert not r.ok and "DataFrame" in r.error

"""Jupyter notebook support: notebook.extract_strategy and audit_cli.py on .ipynb files."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import audit_cli
from notebook import extract_strategy, locate

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "leaky_research.ipynb"


def make_nb(cells: list[tuple[str, str]]) -> str:
    """cells: [("code"|"markdown", source)] -> nbformat-4 JSON (source stored as a list of lines, as Jupyter does)."""
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return json.dumps({"cells": out, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


RESEARCH = make_nb([
    ("markdown", "# Research"),                                                    # cell 1
    ("code", "import numpy as np, matplotlib.pyplot as plt\nimport pandas as pd\n%matplotlib inline"),  # cell 2
    ("code", "prices = pd.read_csv('prices.csv')\nprices.head()"),                 # cell 3
    ("code", "!pip install yfinance\nprices.describe()"),                          # cell 4
    ("code", "def broken(:\n    pass"),                                            # cell 5 (does not parse)
    ("code", "WINDOW = 20\nUNUSED = 99\n\n\n# trailing mean\ndef smooth(x):\n    return x.rolling(WINDOW).mean()"),  # cell 6
    ("markdown", "## Signal"),                                                     # cell 7
    ("code", "def generate_signals(data):\n    close = data['close']\n    s = smooth(close)\n"
             "    return (s > close).astype(float).div(10).replace(np.nan, 0.0)"),  # cell 8
    ("code", "%%time\nw = generate_signals({'close': prices})"),                   # cell 9
    ("code", "%%bash\nls -la"),                                                    # cell 10
])

CLEAN_NB = make_nb([
    ("markdown", "# Point-in-time momentum"),
    ("code", "import numpy as np\nimport pandas as pd\n%load_ext autoreload"),
    ("code", "LOOKBACK = 20\nTOP = 0.9\n\n\ndef trailing_momentum(close, n=LOOKBACK):\n    return close / close.shift(n) - 1"),
    ("code", "def generate_signals(data):\n"
             "    \"\"\"Top-decile 20-day momentum, point-in-time universe, weekly rebalance.\"\"\"\n"
             "    close = data['close']\n"
             "    listed = close.notna()\n"
             "    rank = trailing_momentum(close).where(listed).rank(axis=1, pct=True)\n"
             "    top = ((rank >= TOP) & listed).astype(float)\n"
             "    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0)\n"
             "    week = pd.Series(close.index.to_period('W'), index=close.index)\n"
             "    weights.loc[~(week != week.shift(1)).to_numpy()] = np.nan\n"
             "    return weights.ffill().fillna(0.0).where(listed, 0.0)"),
    ("code", "w = generate_signals({'close': None})  # exploration"),
])


@pytest.fixture(autouse=True)
def _no_ci_env(monkeypatch):
    for k in ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY", "GITHUB_WORKSPACE", "NEBIUS_API_KEY"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------- extraction
def test_extracts_helpers_constants_and_only_needed_imports():
    r = extract_strategy(RESEARCH, "research.ipynb")
    assert r["ok"] and r["error"] is None, r["error"]
    code = r["code"]
    ast.parse(code)
    names = {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    assert names == {"generate_signals", "smooth"}
    assert "WINDOW = 20" in code and "UNUSED" not in code
    assert "import numpy as np" in code and "matplotlib" not in code and "import pandas" not in code
    assert "read_csv" not in code and "describe" not in code and "%" not in code and "!" not in code
    assert "# trailing mean" in code                      # comment right above a helper is kept
    assert r["kept"] == ["np", "WINDOW", "smooth", "generate_signals"]
    assert r["cell_count"] == 10 and r["code_cell_count"] == 8 and r["entry_cell"] == 8
    assert r["is_notebook"] is True


def test_magics_and_unparsable_cells_are_skipped_and_recorded():
    r = extract_strategy(RESEARCH.encode(), "research.ipynb")      # bytes work too
    sk = {(s["cell"], s["line"], s["kind"]) for s in r["skipped"]}
    assert (2, 3, "magic") in sk            # %matplotlib inline
    assert (4, 1, "magic") in sk            # !pip install
    assert (9, 1, "magic") in sk            # %%time line dropped, cell body still parsed
    assert (10, 1, "cell_magic") in sk      # %%bash cell skipped entirely
    assert (5, 1, "syntax_error") in sk     # def broken(:
    assert any("did not parse" in w for w in r["warnings"])


def test_line_map_points_at_cell_and_line():
    r = extract_strategy(RESEARCH, "research.ipynb")
    lines = r["code"].splitlines()
    assert locate(r, 1) is None                                    # header comment
    for i, text in enumerate(lines, 1):
        if text.strip() == "import numpy as np":
            assert locate(r, i) == (2, 1)                          # rebuilt from `import numpy as np, matplotlib...`
        if text.strip().startswith("return x.rolling(WINDOW)"):
            assert locate(r, i) == (6, 7)
        if text.startswith("def generate_signals"):
            assert locate(r, i) == (8, 1)
        if "astype(float).div(10)" in text:
            assert locate(r, i) == (8, 4)
    # every mapped line reproduces the source line of that cell
    cells = json.loads(RESEARCH)["cells"]
    for i, (cell, ln) in r["line_map"].items():
        src = "".join(cells[cell - 1]["source"]).splitlines()[ln - 1]
        assert lines[i - 1].strip() == src.strip() or lines[i - 1].startswith("import numpy")


def test_example_notebook_extraction():
    r = extract_strategy(EXAMPLE.read_text(), EXAMPLE.name)
    assert r["ok"], r["error"]
    assert {"zscore", "momentum", "generate_signals", "LOOKBACK", "TOP_PCT", "np"} == set(r["kept"])
    assert "matplotlib" not in r["code"] and "read_csv" not in r["code"] and "generate_panel" not in r["code"]
    shift_line = next(i for i, t in enumerate(r["code"].splitlines(), 1) if "shift(-1)" in t)
    assert locate(r, shift_line) == (10, 4)


def test_missing_generate_signals_explains_the_contract():
    nb = make_nb([("code", "import pandas as pd\n\ndef my_signal(prices):\n    return prices.rank(axis=1)")])
    r = extract_strategy(nb, "idea.ipynb")
    assert not r["ok"] and r["code"] == ""
    assert "generate_signals" in r["error"] and "my_signal(prices)" in r["error"]
    assert "numpy, pandas and math" in r["error"] and "cell 1, line 3" in r["error"]


def test_invalid_notebook_json_is_an_error_not_an_exception():
    r = extract_strategy("{not json", "x.ipynb")
    assert not r["ok"] and "not valid notebook JSON" in r["error"]


def test_py_file_with_percent_cells_and_plain_py():
    src = ("# %%\nimport numpy as np\nimport os\n\n# %% [markdown]\n# notes\n\n# %%\nK = 3\n\n"
           "def generate_signals(data):\n    return data['close'] * 0 + K + np.nan\n")
    r = extract_strategy(src, "s.py")
    assert r["ok"] and "import os" not in r["code"] and "K = 3" in r["code"]
    gen = next(i for i, t in enumerate(r["code"].splitlines(), 1) if t.startswith("def generate_signals"))
    assert locate(r, gen) == (3, 4)
    plain = extract_strategy("import numpy as np\n\n\ndef generate_signals(data):\n    return data['close']\n", "p.py")
    gen = next(i for i, t in enumerate(plain["code"].splitlines(), 1) if t.startswith("def generate_signals"))
    assert locate(plain, gen) == (None, 4) and plain["cell_count"] == 1


def test_warns_about_undefined_and_conditionally_defined_names():
    nb = make_nb([("code", "if True:\n    SCALE = 2"),
                  ("code", "def generate_signals(data):\n    return data['close'] * SCALE * MISSING")])
    r = extract_strategy(nb, "w.ipynb")
    assert r["ok"]
    assert any("SCALE" in w and "conditional" in w for w in r["warnings"])
    assert any("MISSING" in w and "never defined" in w for w in r["warnings"])


# ---------------------------------------------------------------- CLI
def test_cli_on_example_notebook_exits_1_with_cell_references():
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY")}
    p = subprocess.run([sys.executable, str(ROOT / "audit_cli.py"), str(EXAMPLE)], capture_output=True, text=True,
                       cwd=ROOT, env=env, timeout=180)
    assert p.returncode == 1, p.stderr
    out = p.stdout
    assert "examples/leaky_research.ipynb  ->  FAIL" in out
    assert "[FAIL] future_perturbation" in out
    assert "examples/leaky_research.ipynb (cell 10, line 4)" in out and "shift(-1)" in out
    assert "(cell 9, line 7)" in out                                  # full-sample z-score in the helper
    assert "generate_signals in cell 11" in out and "'%matplotlib inline' (magic)" in out
    if "leak enters" in out:                                          # dynamic localization (audit/localize.py)
        assert "leak enters at cell 10, line 4" in out


def test_cli_notebook_sarif_and_annotations(tmp_path, capsys):
    sarif = tmp_path / "nb.sarif"
    code = audit_cli.main([str(EXAMPLE), "--format", "json", "--sarif-out", str(sarif), "--annotations", "always"])
    assert code == 1
    cap = capsys.readouterr()
    rep = json.loads(cap.out)
    f = rep["files"][0]
    assert f["notebook"]["entry_cell"] == 11 and "def generate_signals" in f["notebook"]["extracted_code"]
    assert any(x.get("cell") == 10 and x.get("cell_line") == 4 for x in f["static_findings"])
    s = json.loads(sarif.read_text())
    results = s["runs"][0]["results"]
    assert results
    for res in results:
        loc = res["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"].endswith("leaky_research.ipynb")
    shift = [res for res in results if res["properties"].get("notebookCell") == 10]
    assert shift and shift[0]["message"]["text"].startswith("[cell 10 line 4]")
    assert "::error file=" in cap.err and "leaky_research.ipynb" in cap.err and "[cell 10 line 4]" in cap.err


def test_clean_notebook_passes(tmp_path, capsys):
    p = tmp_path / "clean_research.ipynb"
    p.write_text(CLEAN_NB)
    assert audit_cli.main([str(p), "--fail-on", "high"]) == 0
    out = capsys.readouterr().out
    assert "->  PASS" in out and "generate_signals in cell 4" in out


def test_notebook_without_strategy_fails_with_contract(tmp_path, capsys):
    p = tmp_path / "eda.ipynb"
    p.write_text(make_nb([("code", "import pandas as pd\nx = 1")]))
    assert audit_cli.main([str(p), "--format", "markdown"]) == 1
    out = capsys.readouterr().out
    assert "notebook" in out and "generate_signals" in out


def test_glob_picks_up_notebooks_but_not_checkpoints(tmp_path, capsys):
    (tmp_path / "clean.ipynb").write_text(CLEAN_NB)
    ck = tmp_path / ".ipynb_checkpoints"
    ck.mkdir()
    (ck / "clean-checkpoint.ipynb").write_text(CLEAN_NB)
    files, missing = audit_cli.expand_paths([str(tmp_path / "**" / "*.ipynb")])
    assert [f.name for f in files] == ["clean.ipynb"] and not missing


def test_annotate_tests_adds_cell_refs():
    from notebook import annotate_tests, where
    lm = {12: (7, 3), 13: (7, 4)}
    tests = [{"name": "future_perturbation", "evidence": {"localized": {"found": True, "entry": {"line": 12},
                                                                      "chain": [{"line": 13}], "per_line": []}}},
             {"name": "static_scan", "evidence": {"findings": [{"line": 13, "rule": "x"}, {"line": 1}]}}]
    annotate_tests(tests, lm)
    assert tests[0]["evidence"]["localized"]["entry"]["where"] == "cell 7, line 3"
    assert tests[0]["evidence"]["localized"]["chain"][0]["cell"] == 7
    assert tests[1]["evidence"]["findings"][0]["where"] == "cell 7, line 4"
    assert "where" not in tests[1]["evidence"]["findings"][1]          # unmapped header line
    assert where({"line_map": {"5": [2, 1]}}, 5) == "cell 2, line 1"  # JSON round-trip keys
    assert where({"line_map": {5: (None, 5)}}, 5) == ""               # plain .py without cells


def test_escape_dollars_outside_code():
    from audit.mdsafe import escape_dollars
    s = "cost $0.003 vs $0 `${{ secrets.X }}`\n```\n$ python a.py\n```\nalready \\$1"
    out = escape_dollars(s)
    assert "cost \\$0.003 vs \\$0" in out
    assert "`${{ secrets.X }}`" in out and "\n$ python a.py\n" in out
    assert "already \\$1" in out and "\\\\$" not in out

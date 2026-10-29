"""Smoke test: the Streamlit app renders and a full mock audit loop runs without exceptions."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_mock_run(monkeypatch):
    monkeypatch.setenv("USE_MOCK", "1")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    [s for s in at.slider if s.label.startswith("Demo pacing")][0].set_value(0.0)
    [r for r in at.radio if "Audit my own strategy code" in r.options][0].set_value("Audit my own strategy code")
    at.run()
    [b for b in at.button if b.label == "Run audit loop"][0].click()
    at.run()
    assert not at.exception
    assert any("Clean after" in s.value for s in at.success)
    assert any("Quick fix (no LLM)" in m.value for m in at.markdown)      # deterministic repair shown, $0
    assert any(c.language == "diff" and "+++ b/" in c.value for c in at.code)
    assert any("Probability of Backtest Overfitting" in m.value for m in at.markdown)   # PBO panel (mock mode)
    assert len(at.get("download_button")) >= 3   # HTML + Markdown audit report + JSON trace


def test_app_real_universe(monkeypatch):
    monkeypatch.setenv("USE_MOCK", "1")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    [s for s in at.slider if s.label.startswith("Demo pacing")][0].set_value(0.0)
    [r for r in at.radio if r.label == "Universe"][0].set_value("Real US equities (40, 2014–2025)")
    at.run()
    [b for b in at.button if b.label == "Run audit loop"][0].click()
    at.run()
    assert not at.exception


def test_app_adapted_backtrader(monkeypatch):
    """Pasted backtrader code is detected, adapted, audited and quick-fixed in the UI."""
    monkeypatch.setenv("USE_MOCK", "1")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    [s for s in at.slider if s.label.startswith("Demo pacing")][0].set_value(0.0)
    [r for r in at.radio if "Audit my own strategy code" in r.options][0].set_value("Audit my own strategy code")
    at.run()
    code = (Path(APP).parent / "examples" / "backtrader_leaky.py").read_text(encoding="utf-8")
    [t for t in at.text_area if t.key == "user_code"][0].set_value(code)
    at.run()
    assert any("Detected: backtrader Strategy" in c.value for c in at.caption)
    [b for b in at.button if b.label == "Run audit loop"][0].click()
    at.run()
    assert not at.exception
    assert any("Clean after" in s.value for s in at.success)


def test_app_notebook_cell_refs(monkeypatch):
    """An uploaded notebook's leak line, static findings and Quick fix are reported as 'cell N, line M'."""
    import sys
    sys.path.insert(0, str(Path(APP).parent))
    from notebook import extract_strategy
    monkeypatch.setenv("USE_MOCK", "1")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    nb = extract_strategy((Path(APP).parent / "examples" / "leaky_research.ipynb").read_bytes(), "leaky_research.ipynb")
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    [s for s in at.slider if s.label.startswith("Demo pacing")][0].set_value(0.0)
    [r for r in at.radio if "Audit my own strategy code" in r.options][0].set_value("Audit my own strategy code")
    at.run()
    # what the uploader branch stores after a successful extraction (AppTest has no file_uploader driver)
    at.session_state["nb_map"] = {"filename": "leaky_research.ipynb", "code": nb["code"], "line_map": nb["line_map"]}
    [t for t in at.text_area if t.key == "user_code"][0].set_value(nb["code"])
    at.run()
    [b for b in at.button if b.label == "Run audit loop"][0].click()
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Leak enters at cell 10, line 4" in md            # mom.shift(-1) in the momentum() helper cell
    assert "cell 9, line 7" in md or "cell 10, line 4" in md  # static findings / flagged-code notes
    tr = at.session_state["trace"]
    assert tr["notebook"]["filename"] == "leaky_research.ipynb"
    from audit.report import report_from_trace
    rep = report_from_trace(tr, "md", "synthetic")
    assert "cell 10, line 4" in rep
    assert not any("nemotron" in m.value.lower() and "code by" in m.value for m in at.markdown)  # mock label


def test_app_mock_label_and_dollars(monkeypatch):
    """Mock iterations are labelled as a replay, and money in markdown is escaped (no LaTeX)."""
    monkeypatch.setenv("USE_MOCK", "1")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    [s for s in at.slider if s.label.startswith("Demo pacing")][0].set_value(0.0)
    at.run()
    [b for b in at.button if b.label == "Run audit loop"][0].click()
    at.run()
    assert not at.exception
    assert any("mock replay (scripted Nemotron-style draft)" in m.value for m in at.markdown)
    for el in list(at.markdown) + list(at.caption):
        v = el.value
        assert "$0" not in v.replace("\\$", ""), v

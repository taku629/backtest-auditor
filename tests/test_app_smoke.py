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

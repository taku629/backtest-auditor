"""Downloadable audit report (audit/report.py): HTML + Markdown, from a loop trace and from audit_cli results."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import audit_cli
from agents.llm import make_llm
from agents.loop import run_audit_loop
from agents.mock_strategies import LEAKY
from audit.report import (code_hash, equity_svg, render_html, render_markdown, report_from_trace,
                          section_from_trace)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def trace(panel):
    tr = run_audit_loop("Buy stocks whose 20-day momentum is in the top decile, rebalance weekly.", panel,
                        make_llm(mock=True, mock_latency_s=0), initial_code=LEAKY["momentum"])
    return tr.to_dict()


def _external_refs(html: str) -> list[str]:
    urls = re.findall(r"(?:src|href)\s*=\s*['\"]([^'\"]+)", html) + re.findall(r"url\(([^)]+)\)", html)
    return [u for u in urls if not u.startswith("#")]


def test_html_report_from_trace_is_complete_and_self_contained(trace):
    h = report_from_trace(trace, "html", "synthetic test panel")
    assert h.startswith("<!DOCTYPE html>") and h.rstrip().endswith("</html>")
    assert "CLEAN" in h and "data-verdict='clean'" in h
    for name in ("future_perturbation", "survivorship", "execution_delay", "static_scan", "deflated_sharpe", "pbo"):
        assert name in h
    assert "LEAK ENTERS HERE" in h and 'class="cl leak"' in h          # localized line highlighted
    assert "Quick fix (deterministic, no LLM, $0)" in h and 'class="del"' in h and 'class="add"' in h
    assert "<svg" in h and "<polyline" in h                            # inline equity curve
    assert code_hash(LEAKY["momentum"]) in h                           # audited code hash
    assert "Models and token cost" in h and "nemotron" in h.lower()
    assert "Generated " in h and "UTC" in h
    assert "<script" not in h.lower() and _external_refs(h) == []       # no CDN, no JS
    assert "Before / after" in h and "Sharpe" in h


def test_markdown_report_from_trace(trace):
    md = report_from_trace(trace, "md")
    assert md.startswith("# Backtest Auditor audit report")
    assert "**Verdict: ✅ CLEAN**" in md
    assert "# <-- LEAK ENTERS HERE" in md and "```diff" in md
    assert "| pbo |" in md and "| deflated_sharpe |" in md
    assert "### Models and token cost" in md


def test_section_draft_vs_final(trace):
    sec = section_from_trace(trace)
    assert sec["verdict"] == "clean" and sec["localized"]["line"] >= 1
    assert sec["metrics_after"] is not None and sec["metrics_before"]["sharpe"] > sec["metrics_after"]["sharpe"]
    assert len(sec["equity"]) == 2 and sec["final_code_hash"] and sec["final_code_hash"] != sec["code_hash"]


def test_html_escapes_code_and_handles_empty_section():
    sec = {"name": "<b>x</b>", "verdict": "not_clean", "code": "def generate_signals(data):\n    s = '<script>'\n",
           "code_hash": "abc", "tests": [{"name": "static_scan", "passed": False, "blocking": True,
                                          "severity": "critical", "summary": "<img src=x>", "evidence": {}}]}
    h = render_html([sec])
    assert "<script>" not in h and "&lt;script&gt;" in h and "<img" not in h and "&lt;b&gt;x&lt;/b&gt;" in h
    assert "NOT CLEAN" in h and "No LLM calls" in h
    assert render_markdown([sec]).count("NOT CLEAN") == 1
    assert equity_svg([]) == "" and equity_svg([{"label": "a", "values": [1.0, 1.0], "dates": []}]).startswith("<svg")


def test_cli_report_html_and_md(tmp_path):
    out_html, out_md = tmp_path / "r.html", tmp_path / "r.md"
    code = audit_cli.main([str(ROOT / "examples" / "leaky_momentum.py"), "--fix", "--no-llm", "--no-pbo",
                           "--report", str(out_html), "--out", str(tmp_path / "o.txt")])
    assert code == 1
    h = out_html.read_text(encoding="utf-8")
    assert "NOT CLEAN" in h and "LEAK ENTERS HERE" in h and "After Quick fix" in h and "<svg" in h
    assert "No LLM calls: deterministic audit only ($0)." in h
    audit_cli.main([str(ROOT / "examples" / "leaky_momentum.py"), "--no-pbo", "--report", str(out_md),
                    "--out", str(tmp_path / "o2.txt")])
    md = out_md.read_text(encoding="utf-8")
    assert "**Verdict: ❌ NOT CLEAN**" in md and "line 9" in md

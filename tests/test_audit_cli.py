"""CI entry point (audit_cli.py): exit codes, report formats, SARIF, GitHub annotations, --fix."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import audit_cli

ROOT = Path(__file__).resolve().parent.parent
LEAKY = ROOT / "examples" / "leaky_momentum.py"
LEAKY_LINES = {9, 11, 17}   # dropna(axis=1), shift(-1), bfill()

CLEAN_CODE = '''import numpy as np
import pandas as pd


def generate_signals(data):
    """Top-decile 20-day momentum, point-in-time universe, weekly rebalance."""
    close = data["close"]
    listed = close.notna()
    mom = (close / close.shift(20) - 1).where(listed)
    rank = mom.rank(axis=1, pct=True)
    top = ((rank >= 0.9) & listed).astype(float)
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0)
    week = pd.Series(close.index.to_period("W"), index=close.index)
    is_rebal = (week != week.shift(1)).to_numpy()
    weights.loc[~is_rebal] = np.nan
    return weights.ffill().fillna(0.0).where(listed, 0.0)
'''

POLICY_VIOLATION_CODE = '''import os


def generate_signals(data):
    return data["close"] * 0.0
'''


@pytest.fixture(autouse=True)
def _no_ci_env(monkeypatch):
    for k in ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY", "GITHUB_WORKSPACE", "NEBIUS_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def clean_file(tmp_path):
    p = tmp_path / "clean_momentum.py"
    p.write_text(CLEAN_CODE)
    return p


@pytest.fixture(scope="module")
def leaky_json(tmp_path_factory):
    """One in-process JSON run of the leaky example, shared by several assertions."""
    out = tmp_path_factory.mktemp("bta") / "leaky.json"
    code = audit_cli.main([str(LEAKY), "--format", "json", "--out", str(out), "--annotations", "never"])
    return code, json.loads(out.read_text())


# ---------------------------------------------------------------- exit codes
def test_leaky_example_fails_with_exit_1_as_subprocess():
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY")}
    p = subprocess.run([sys.executable, str(ROOT / "audit_cli.py"), str(LEAKY)], capture_output=True, text=True,
                       cwd=ROOT, env=env, timeout=120)
    assert p.returncode == 1, p.stderr
    out = p.stdout
    assert "examples/leaky_momentum.py  ->  FAIL" in out
    assert "[FAIL] future_perturbation" in out and "first leaking date: 2019-" in out
    for n in LEAKY_LINES:
        assert f"examples/leaky_momentum.py:{n}" in out
    assert "fix:" in out


def test_clean_strategy_passes_even_at_warning(clean_file, capsys):
    assert audit_cli.main([str(clean_file), "--fail-on", "warning"]) == 0
    out = capsys.readouterr().out
    assert "->  PASS" in out and "[FAIL]" not in out


def test_fail_on_threshold_controls_exit(tmp_path, capsys):
    from agents.mock_strategies import REVERSAL_V2   # honest but unprofitable: fails only deflated_sharpe (medium)
    p = tmp_path / "reversal.py"
    p.write_text(REVERSAL_V2)
    assert audit_cli.main([str(p), "--fail-on", "high", "--format", "json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    failing = [t for t in rep["files"][0]["tests"] if not t["passed"]]
    assert failing and all(t["severity"] not in ("critical", "high") for t in failing)
    assert audit_cli.main([str(p), "--fail-on", "warning", "--format", "json"]) == 1
    assert audit_cli.main([str(LEAKY), "--fail-on", "critical", "--format", "json"]) == 1


def test_missing_file_is_usage_error(capsys):
    assert audit_cli.main(["does/not/exist.py"]) == 2
    assert "nothing to audit" in capsys.readouterr().err


def test_bad_data_spec_is_usage_error(clean_file):
    assert audit_cli.main([str(clean_file), "--data", "parquet:x"]) == 2


def test_sandbox_policy_violation_fails(tmp_path, capsys):
    p = tmp_path / "bad.py"
    p.write_text(POLICY_VIOLATION_CODE)
    assert audit_cli.main([str(p), "--format", "json"]) == 1
    f = json.loads(capsys.readouterr().out)["files"][0]
    sbx = next(t for t in f["tests"] if t["name"] == "sandbox")
    assert not sbx["passed"] and sbx["severity"] == "critical"
    assert sbx["lines"][0]["line"] == 1 and "os" in f["error"]


def test_glob_expansion_and_multiple_files(tmp_path, clean_file, capsys):
    (tmp_path / "leaky.py").write_text(LEAKY.read_text())
    (tmp_path / "leaky.fixed.py").write_text(CLEAN_CODE)    # our own outputs are skipped by globs
    assert audit_cli.main([str(tmp_path / "*.py"), "--format", "json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    names = sorted(Path(f["path"]).name for f in rep["files"])
    assert names == ["clean_momentum.py", "leaky.py"]
    assert {Path(f["path"]).name: f["passed"] for f in rep["files"]} == {"clean_momentum.py": True, "leaky.py": False}


def test_csv_data_source(tmp_path, clean_file):
    from data import generate_panel
    panel = generate_panel(n_tickers=12, years=2, n_delisted=2, n_ipo=2)
    long = panel["close"].stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    csv = tmp_path / "prices.csv"
    long.to_csv(csv, index=False)
    out = tmp_path / "r.json"
    code = audit_cli.main([str(clean_file), "--data", f"csv:{csv}", "--format", "json", "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rep["data"].startswith(f"csv:{csv}") and "12 tickers" in rep["data"]
    assert code in (0, 1) and rep["files"][0]["tests"]


# ---------------------------------------------------------------- JSON / SARIF / Markdown
def test_json_report_structure(leaky_json):
    code, rep = leaky_json
    assert code == 1 and rep["passed"] is False and rep["fail_on"] == "high"
    f = rep["files"][0]
    assert f["path"].endswith("examples/leaky_momentum.py") and f["metrics"]["sharpe"] > 3
    tests = {t["name"]: t for t in f["tests"]}
    fp = tests["future_perturbation"]
    assert not fp["passed"] and fp["severity"] == "critical" and fp["evidence"]["first_leaking_date"]
    assert {ln["line"] for ln in fp["lines"]} == LEAKY_LINES
    assert {ln["line"] for ln in tests["survivorship"]["lines"]} == {9}
    assert {x["rule"] for x in f["static_findings"]} >= {"negative_shift", "backward_fill", "survivor_universe"}


SARIF_MIN_SCHEMA = {
    "type": "object", "required": ["version", "runs"],
    "properties": {
        "version": {"const": "2.1.0"},
        "runs": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["tool", "results"],
            "properties": {
                "tool": {"type": "object", "required": ["driver"], "properties": {"driver": {
                    "type": "object", "required": ["name", "rules"]}}},
                "results": {"type": "array", "items": {
                    "type": "object", "required": ["ruleId", "message", "locations", "level"],
                    "properties": {
                        "level": {"enum": ["none", "note", "warning", "error"]},
                        "message": {"type": "object", "required": ["text"]},
                        "locations": {"type": "array", "minItems": 1, "items": {
                            "type": "object", "required": ["physicalLocation"], "properties": {
                                "physicalLocation": {"type": "object",
                                                     "required": ["artifactLocation", "region"], "properties": {
                                                         "region": {"type": "object", "required": ["startLine"],
                                                                    "properties": {"startLine": {"type": "integer",
                                                                                                 "minimum": 1}}}}}}}}}}}}}},
    },
}


def test_sarif_is_valid_and_points_at_offending_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "r.sarif"
    assert audit_cli.main([str(LEAKY), "--format", "sarif", "--out", str(out), "--annotations", "never"]) == 1
    sarif = json.loads(out.read_text())
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(sarif, SARIF_MIN_SCHEMA)
    run = sarif["runs"][0]
    rule_ids = [r["id"] for r in run["tool"]["driver"]["rules"]]
    assert len(rule_ids) == len(set(rule_ids))
    for res in run["results"]:
        assert rule_ids[res["ruleIndex"]] == res["ruleId"]
    by_rule = {r["ruleId"]: r for r in run["results"]}
    assert {"dynamic/future_perturbation", "dynamic/survivorship", "static/negative_shift",
            "static/backward_fill", "static/survivor_universe"} <= set(by_rule)
    loc = by_rule["static/negative_shift"]["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "examples/leaky_momentum.py"   # relative -> PR annotations
    assert loc["region"]["startLine"] == 11
    assert by_rule["dynamic/future_perturbation"]["level"] == "error"
    assert "First leaking date" in by_rule["dynamic/future_perturbation"]["message"]["text"]


def test_sarif_for_clean_file_has_no_results(clean_file, capsys):
    assert audit_cli.main([str(clean_file), "--format", "sarif"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["results"] == []


def test_markdown_report(capsys):
    audit_cli.main([str(LEAKY), "--format", "markdown"])
    md = capsys.readouterr().out
    assert md.startswith("## Backtest Auditor report") and "| future_perturbation | ❌ fail | critical |" in md
    assert "**Offending lines**" in md and "L9, L11, L17" in md


# ---------------------------------------------------------------- GitHub Actions integration
def test_github_annotations_and_step_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert audit_cli.main([str(LEAKY)]) == 1
    out = capsys.readouterr().out
    ann = [ln for ln in out.splitlines() if ln.startswith("::")]
    assert any(ln.startswith("::error file=examples/leaky_momentum.py,line=11,") for ln in ann)
    assert any(ln.startswith("::error file=examples/leaky_momentum.py,line=9,") and "survivorship" in ln
               for ln in ann)
    assert all("\n" not in ln for ln in ann) and any("title=Backtest Auditor%3A" in ln for ln in ann)
    assert "## Backtest Auditor report" in summary.read_text()


def test_annotations_go_to_stderr_when_stdout_is_machine_readable(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    audit_cli.main([str(LEAKY), "--format", "json"])
    cap = capsys.readouterr()
    json.loads(cap.out)                     # stdout stays valid JSON
    assert "::error file=" in cap.err


def test_no_annotations_outside_actions(capsys):
    audit_cli.main([str(LEAKY)])
    assert "::error" not in capsys.readouterr().out


# ---------------------------------------------------------------- --fix
def test_fix_without_api_key_is_skipped(tmp_path, capsys):
    p = tmp_path / "leaky.py"
    p.write_text(LEAKY.read_text())
    assert audit_cli.main([str(p), "--fix", "--no-quickfix", "--format", "json"]) == 1
    cap = capsys.readouterr()
    assert json.loads(cap.out)["files"][0]["fix_status"] == "skipped_no_api_key"
    assert "NEBIUS_API_KEY" in cap.err and not (tmp_path / "leaky.fixed.py").exists()


def test_fix_quick_fix_needs_no_api_key(tmp_path, capsys):
    p = tmp_path / "leaky.py"
    p.write_text(LEAKY.read_text())
    assert audit_cli.main([str(p), "--fix", "--format", "json"]) == 1      # the original still fails
    f = json.loads(capsys.readouterr().out)["files"][0]
    assert f["fix_status"] == "quick_fix" and f["quick_fix"]["status"] == "fixed"
    assert "+++ b/leaky.py" in f["quick_fix"]["diff"] and f["quick_fix"]["cost_usd"] == 0.0
    assert not (tmp_path / "leaky.fixed.py").exists()                     # only with --write
    assert audit_cli.main([str(p), "--fix", "--write", "--no-llm"]) == 1
    out = capsys.readouterr().out
    assert "Quick fix (no LLM" in out and "-    mom = " in out
    fixed = tmp_path / "leaky.fixed.py"
    assert fixed.exists() and audit_cli.main([str(fixed), "--fail-on", "high"]) == 0


def test_fix_with_mock_llm_writes_clean_candidate(tmp_path, capsys):
    p = tmp_path / "leaky.py"
    p.write_text(LEAKY.read_text())
    assert audit_cli.main([str(p), "--fix", "--no-quickfix", "--mock-llm", "--format", "json"]) == 1
    f = json.loads(capsys.readouterr().out)["files"][0]
    fixed = tmp_path / "leaky.fixed.py"
    assert f["fix_status"] == "clean" and fixed.exists()
    assert audit_cli.main([str(fixed), "--fail-on", "high"]) == 0

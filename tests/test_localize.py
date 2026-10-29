"""Dynamic leak localizer (audit/localize.py): offline, runs through the trace sandbox."""
from pathlib import Path

import pytest

from audit.localize import instrument, localize_leak, localized_message
from sandbox.tracer import TraceError

ROOT = Path(__file__).resolve().parent.parent
STRAT = ROOT / "bench" / "strategies"


def _line_of(code: str, needle: str) -> int:
    return next(i for i, ln in enumerate(code.splitlines(), 1) if needle in ln and not ln.lstrip().startswith("#"))


def test_instrument_keeps_semantics_and_records_every_assignment():
    code = (STRAT / "L12_helper_hidden_shift.py").read_text()
    inst, meta = instrument(code)
    kinds = {(m["func"], m["var"], m["kind"]) for m in meta}
    assert ("to_trade_date", "df", "param") in kinds
    assert ("to_trade_date", "<return>", "return") in kinds
    assert ("generate_signals", "mom", "assign") in kinds
    mom = next(m for m in meta if m["var"] == "mom")
    assert "to_trade_date" in mom["calls"] and mom["line"] == _line_of(code, "mom = to_trade_date")
    assert "_bta_rec(" in inst and "_bta_ret(" in inst


def test_leaky_momentum_example_localizes_to_first_leaking_statement(panel):
    # In execution order the survivor universe (dropna(axis=1)) comes before the negative shift, and it already
    # changes past values (killed tickers vanish from the columns), so it is the entry point.
    code = (ROOT / "examples" / "leaky_momentum.py").read_text()
    loc = localize_leak(code, panel)
    assert loc["found"]
    assert loc["entry"]["line"] == _line_of(code, "close.dropna(axis=1)")
    assert loc["entry"]["var"] == "close"
    chain_lines = [c["line"] for c in loc["chain"]]
    assert _line_of(code, ".shift(-1)") in chain_lines
    assert "Leak enters at line" in localized_message(loc)


@pytest.mark.parametrize("fname,needle,kind", [
    ("L17_rank_default_axis.py", "pctl = mom.rank(pct=True)", "time-indexed rows changed"),
    ("L18_sub_mean_axis_trap.py", "mom.sub(mom.mean(), axis=1)", "time-indexed rows changed"),
    ("L19_resample_ms_last.py", 'resample("MS").last()', "time-indexed rows changed"),
    ("L22_descending_sort_rolling.py", "recent_first.rolling(20", "time-indexed rows changed"),
    ("L15_min_history_filter.py", "close.count() >= 1200", "non-time object changed"),
    ("L12_helper_hidden_shift.py", "return df.shift(-lag)", "time-indexed rows changed"),
])
def test_static_blind_and_helper_leaks_localize(panel, fname, needle, kind):
    code = (STRAT / fname).read_text()
    loc = localize_leak(code, panel)
    assert loc["found"], loc
    assert loc["entry"]["line"] == _line_of(code, needle)
    assert loc["entry"]["kind"] == kind


def test_reversed_index_is_aligned_by_label_not_position(panel):
    # sort_index(ascending=False) only reorders rows: it must NOT be reported as the entry.
    code = (STRAT / "L22_descending_sort_rolling.py").read_text()
    loc = localize_leak(code, panel)
    rf = [p for p in loc["per_line"] if p["var"] == "recent_first"]
    assert rf and not rf[0]["diverged"]


@pytest.mark.parametrize("fname", ["C01_momentum_12_1.py", "C15_resample_me_last.py", "C13_transpose_rank.py"])
def test_clean_strategy_has_no_divergence(panel, fname):
    loc = localize_leak((STRAT / fname).read_text(), panel)
    assert loc["found"] is False
    assert localized_message(loc) == ""


def test_policy_still_enforced_and_reserved_names_rejected(small_panel):
    bad = "import os\ndef generate_signals(data):\n    return data['close'] * 0\n"
    with pytest.raises(TraceError):
        localize_leak(bad, small_panel)
    spoof = "def generate_signals(data):\n    _bta_rec(0, 1)\n    return data['close'] * 0\n"
    with pytest.raises(TraceError):
        localize_leak(spoof, small_panel)


def test_hard_timeout(small_panel):
    code = "def generate_signals(data):\n    while True:\n        x = 1\n"
    with pytest.raises(TraceError):
        localize_leak(code, small_panel, timeout=3)


def test_run_audit_attaches_localized_evidence(small_panel):
    from audit.leak_tests import run_audit
    from sandbox.runner import SandboxExecutor
    code = (STRAT / "L17_rank_default_axis.py").read_text()
    rep = run_audit(small_panel, SandboxExecutor(code), code)
    fp = next(t for t in rep.tests if t.name == "future_perturbation")
    assert not fp.passed
    lz = fp.evidence["localized"]
    assert lz["found"] and lz["entry"]["line"] == _line_of(code, "mom.rank(pct=True)")
    # prompts: the Auditor input carries the line
    from agents import prompts as P
    tests = rep.to_dict()["tests"]
    assert f"Leak enters at line {lz['entry']['line']}" in P.localized_hint(tests)
    assert "per_line" not in P.compact_tests(tests)

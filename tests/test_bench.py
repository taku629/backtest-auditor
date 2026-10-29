"""LeakBench: offline checks that the deterministic suite matches the labels.

Known, documented deterministic errors are listed in KNOWN_DETERMINISTIC_ERRORS and
reported in bench/RESULTS.md; the test pins their current behaviour so any change
(fixed or regressed) is noticed.
"""
import json

import pytest

from bench.run_bench import (RESULTS_JSON, _cache_key, deterministic_audit, load_strategies, parse_verdict,
                             score, strip_header, write_markdown)
from sandbox.runner import check_policy

STRATEGIES = load_strategies()
IDS = [s["id"] for s in STRATEGIES]

# id -> the only blocking test expected to (wrongly) fire
KNOWN_DETERMINISTIC_ERRORS = {
    # Clean: month.shift(-1) on the trading calendar. Dynamic tests pass; the static
    # negative_shift rule (critical) fires anyway. See bench/RESULTS.md.

}


def test_benchmark_shape():
    leaky = [s for s in STRATEGIES if s["leaky"]]
    clean = [s for s in STRATEGIES if not s["leaky"]]
    assert len(leaky) >= 14 and len(clean) >= 10
    assert len(set(IDS)) == len(IDS)
    for s in STRATEGIES:
        assert s["bug"], s["id"]
        assert "LEAKBENCH" not in s["stripped"], s["id"]
        assert s["stripped"].lstrip().startswith("import"), s["id"]
        assert "def generate_signals" in s["stripped"]
        assert check_policy(s["code"]) == [], (s["id"], check_policy(s["code"]))


def test_strip_header():
    code = "# LEAKBENCH-LABEL: leaky\n# LEAKBENCH-BUG: x\n\nimport numpy as np\n# keep me\n"
    assert strip_header(code) == "import numpy as np\n# keep me\n"


@pytest.mark.parametrize("sid", IDS)
def test_deterministic_suite_matches_label(panel, sid):
    s = next(x for x in STRATEGIES if x["id"] == sid)
    d = deterministic_audit(s["code"], panel, sandbox=False)
    assert not d["error"], d["error"]
    if sid in KNOWN_DETERMINISTIC_ERRORS:
        assert d["blocking"] == [KNOWN_DETERMINISTIC_ERRORS[sid]], d["blocking"]
        assert d["tests"]["future_perturbation"]["passed"]
        return
    if s["leaky"]:
        assert d["flagged"], f"{sid}: leaky strategy passed every blocking test"
        # every leak in the bench is caught by a dynamic test, not only by the static scan
        assert set(d["blocking"]) - {"static_scan"}, (sid, d["blocking"])
    else:
        assert not d["flagged"], f"{sid}: clean strategy flagged by {d['blocking']}"


def test_score():
    m = score([True, True, False, False], [True, None, True, False])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["recall"] == 0.5 and m["false_positive_rate"] == 0.5 and m["unanswered"] == 1


@pytest.mark.parametrize("text,expected", [
    ('{"leaky": true, "reason": "shift(-1)"}', True),
    ('reasoning... {"a": 1} final: {"leaky": false, "reason": "ok"}', False),
    ('```json\n{"leaky": "true", "reason": "x"}\n```', True),
    ('"leaky": false and then it stopped', False),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text)["leaky"] is expected


def test_parse_verdict_none():
    assert parse_verdict("I think it is fine") is None


def test_cache_key_stable():
    assert _cache_key("m", "code") == _cache_key("m", "code") != _cache_key("m2", "code")


def test_results_file_consistent(tmp_path):
    if not RESULTS_JSON.exists():
        pytest.skip("bench/results.json not generated yet")
    res = json.loads(RESULTS_JSON.read_text())
    assert res["n_strategies"] == len(res["strategies"])
    labels = [r["leaky"] for r in res["strategies"]]
    for m in res["methods"]:
        again = score(labels, [r["pred"][m] for r in res["strategies"]])
        assert again == res["metrics"][m], m
    write_markdown(res, tmp_path / "R.md")
    assert "Deterministic audit" in (tmp_path / "R.md").read_text()

    from bench.ui import bar_chart, strategy_frame, summary_frame
    assert len(summary_frame(res)) == len(res["methods"])
    assert len(strategy_frame(res)) == res["n_strategies"]
    assert len(bar_chart(res).data) == 2

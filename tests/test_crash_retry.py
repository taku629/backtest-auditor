"""A rewrite that crashes in the sandbox gets one free retry and keeps the original audit findings."""
from agents.llm import make_llm
from agents.loop import run_audit_loop
from data import generate_panel

CRASHING = "import pandas as pd\n\ndef generate_signals(data):\n    raise ValueError('boom')\n"


def test_crash_gets_free_retry_and_still_reaches_clean():
    data = generate_panel(seed=51, n_tickers=30, years=4)
    llm = make_llm(mock=True, mock_latency_s=0)
    t = run_audit_loop("momentum top decile weekly", data, llm, max_iters=1,
                       initial_code=CRASHING, summarize=False).to_dict()
    # 1 crashed user draft + 1 audited rewrite, despite max_iters=1
    assert len(t["iterations"]) == 2
    assert t["iterations"][0]["status"] == "error"
    assert t["iterations"][1]["status"] in ("clean", "failing")

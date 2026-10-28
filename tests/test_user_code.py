from agents.llm import make_llm
from agents.loop import run_audit_loop
from agents.mock_strategies import LEAKY
from data.synthetic import generate_panel


def test_user_code_is_audited_first_without_coder_call():
    tr = run_audit_loop("momentum", generate_panel(n_tickers=20, years=3), make_llm(mock=True), max_iters=3,
                        initial_code=LEAKY["momentum"])
    first = tr.iterations[0]
    assert first.user_code and first.code.strip() == LEAKY["momentum"].strip()
    assert first.status == "failing" and all(c["role"] != "coder" for c in first.llm_calls)
    assert tr.settings["user_code"] is True
    assert not any(c["step"] == "write_code" for c in tr.calls)

"""Headless run: python cli.py "buy top-decile 20-day momentum, rebalance weekly" [--live] [--out trace.json]"""
from __future__ import annotations

import argparse
import sys

from agents.llm import make_llm
from agents.loop import run_audit_loop
from data import generate_panel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backtest Auditor (headless)")
    ap.add_argument("hypothesis", nargs="?", default="Buy stocks whose 20-day momentum is in the top decile, "
                                                     "equal weight, rebalance weekly.")
    ap.add_argument("--live", action="store_true", help="call Nebius Token Factory (needs NEBIUS_API_KEY)")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    def show(e):
        t = e["type"]
        if t == "coder_done":
            print(f"[{e['iteration'] + 1}] coder  ({e['call']['model']}) -> {len(e['code'].splitlines())} lines")
        elif t == "tests_done":
            fails = [f"{x['name']}({x['severity']})" for x in e["tests"] if not x["passed"]]
            print(f"[{e['iteration'] + 1}] audit  sharpe={e['metrics']['sharpe']:.2f} "
                  f"{'CLEAN' if e['clean'] else 'FAIL: ' + ', '.join(fails)}")
        elif t == "auditor_done":
            print(f"[{e['iteration'] + 1}] auditor({e['call']['model']}) -> fix instructions")
        elif t == "error":
            print(f"[{e['iteration'] + 1}] sandbox error: {e['error'][:200]}")

    trace = run_audit_loop(a.hypothesis, generate_panel(), make_llm(mock=not a.live), max_iters=a.max_iters,
                           cost_bps=a.cost_bps, on_event=show)
    print(f"\nFinal: {trace.final_status}\n{trace.summary}")
    if a.out:
        with open(a.out, "w") as f:
            f.write(trace.to_json(indent=1))
        print(f"trace written to {a.out}")
    return 0 if trace.final_status == "clean" else 1


if __name__ == "__main__":
    sys.exit(main())

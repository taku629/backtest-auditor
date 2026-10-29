# Backtest Auditor in CI

`audit_cli.py` runs the **deterministic** audit on strategy files (`generate_signals(data) -> weights`):
each file executes in the sandbox, then the test battery (future-perturbation, survivorship counterfactual,
execution delay, AST scan, plausibility, deflated Sharpe, train/test split) decides pass/fail.
No LLM, no API key, no network. Same input, same seed, same verdict.

## Command line

```bash
pip install -r requirements.txt
python audit_cli.py strategies/*.py                       # text report, exit 1 on critical/high failures
python audit_cli.py "strategies/**/*.py" --fail-on warning # stricter: medium findings fail too
python audit_cli.py strat.py --format sarif --out results.sarif
python audit_cli.py strat.py --data csv:prices.csv         # long CSV: date,ticker,close[,open,high,low,volume]
python audit_cli.py strat.py --fix                         # + Nemotron repair -> strat.fixed.py (needs NEBIUS_API_KEY)
```

| Option | Default | Meaning |
|---|---|---|
| `paths` | | files or globs (quote globs to use `**`); `*.fixed.py` is skipped by globs |
| `--data` | `synthetic` | seeded panel (50 tickers, 6 years, 8 delisted, 6 IPOs) or `csv:PATH` |
| `--format` | `text` | `text`, `markdown`, `json`, `sarif` (2.1.0) |
| `--fail-on` | `high` | `critical`, `high`, or `warning` (medium/low failures also fail) |
| `--out` / `--sarif-out` | stdout / none | write the report / an extra SARIF file |
| `--annotations` | `auto` | `::error file=..,line=..::` lines when `GITHUB_ACTIONS=true` |
| `--seed`, `--n-cuts`, `--cost-bps`, `--timeout` | 7, 6, 10, 90 | audit settings |
| `--fix` | off | runs `agents/loop.py` on the file and writes `<name>.fixed.py`; skipped with a warning when `NEBIUS_API_KEY` is unset (`--mock-llm` = offline demo) |

Exit codes: `0` passed, `1` a test at or above `--fail-on` failed (or the file could not run in the sandbox), `2` usage error.
Offending line numbers come from the AST scan; dynamic failures are linked to the static findings that can cause them.

### Example output (`python audit_cli.py examples/leaky_momentum.py`, exit 1)

```text
Backtest Auditor 0.2.0  |  data: synthetic (50 tickers, 1512 dates, 2019-01-02..2024-10-17)  |  fail-on: high

==============================================================================
examples/leaky_momentum.py  ->  FAIL  (3 test(s) at or above 'high')
  backtest: Sharpe 6.63, CAGR 545.2%, max DD -12.8%, turnover 46.3x/yr
==============================================================================
  [FAIL] future_perturbation  critical LOOK-AHEAD: scrambling data after the cut changed past signals at 6/6 cut dates (first leaking date 2019-02-04, max weight change 0.250). Changes reach the start of the sample -> full-sample statistic or survivor universe.
         first leaking date: 2019-02-04
         evidence: leaking cut dates: 6/6; max |weight change| 0.25
         evidence: cut 2022-05-19: 224 earlier rows changed (2019-02-18 .. 2022-05-19), e.g. SYN13, SYN16, SYN28
         evidence: cut 2022-07-29: 165 earlier rows changed (2019-02-04 .. 2022-07-29), e.g. SYN10, SYN26
         evidence: cut 2022-11-01: 227 earlier rows changed (2019-04-08 .. 2022-11-01), e.g. SYN12, SYN32
         examples/leaky_momentum.py:9  [survivor_universe]  close = close.dropna(axis=1)
         examples/leaky_momentum.py:11  [negative_shift]  mom = (close / close.shift(20) - 1).shift(-1)
         examples/leaky_momentum.py:17  [backward_fill]  return weekly.reindex(close.index).bfill().fillna(0.0)
         fix: Every value used for row t must be computable from rows <= t. Remove negative shifts, centred windows, backward fills, full-sample normalisation and universes defined by future availability (e.g. dropna(axis=1)). Use rolling/expanding estimators.
  [FAIL] static_scan          critical 3 red flag(s): backward_fill, negative_shift, survivor_universe.
         examples/leaky_momentum.py:9  [survivor_universe]  close = close.dropna(axis=1)
         examples/leaky_momentum.py:11  [negative_shift]  mom = (close / close.shift(20) - 1).shift(-1)
         examples/leaky_momentum.py:17  [backward_fill]  return weekly.reindex(close.index).bfill().fillna(0.0)
         fix: Build a point-in-time universe: eligible = close.notna() on each date. Use a non-negative period (e.g. .shift(1)); the engine already trades at t+1. Use .ffill() (or leave NaN) so row t only sees data up to t.
  [FAIL] survivorship         high     SURVIVORSHIP: past signals change when delisted names are made to survive (universe depends on future survival).
         first leaking date: 2019-02-25
         evidence: weight_before_listing: 0.0
         evidence: counterfactual_revived: ['SYN38', 'SYN39', 'SYN02']
         evidence: counterfactual_rows_changed: 200
         evidence: counterfactual_first_change: 2019-02-25
         evidence: avg_weight_on_later_delisted: 0.0
         examples/leaky_momentum.py:9  [survivor_universe]  close = close.dropna(axis=1)
         fix: Define the universe point-in-time (eligible = close.notna() on each date); never drop tickers because they have missing data later.
  [FAIL] plausibility         medium   Sharpe 6.63 (CAGR 545%) after costs is implausibly high for daily equities; treat as a bug until proven otherwise.
         evidence: Sharpe 6.6263, CAGR 5.4524
         fix: Re-check timing and data alignment; real edges rarely exceed Sharpe 2-3.
  [PASS] execution_delay      high     Sharpe 6.63 at t+1 vs 5.30 at t+2 (drop 20%); robust to one extra bar of delay.
  [PASS] deflated_sharpe      medium   Deflated Sharpe 1.00 >= 0.95 after 1 trial(s): edge survives selection bias.
  [PASS] train_test_split     medium   Sharpe train 6.31 vs test 7.51 (70/30 split): no severe degradation.

Summary: 0/1 file(s) passed (fail-on: high).
```

## Jupyter notebooks

`.ipynb` files (and globs such as `"research/**/*.ipynb"`) are audited directly; `.ipynb_checkpoints/` is skipped.
`notebook.py` (`extract_strategy(source, filename)`) reads the notebook JSON (no nbformat or Jupyter needed),
neutralises IPython syntax (`%matplotlib`, `!pip`, `%%time`; `%%bash`-style cells are skipped), skips cells
that do not parse, and walks the AST from `generate_signals` to keep only the helper functions, constants and
imports it depends on. Plots, exploration and data-loading cells are left out; the strategy still receives the
auditor's `data`. What was skipped is listed in the report.

```text
$ python audit_cli.py examples/leaky_research.ipynb          # exit 1
  notebook: 14 cells (9 code); generate_signals in cell 11; audited np, LOOKBACK, TOP_PCT, zscore, momentum, generate_signals
  [FAIL] future_perturbation  critical LOOK-AHEAD: ...
         examples/leaky_research.ipynb (cell 9, line 7)  [...]  return (df - df.mean()) / df.std()
         examples/leaky_research.ipynb (cell 10, line 4)  [negative_shift]  return mom.shift(-1)
```

Cells are numbered 1-based over all cells, markdown included (the order VS Code and JupyterLab show).
Text, Markdown and JSON reports say "cell N, line M"; JSON also includes the extracted module and the line map.
**SARIF and `::error` annotations** point at the notebook file, but GitHub cannot reliably place annotations
inside `.ipynb` JSON, so `startLine` is the line in the *extracted module* and every message starts with
`[cell N line M]` (SARIF results also carry `notebookCell` / `notebookCellLine` properties). Expect the alert on
the notebook file in the PR, and read the cell reference from the message. A notebook with no `generate_signals`
fails with a `notebook` test that explains the contract and names any function that looks like a signal.
`--fix` writes the repaired extracted module to `<name>.fixed.py` next to the notebook.

In the GitHub Action, `paths` takes several whitespace-separated globs, so add notebooks there:
`paths: "strategies/**/*.py strategies/**/*.ipynb"`. This repo's example workflow audits `examples/*.py` only,
so `examples/leaky_research.ipynb` is not part of that CI job.

## GitHub Action

```yaml
# .github/workflows/backtest-audit.yml
on: [push, pull_request]
permissions:
  contents: read
  security-events: write          # SARIF -> annotations on the exact lines in the PR
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: taku629/backtest-auditor@main   # pin a tag or SHA
        with:
          paths: "strategies/**/*.py"
          fail-on: high                       # critical | high | warning
          format: text                        # text | markdown | json | sarif
          # data: csv:data/prices.csv
          # upload-sarif: "false"
```

The composite action sets up Python 3.11, installs `requirements.txt` from the action, runs `audit_cli.py`,
prints the report, writes `::error` annotations and a Markdown job summary, uploads SARIF to code scanning
(skipped quietly when the token lacks `security-events: write`, e.g. fork PRs), then fails the job if the
audit failed. Outputs: `exit-code`, `sarif-file`, `report-file`.

Annotations look like this in the log and on the PR diff:

```text
::error file=examples/leaky_momentum.py,line=11,title=Backtest Auditor%3A static_scan (critical)::.shift() with a negative period pulls FUTURE rows into row t. Line: mom = (close / close.shift(20) - 1).shift(-1) [negative_shift]. Fix: Use a non-negative period (e.g. .shift(1)); the engine already trades at t+1.
::error file=examples/leaky_momentum.py,line=9,title=Backtest Auditor%3A survivorship (high)::SURVIVORSHIP: past signals change when delisted names are made to survive (universe depends on future survival). First leaking date: 2019-02-25. ...
```

`.github/workflows/example-audit.yml` in this repo runs the action on `examples/*.py` and expects the
deliberately leaky example to fail, and runs it on a point-in-time momentum strategy that must pass at `fail-on: warning`.

To propose fixes in CI, add `extra-args: --fix` and `env: NEBIUS_API_KEY: ${{ secrets.NEBIUS_API_KEY }}`;
upload the `*.fixed.py` files as artifacts. The verdict on the original file is still deterministic.

## pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/taku629/backtest-auditor
    rev: main                      # pin a tag or SHA
    hooks:
      - id: backtest-auditor
        files: ^strategies/.*\.py$
        # args: [--fail-on, critical]
```

The hook uses the `python3` on your PATH, which needs `numpy` and `pandas`. Inside this repo you can use
`repo: local` with `entry: python audit_cli.py`, `language: system`.

## Limits

- The verdict is only as good as the data: on `synthetic` it proves the code is causal and point-in-time;
  survivorship on your own CSV needs delisted tickers in it (the audit warns when there are none).
- The sandbox is defence in depth, not a multi-tenant boundary; run CI on ephemeral runners.

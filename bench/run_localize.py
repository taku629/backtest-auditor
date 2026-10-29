"""Run the dynamic leak localizer on every LeakBench strategy and write ``bench/LOCALIZE.md``.

Usage::

    python -m bench.run_localize            # leaky strategies + clean controls
    python -m bench.run_localize --leaky-only

Ground truth is the line where the labelled bug is written (``TRUTH`` below, read from each file by hand; a few
bugs legitimately span two lines and accept either). The localizer runs through the sandbox on the synthetic panel
at the cuts that ``run_audit`` reports as leaking (falling back to its default cuts).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(__file__).resolve().parent
STRAT_DIR = BENCH_DIR / "strategies"
OUT_MD = BENCH_DIR / "LOCALIZE.md"
OUT_JSON = BENCH_DIR / "localize_results.json"

# strategy id -> (accepted bug lines, where the bug is)
TRUTH: dict[str, tuple[tuple[int, ...], str]] = {
    "L01": ((14,), "mom.shift(-1)"),
    "L02": ((13,), "rolling(center=True)"),
    "L03": ((12,), "close.bfill()"),
    "L04": ((14,), "(mom - mom.mean()) / mom.std()"),
    "L05": ((12,), "close.dropna(axis=1)"),
    "L06": ((13,), "close.iloc[-1] / close.iloc[-252]"),
    "L07": ((14,), "rank(axis=0)"),
    "L08": ((13,), "iloc[::-1].cummax().iloc[::-1]"),
    "L09": ((18,), "reindex(method='bfill') of the W-FRI .last() portfolio"),
    "L10": ((14,), "iloc[::-1].expanding().std()"),
    "L11": ((15,), "groupby(level=1).quantile(0.8) over the full sample"),
    "L12": ((10,), "df.shift(-lag) inside the helper"),
    "L13": ((12,), "close.iloc[-1].notna()"),
    "L14": ((18,), "np.polyfit over the whole series"),
    "L15": ((13,), "close.count() >= 1200"),
    "L16": ((13,), "volume.iloc[-60:].median()"),
    "L17": ((13,), "mom.rank(pct=True) (axis=0 default)"),
    "L18": ((14,), "mom.sub(mom.mean(), axis=1)"),
    "L19": ((12,), "resample('MS').last()"),
    "L20": ((18,), "groupby(month).transform('last')"),
    "L21": ((14,), "ret.shift(-1)"),
    "L22": ((13,), "rolling on the newest-first index"),
    "L23": ((13,), "fillna(mom.mean())"),
    "L24": ((14,), "pd.qcut on the stacked panel"),
}
STATIC_BLIND = {"L17", "L18", "L19", "L22"}


def run(leaky_only: bool = False, log=print, only: list[str] | None = None) -> dict:
    from audit.leak_tests import run_audit
    from audit.localize import localize_leak
    from data.synthetic import generate_panel
    from sandbox.runner import SandboxExecutor

    data = generate_panel()
    idx = data["close"].index
    rows = []
    files = sorted(STRAT_DIR.glob("L*.py")) + ([] if leaky_only else sorted(STRAT_DIR.glob("C*.py")))
    if only:
        files = [f for f in files if any(f.name.startswith(o) for o in only)]
    for f in files:
        sid = f.name[:3]
        code = f.read_text()
        t0 = time.time()
        rep = run_audit(data, SandboxExecutor(code), code, localize=False)
        t_audit = time.time() - t0
        fp = next(t for t in rep.tests if t.name == "future_perturbation")
        leaking = [int(idx.get_loc(c["cut_date"])) for c in fp.evidence.get("cuts", []) if c.get("leaked")]
        t1 = time.time()
        try:
            loc = localize_leak(code, data, cut_positions=leaking or None)
            err = ""
        except Exception as e:  # noqa: BLE001
            loc, err = {"found": False}, f"{type(e).__name__}: {e}"
        t_loc = time.time() - t1
        truth = TRUTH.get(sid)
        line = loc.get("entry", {}).get("line") if loc.get("found") else None
        row = {"id": sid, "file": f.name, "label": "leaky" if sid.startswith("L") else "clean",
               "perturbation_failed": not fp.passed, "truth_lines": list(truth[0]) if truth else [],
               "truth_desc": truth[1] if truth else "", "found": bool(loc.get("found")), "line": line,
               "var": loc.get("entry", {}).get("var"), "source_line": loc.get("entry", {}).get("source_line"),
               "kind": loc.get("entry", {}).get("kind"),
               "first_changed_date": loc.get("entry", {}).get("first_changed_date"),
               "chain_lines": [c["line"] for c in loc.get("chain", [])], "agreement": loc.get("agreement"),
               "audit_s": round(t_audit, 2), "localize_s": round(t_loc, 2), "error": err}
        row["correct"] = (line in truth[0]) if truth else (not loc.get("found"))
        rows.append(row)
        log(f"{sid}: line={line} truth={row['truth_lines']} correct={row['correct']} "
            f"({t_audit:.1f}s audit, {t_loc:.1f}s localize){' ' + err if err else ''}")
    return {"rows": rows}


def write_md(res: dict, path: Path = OUT_MD) -> None:
    rows = res["rows"]
    leaky = [r for r in rows if r["label"] == "leaky"]
    clean = [r for r in rows if r["label"] == "clean"]
    hit = sum(r["correct"] for r in leaky)
    blind = [r for r in leaky if r["id"] in STATIC_BLIND]
    out = ["# LeakBench: dynamic leak localization", "",
           "The localizer (`audit/localize.py`) AST-instruments the strategy, runs it in the sandbox on the "
           "original panel and on copies scrambled after the leaking cut dates, and reports the first statement "
           "(in execution order, on the chain that reaches the returned weights) whose value at dates <= cut "
           "changes although nothing it reads had changed. Ground truth = the line where the labelled bug is "
           "written. Reproduce with `python -m bench.run_localize`.", "",
           f"**Leaky strategies: {hit}/{len(leaky)} localized to the exact bug line.** "
           f"Static-scan-blind cases (L17, L18, L19, L22): {sum(r['correct'] for r in blind)}/{len(blind)}.", ""]
    if clean:
        fp = [r for r in clean if r["found"]]
        out += [f"Clean controls: {len(clean) - len(fp)}/{len(clean)} report no leak "
                f"(the localizer is only run when future-perturbation fails; here it is forced on clean code too).", ""]
    secs = [r["localize_s"] for r in rows]
    aud = [r["audit_s"] for r in rows]
    if secs:
        out += [f"Runtime: localization {sum(secs) / len(secs):.2f}s per strategy on average "
                f"(max {max(secs):.2f}s) vs {sum(aud) / len(aud):.2f}s for the full audit without it.", ""]
    out += ["| Strategy | Bug | Truth line | Localized line | Variable | Source | Kind | Chain | Correct |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in leaky + clean:
        src = (r["source_line"] or "").replace("|", "\\|")
        out.append(f"| {r['file'][:-3]} | {r['truth_desc'].replace('|', '/') or '-'} | "
                   f"{', '.join(map(str, r['truth_lines'])) or '-'} | {r['line'] if r['found'] else 'none'} | "
                   f"{r['var'] or ''} | `{src}` | {r['kind'] or ''} | "
                   f"{' > '.join(map(str, r['chain_lines']))} | {'yes' if r['correct'] else 'NO'} |")
    misses = [r for r in leaky if not r["correct"]]
    if misses:
        out += ["", "## Misses", ""]
        for r in misses:
            out.append(f"- **{r['id']}**: localized {r['line']} (`{r['source_line']}`), truth "
                       f"{r['truth_lines']}. {r['error']}")
    out += ["", "## Limits", "",
            "- Taint follows variable names (plus helper-function returns), not control flow: a leak that only "
            "steers an `if` is attributed to the first assigned value that changes.",
            "- A leak written inline in a `return` expression is attributed to the `return` line (L09, L12 here, "
            "which is the right line).",
            "- Only cuts where the weights actually changed are traced (at most 2, chosen from the audit's "
            "leaking cuts); a leak that exists at no tested cut is not localized.",
            "- Loops: every occurrence is compared up to 100, then every 25th; the first diverging occurrence "
            "is reported."]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaky-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these id prefixes (e.g. L17 C0) and merge into the "
                                              "existing results file")
    a = ap.parse_args(argv)
    res = run(a.leaky_only, only=a.only)
    if a.only and OUT_JSON.exists():  # merge partial runs
        old = {r["id"]: r for r in json.loads(OUT_JSON.read_text())["rows"]}
        old.update({r["id"]: r for r in res["rows"]})
        res = {"rows": [old[k] for k in sorted(old, key=lambda k: (k[0] != "L", k))]}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    write_md(res)
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

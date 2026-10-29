"""perturb_after: post-cut NaN gaps and relisting expose back-fill / interpolation leaks."""
import numpy as np
import pandas as pd

from audit.leak_tests import in_process_executor, perturb_after, run_audit
from data.synthetic import FIELDS


def _cut_with_gone(panel):
    c = panel["close"].to_numpy()
    T = len(c)
    for cut in range(int(0.3 * T), int(0.9 * T)):
        if any(np.isnan(c[cut, j]) and np.isfinite(c[:cut, j]).any() for j in range(c.shape[1])):
            return cut
    raise AssertionError("panel has no name delisted before a usable cut")


def test_rows_up_to_cut_identical_and_deterministic(small_panel):
    cut = _cut_with_gone(small_panel)
    a = perturb_after(small_panel, cut, np.random.default_rng(3))
    b = perturb_after(small_panel, cut, np.random.default_rng(3))
    for f in FIELDS:
        x, y = small_panel[f].to_numpy()[: cut + 1], a[f].to_numpy()[: cut + 1]
        assert np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(np.nan_to_num(x), np.nan_to_num(y)), f
        assert a[f].equals(b[f]), f


def test_gaps_and_relisting_only_after_cut(small_panel):
    cut = _cut_with_gone(small_panel)
    c0 = small_panel["close"].to_numpy()
    p = perturb_after(small_panel, cut, np.random.default_rng(5), n_kill=0)["close"].to_numpy()
    post0, post = np.isnan(c0[cut + 1:]), np.isnan(p[cut + 1:])
    assert (post & ~post0).any(), "no new NaN gap after the cut"
    assert (~post & post0).any(), "no delisted name trades again after the cut"
    # interior gap: a survivor has NaN after the cut and a valid value later
    gap_cols = np.where((post & ~post0).any(axis=0))[0]
    assert any(np.isfinite(p[cut + 1:, j][np.argmax(post[:, j]):]).any() for j in gap_cols)


def _audit(fn, panel):
    return {t.name: t for t in run_audit(panel, in_process_executor(fn), None).tests}


def test_bfill_and_interpolate_caught_by_perturbation(small_panel):
    def bfill_mom(data):
        px = data["close"].bfill()
        mom = px / px.shift(20) - 1
        r = mom.rank(axis=1, ascending=False)
        pick = (r <= 3).astype(float)
        return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    def interp_mom(data):
        px = data["close"].interpolate()
        mom = (px / px.shift(20) - 1).where(px.notna())
        r = mom.rank(axis=1, ascending=False)
        pick = (r <= 3).astype(float)
        return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    def ffill_mom(data):  # honest twin: forward fill only
        px = data["close"].ffill()
        mom = (px / px.shift(20) - 1).where(data["close"].notna())
        r = mom.rank(axis=1, ascending=False)
        pick = (r <= 3).astype(float)
        return pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    assert not _audit(bfill_mom, small_panel)["future_perturbation"].passed
    assert not _audit(interp_mom, small_panel)["future_perturbation"].passed
    assert _audit(ffill_mom, small_panel)["future_perturbation"].passed

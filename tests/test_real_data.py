import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.mock_strategies import CLEAN
from audit.leak_tests import in_process_executor, run_audit
from data import FIELDS, available_universes, listing_table, load_real_panel
from data.real import DATA_FILE, IPO_TICKERS, REAL_TICKERS


@pytest.fixture(scope="module")
def real():
    return load_real_panel()


def _fn(code):
    g = {}
    exec(code, g)
    return g["generate_signals"]


def _by(rep):
    return {t.name: t for t in rep.tests}


def test_file_bundled_and_small():
    assert DATA_FILE.exists()
    assert DATA_FILE.stat().st_size < 3_000_000


def test_shape_and_fields(real):
    close = real["close"]
    assert set(real) == set(FIELDS)
    assert list(close.columns) == sorted(REAL_TICKERS)
    assert close.shape[1] == 40 and close.shape[0] > 2900
    assert isinstance(close.index, pd.DatetimeIndex) and close.index.is_monotonic_increasing
    for f in FIELDS:
        assert real[f].index.equals(close.index) and real[f].columns.equals(close.columns)
        assert (real[f].isna() == close.isna()).all().all()
    h, l, o, c = (real[k].to_numpy() for k in ("high", "low", "open", "close"))
    m = ~np.isnan(c)
    assert (c[m] > 0).all()
    assert (h[m] >= np.maximum(o[m], c[m]) - 1e-9).all()
    assert (l[m] <= np.minimum(o[m], c[m]) + 1e-9).all()


def test_no_future_dates(real):
    idx = real["close"].index
    assert idx[0] >= pd.Timestamp("2014-01-01")
    assert idx[-1] <= pd.Timestamp("2025-12-31")
    assert idx[-1] <= pd.Timestamp.today().normalize()
    assert not idx.duplicated().any()


def test_nan_before_ipo(real):
    lt = listing_table(real)
    for t in IPO_TICKERS:
        first = lt.loc[t, "first_date"]
        assert first > real["close"].index[0], t
        assert real["close"].loc[:first, t].iloc[:-1].isna().all(), t
        assert real["close"].loc[first:, t].notna().all(), t  # continuous after listing
    assert lt.loc["UBER", "first_date"] == pd.Timestamp("2019-05-10")
    assert lt.loc["RIVN", "first_date"] == pd.Timestamp("2021-11-10")
    # free data has no delisted names: the dataset's own survivorship bias
    assert not lt["delisted"].any()


def test_subset_and_dates():
    p = load_real_panel(["AAPL", "UBER"], start="2019-01-01", end="2019-12-31")
    assert list(p["close"].columns) == ["AAPL", "UBER"]
    assert p["close"].index.min() >= pd.Timestamp("2019-01-01")
    assert p["close"].index.max() <= pd.Timestamp("2019-12-31")
    assert p["close"]["UBER"].loc[:"2019-05-09"].isna().all()
    with pytest.raises(KeyError):
        load_real_panel(["NOPE"])


def test_registry():
    u = available_universes()
    assert "Synthetic (seeded)" in u and "Real US equities (40, 2014–2025)" in u
    for v in u.values():
        assert isinstance(v["description"], str) and v["description"]
    assert u["Real US equities (40, 2014–2025)"]["loader"]()["close"].shape[1] == 40


def test_leaky_example_flagged_on_real(real):
    code = (Path(__file__).resolve().parents[1] / "examples" / "leaky_momentum.py").read_text()
    t0 = time.time()
    rep = run_audit(real, in_process_executor(_fn(code)), code)
    assert time.time() - t0 < 30
    t = _by(rep)
    assert not rep.clean
    assert not t["future_perturbation"].passed and t["future_perturbation"].severity == "critical"
    assert not t["static_scan"].passed
    assert not t["plausibility"].passed
    surv = t["survivorship"]
    assert "data_warning" in surv.evidence and not surv.blocking   # honest: dataset has no delisted names
    assert rep.metrics["sharpe"] > 3.0


def test_clean_momentum_passes_on_real(real):
    rep = run_audit(real, in_process_executor(_fn(CLEAN["momentum"])), CLEAN["momentum"])
    t = _by(rep)
    assert rep.clean, [x.to_dict() for x in rep.tests if x.blocking]
    assert t["future_perturbation"].passed
    assert 0.0 < rep.metrics["sharpe"] < 2.0

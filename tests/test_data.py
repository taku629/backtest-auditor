import numpy as np

from data import FIELDS, generate_panel, listing_table, load_long_csv


def test_deterministic():
    a, b = generate_panel(), generate_panel()
    for f in FIELDS:
        assert a[f].equals(b[f])


def test_shape_and_lifecycle(panel):
    close = panel["close"]
    assert close.shape == (252 * 6, 50)
    lt = listing_table(panel)
    assert lt["delisted"].sum() == 8
    assert lt["ipo_after_start"].sum() == 6
    # OHLC consistency where listed
    h, l, o, c = (panel[k].to_numpy() for k in ("high", "low", "open", "close"))
    m = ~np.isnan(c)
    assert (h[m] >= np.maximum(o[m], c[m]) - 1e-9).all()
    assert (l[m] <= np.minimum(o[m], c[m]) + 1e-9).all()
    # NaN pattern identical across fields
    for f in FIELDS:
        assert (panel[f].isna() == close.isna()).all().all()


def test_csv_roundtrip(tmp_path, small_panel):
    long = small_panel["close"].stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    p = tmp_path / "px.csv"
    long.to_csv(p, index=False)
    d = load_long_csv(p)
    assert set(d) == set(FIELDS)
    assert np.allclose(d["close"].to_numpy(), small_panel["close"].dropna(how="all").reindex(d["close"].index)
                       .reindex(columns=d["close"].columns).to_numpy(), equal_nan=True)

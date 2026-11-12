"""Loaders are tested on tiny files with the real datasets' column layouts."""
import numpy as np
import pandas as pd

from potf.loaders import bustabit, dunnhumby, elliptic, ipl, recruit
from potf.pipeline import run


def _dunnhumby(tmp_path, stores=12, days=70, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    bid = 0
    for st in range(stores):
        hh = np.arange(st * 1000, st * 1000 + 300)
        for d in range(1, days + 1):
            for _ in range(rng.poisson(40)):
                bid += 1
                h = rng.choice(hh)
                for _ in range(rng.integers(1, 4)):
                    rows.append((h, bid, d, round(rng.gamma(2, 1.5), 2), st, int(rng.integers(700, 2300))))
    df = pd.DataFrame(rows, columns=["household_key", "BASKET_ID", "DAY", "SALES_VALUE", "STORE_ID", "TRANS_TIME"])
    df["PRODUCT_ID"] = 1
    f = tmp_path / "transaction_data.csv"
    df.to_csv(f, index=False)
    return f


def _bustabit(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    n = 4000
    t0 = pd.Timestamp("2016-11-01")
    df = pd.DataFrame({
        "Id": range(n), "GameID": rng.integers(1, 500, n), "Username": rng.choice([f"u{i}" for i in range(200)], n),
        "Bet": rng.choice([100, 200, 500, 1000, 50], n), "CashedOut": 1.5, "Bonus": 0, "Profit": 0, "BustedAt": 2.0,
        "PlayDate": [str(t0 + pd.Timedelta(minutes=int(m))) for m in rng.integers(0, 60 * 24 * 30, n)],
    })
    f = tmp_path / "bustabit.csv"
    df.to_csv(f, index=False)
    return f


def test_dunnhumby_panel_shape(tmp_path):
    panel, agg = dunnhumby.load_panel(_dunnhumby(tmp_path), days=60)
    assert panel.X.shape[1:] == (60, 6)
    assert panel.M >= 10
    assert np.isfinite(panel.X).all()


def test_bustabit_injection_runs_end_to_end(tmp_path):
    panel, agg = dunnhumby.load_panel(_dunnhumby(tmp_path), days=60)
    bets = bustabit.load_bets(_bustabit(tmp_path))
    payers = dict(zip(panel.merchants.index, panel.payers))
    p2 = bustabit.inject(agg, panel.merchants, bets, n_fronts=3, onset_range=(32, 40), payers=payers)
    assert p2.merchants.is_front.sum() == 3
    res = run(p2)
    assert len(res.summary) == p2.M


def test_ipl_match_days(tmp_path):
    f = tmp_path / "matches.csv"
    pd.DataFrame({"id": [1, 2, 3], "season": [2023, 2024, 2024],
                  "date": ["2023-04-01", "2024-03-22", "2024-03-23"]}).to_csv(f, index=False)
    assert ipl.match_days(f) == {"2026-03-22", "2026-03-23"}


def test_recruit_uplift(tmp_path):
    dates = pd.date_range("2017-01-01", periods=28)
    v = pd.DataFrame([(s, d.strftime("%Y-%m-%d"), 20 + (10 if d.dayofweek >= 5 else 0) + (8 if d.day == 9 else 0))
                      for s in ["a", "b"] for d in dates], columns=["air_store_id", "visit_date", "visitors"])
    di = pd.DataFrame({"calendar_date": dates.strftime("%Y-%m-%d"), "day_of_week": dates.day_name(),
                       "holiday_flg": (dates.day == 9).astype(int)})
    (tmp_path / "v.csv").write_text(v.to_csv(index=False))
    (tmp_path / "d.csv").write_text(di.to_csv(index=False))
    up = recruit.calendar_uplift(tmp_path / "v.csv", tmp_path / "d.csv")
    assert up["weekend"] > 0.3 and up["festival"] > 0.2


def test_elliptic_benchmark(tmp_path):
    cls = pd.DataFrame({"txId": range(12), "class": ["1"] * 4 + ["2"] * 6 + ["unknown"] * 2})
    edges = pd.DataFrame({"txId1": [0, 1, 2, 4, 5, 6], "txId2": [1, 2, 3, 5, 6, 7]})
    cls.to_csv(tmp_path / "c.csv", index=False)
    edges.to_csv(tmp_path / "e.csv", index=False)
    out = elliptic.fleet_benchmark(tmp_path / "c.csv", tmp_path / "e.csv")
    assert out["P(neighbour illicit | node illicit)"] == 1.0
    assert out["illicit_components_ge_min"] == 1

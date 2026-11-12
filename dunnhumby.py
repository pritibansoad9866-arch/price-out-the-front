"""dunnhumby "The Complete Journey": real grocery baskets -> legitimate store panel.

File: transaction_data.csv
Columns used: household_key, BASKET_ID, DAY, SALES_VALUE, STORE_ID, TRANS_TIME
Amounts are converted to rupees with `fx` so the node's round-amount rule applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._io import read_csv_ci
from .aggregates import panel_from_aggregates

COLS = ["household_key", "BASKET_ID", "DAY", "SALES_VALUE", "STORE_ID", "TRANS_TIME"]


def basket_table(path, fx: float = 83.0) -> pd.DataFrame:
    df = read_csv_ci(path, COLS)
    b = df.groupby(["STORE_ID", "DAY", "BASKET_ID", "household_key"], as_index=False).agg(
        value=("SALES_VALUE", "sum"), time=("TRANS_TIME", "min"))
    b["value"] = b["value"] * fx
    return b


def aggregates(baskets: pd.DataFrame, start: str = "2026-03-01", first_day: int | None = None,
               days: int = 120, min_active: float = 0.8):
    b = baskets.copy()
    first_day = int(b["DAY"].min()) if first_day is None else first_day
    b = b[(b["DAY"] >= first_day) & (b["DAY"] < first_day + days)]
    b["date"] = pd.Timestamp(start) + pd.to_timedelta(b["DAY"] - first_day, unit="D")
    hh = b.sort_values(["STORE_ID", "DAY"])
    first_seen = hh.groupby(["STORE_ID", "household_key"])["DAY"].transform("min")
    b["is_new"] = (b["DAY"] == first_seen) & (b["DAY"] > first_day + 6)
    b["is_round"] = (np.round(b["value"]) % 100 == 0) & (b["value"] >= 500)
    b["is_night"] = (b["time"] >= 2200) | (b["time"] < 400)
    g = b.groupby(["STORE_ID", "date"])
    agg = pd.DataFrame({
        "n": g.size(), "value": g["value"].sum(), "n_round": g["is_round"].sum(),
        "n_night": g["is_night"].sum(), "n_new": g["is_new"].sum(),
    })
    top = (b.groupby(["STORE_ID", "date", "household_key"])["value"].sum()
           .groupby(level=[0, 1]).apply(lambda s: s.nlargest(5).sum()))
    agg["top5_value"] = top
    active = agg.groupby(level=0).size() / days
    keep = active[active >= min_active].index
    agg = agg.loc[agg.index.get_level_values(0).isin(keep)]
    agg.index = agg.index.set_levels([agg.index.levels[0].astype(str).map(lambda s: f"D{s}"), agg.index.levels[1]])
    payers = b[b["STORE_ID"].isin(keep)].groupby("STORE_ID")["household_key"].unique()
    payers.index = payers.index.astype(str).map(lambda s: f"D{s}")
    return agg, payers.to_dict()


def load_panel(path, days: int = 120, start: str = "2026-03-01", fx: float = 83.0, opt_in_rate: float = 0.55,
               seed: int = 0, match_days: set | None = None):
    rng = np.random.default_rng(seed)
    agg, payers = aggregates(basket_table(path, fx), start=start, days=days)
    ids = agg.index.get_level_values(0).unique()
    vol = agg.groupby(level=0)["n"].median().reindex(ids)
    scale = pd.qcut(vol.rank(method="first"), 3, labels=["small", "medium", "large"]).astype(str)
    merchants = pd.DataFrame({"node": "kirana", "geo": "dunnhumby", "scale": scale.values,
                              "opted_in": rng.random(len(ids)) < opt_in_rate}, index=pd.Index(ids, name="merchant_id"))
    return panel_from_aggregates(agg, merchants, payers, match_days=match_days), agg

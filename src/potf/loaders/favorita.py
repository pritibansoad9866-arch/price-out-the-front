"""Corporacion Favorita grocery sales: real holiday uplift for grocery stores.

Files: train.csv (date, store_nbr, unit_sales; large, read in chunks), holidays_events.csv (date, type, locale, transferred)
"""
from __future__ import annotations

import pandas as pd

from ._io import read_csv_ci


def store_daily(train_path, chunksize: int = 5_000_000, max_chunks: int | None = None) -> pd.DataFrame:
    parts = []
    head = pd.read_csv(train_path, nrows=0)
    lower = {c.lower(): c for c in head.columns}
    cols = [lower["date"], lower["store_nbr"], lower["unit_sales"]]
    for k, ch in enumerate(pd.read_csv(train_path, usecols=cols, chunksize=chunksize, parse_dates=[cols[0]])):
        ch.columns = ["date", "store_nbr", "unit_sales"]
        parts.append(ch.groupby(["store_nbr", "date"])["unit_sales"].sum())
        if max_chunks and k + 1 >= max_chunks:
            break
    return pd.concat(parts).groupby(level=[0, 1]).sum().rename("sales").reset_index()


def holiday_uplift(train_path, holidays_path, **kw) -> dict:
    s = store_daily(train_path, **kw)
    h = read_csv_ci(holidays_path, ["date", "type", "locale", "transferred"], parse_dates=["date"])
    nat = h[(h["locale"] == "National") & (h["transferred"].astype(str).str.lower() != "true")
            & (h["type"].isin(["Holiday", "Additional", "Bridge", "Event"]))]["date"]
    s["holiday"] = s["date"].isin(nat)
    s["weekday"] = s["date"].dt.dayofweek
    base = s[~s["holiday"]].groupby(["store_nbr", "weekday"])["sales"].mean().rename("base")
    s = s.join(base, on=["store_nbr", "weekday"])
    up = (s.loc[s["holiday"], "sales"] / s.loc[s["holiday"], "base"]).median() - 1
    return {"festival": float(up), "stores": int(s["store_nbr"].nunique())}

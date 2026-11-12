"""Recruit Restaurant Visitor Forecasting: real restaurant calendar effects.

Files: air_visit_data.csv (air_store_id, visit_date, visitors), date_info.csv (calendar_date, holiday_flg)
Returns holiday and weekend uplift to calibrate the restaurant node's `calendar`.
"""
from __future__ import annotations

import pandas as pd

from ._io import read_csv_ci


def calendar_uplift(visits_path, dates_path) -> dict:
    v = read_csv_ci(visits_path, ["air_store_id", "visit_date", "visitors"], parse_dates=["visit_date"])
    d = read_csv_ci(dates_path, ["calendar_date", "holiday_flg"], parse_dates=["calendar_date"])
    v = v.merge(d, left_on="visit_date", right_on="calendar_date", how="left")
    v["weekend"] = v["visit_date"].dt.dayofweek >= 5
    base = v[(~v["weekend"]) & (v["holiday_flg"] == 0)].groupby("air_store_id")["visitors"].mean()
    hol = v[(~v["weekend"]) & (v["holiday_flg"] == 1)].groupby("air_store_id")["visitors"].mean()
    wkd = v[v["weekend"] & (v["holiday_flg"] == 0)].groupby("air_store_id")["visitors"].mean()
    return {"festival": float((hol / base).median() - 1), "weekend": float((wkd / base).median() - 1),
            "stores": int(base.size)}

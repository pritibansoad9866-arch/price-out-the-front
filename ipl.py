"""IPL match data: the real match calendar for match-night effects.

File: matches.csv (columns include season/Season and date)
"""
from __future__ import annotations

import pandas as pd

from ._io import read_csv_ci


def match_days(path, season=None, to_year: int | None = 2026) -> set:
    head = pd.read_csv(path, nrows=0)
    season_col = next((c for c in head.columns if c.lower() == "season"), None)
    needed = ["date"] + ([season_col] if season_col else [])
    df = read_csv_ci(path, needed)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["date"])
    if season_col:
        seasons = df[season_col].astype(str)
        pick = str(season) if season is not None else sorted(seasons.unique())[-1]
        df = df[seasons == pick]
    days = df["date"]
    if to_year is not None:
        days = days.map(lambda d: d.replace(year=to_year))
    return set(days.dt.strftime("%Y-%m-%d"))

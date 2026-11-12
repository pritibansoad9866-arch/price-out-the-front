"""Bustabit: real online betting records -> betting flows injected into real stores.

File: bustabit.csv
Columns used: Username, Bet, PlayDate
This is the semi-real front: a real store's real baskets, plus a real betting
book run through the same account from `onset` onward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._io import read_csv_ci
from .aggregates import AGG_COLS, panel_from_aggregates


def load_bets(path, bet_to_rs: float = 1.0) -> pd.DataFrame:
    df = read_csv_ci(path, ["Username", "Bet", "PlayDate"])
    df["PlayDate"] = pd.to_datetime(df["PlayDate"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["PlayDate"])
    df["amount"] = df["Bet"] * bet_to_rs
    return df


def betting_days(bets: pd.DataFrame, n_bettors: int, rng, round_to: int = 100) -> list[pd.DataFrame]:
    """Daily aggregates of a random bettor group. Stakes are paid in round rupee amounts, as top-ups are."""
    users = bets["Username"].unique()
    pick = rng.choice(users, size=min(n_bettors, len(users)), replace=False)
    b = bets[bets["Username"].isin(pick)].copy()
    b["amount"] = np.maximum(np.round(b["amount"] / round_to) * round_to, round_to)
    b["day"] = b["PlayDate"].dt.floor("D")
    b["hour"] = b["PlayDate"].dt.hour
    return [d for _, d in b.groupby("day")]


def inject(agg: pd.DataFrame, merchants: pd.DataFrame, bets: pd.DataFrame, n_fronts: int = 10,
           onset_range=(40, 90), bettors=(15, 40), seed: int = 0, payers: dict | None = None,
           match_days: set | None = None):
    """Turn n_fronts real stores into fronts by adding real betting days from onset on."""
    rng = np.random.default_rng(seed)
    agg = agg.copy()
    merchants = merchants.copy()
    merchants["is_front"] = False
    merchants["front_type"] = ""
    merchants["onset"] = -1
    merchants["syndicate"] = -1
    merchants["benign_change"] = ""
    dates = sorted(agg.index.get_level_values(1).unique())
    chosen = rng.choice(merchants.index, size=n_fronts, replace=False)
    seen_all = {}
    for mid in chosen:
        onset = int(rng.integers(*onset_range))
        days = betting_days(bets, int(rng.integers(*bettors)), rng)
        seen = set()
        for k, t in enumerate(range(onset, len(dates))):
            if not days:
                break
            d = days[k % len(days)]
            new = (~d["Username"].isin(seen)).sum()
            seen.update(d["Username"])
            add = pd.Series({
                "n": len(d), "value": d["amount"].sum(), "n_round": len(d),
                "n_night": int(((d["hour"] >= 22) | (d["hour"] < 4)).sum()), "n_new": int(new),
                "top5_value": d.groupby("Username")["amount"].sum().nlargest(5).sum(),
            })[AGG_COLS]
            key = (mid, dates[t])
            if key in agg.index:
                row = agg.loc[key, AGG_COLS]
                top5 = max(row["top5_value"], add["top5_value"])
                agg.loc[key, AGG_COLS] = (row + add).to_numpy()
                agg.loc[key, "top5_value"] = top5
        merchants.loc[mid, ["is_front", "front_type", "onset"]] = [True, "betting", onset]
        seen_all[mid] = seen
    if payers is not None:
        for mid, s in seen_all.items():
            extra = np.array([hash(u) % (10**9) + 10**10 for u in s], dtype=np.int64)
            payers[mid] = np.concatenate([np.asarray(payers.get(mid, []), dtype=np.int64), extra])
    panel = panel_from_aggregates(agg, merchants, payers, match_days=match_days)
    return panel

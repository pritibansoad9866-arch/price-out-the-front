"""Merchant-day aggregates -> features, and -> Panel.

Loaders produce counts (so real flows can be mixed exactly, e.g. injecting
real bets into a real store), then features are derived the same way for all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..nodes import FEATURES
from ..panel import Panel, build_calendar

AGG_COLS = ["n", "value", "n_round", "n_night", "n_new", "top5_value"]


def features_from_aggregates(a: pd.DataFrame) -> pd.DataFrame:
    n = a["n"].clip(lower=1)
    out = pd.DataFrame(index=a.index)
    out["log_txn"] = np.log1p(a["n"])
    out["log_ticket"] = np.log((a["value"] / n).clip(lower=1e-6))
    out["round_share"] = a["n_round"] / n
    out["night_share"] = a["n_night"] / n
    out["new_payer_share"] = a["n_new"] / n
    out["top_payer_share"] = (a["top5_value"] / a["value"].clip(lower=1e-9)).clip(0, 1)
    return out[FEATURES]


def panel_from_aggregates(agg: pd.DataFrame, merchants: pd.DataFrame, payers: dict | None = None,
                          match_days: set | None = None) -> Panel:
    """agg: index (merchant_id, date) with AGG_COLS. merchants: index merchant_id with node, geo, scale, opted_in."""
    dates = pd.DatetimeIndex(sorted(agg.index.get_level_values(1).unique()))
    ids = merchants.index
    full = agg.reindex(pd.MultiIndex.from_product([ids, dates]), fill_value=0)
    feats = features_from_aggregates(full)
    X = feats.to_numpy().reshape(len(ids), len(dates), len(FEATURES))
    cal = build_calendar(dates, match_days=match_days)
    pl = [np.asarray(payers.get(i, []), dtype=np.int64) for i in ids] if payers else []
    return Panel(merchants=merchants, X=X, dates=dates, calendar=cal, payers=pl)

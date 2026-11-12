"""The common data shape every source (simulator or real data) is turned into.

A Panel is merchants x days x features. Ground-truth columns (is_front, ...)
exist only for simulated or injected data and are never read by the
detection pipeline; they are used only to score it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .nodes import FEATURES

# Illustrative 2026 calendar. Replace with real dates (see loaders.ipl / loaders.favorita).
FESTIVALS_2026 = {
    "2026-03-04": "Holi", "2026-03-20": "Eid al-Fitr", "2026-03-26": "Ram Navami",
    "2026-04-14": "Baisakhi", "2026-04-19": "Akshaya Tritiya", "2026-05-27": "Eid al-Adha",
}
IPL_WINDOW_2026 = ("2026-03-26", "2026-05-31")


@dataclass
class Panel:
    merchants: pd.DataFrame          # index: merchant_id; columns: node, geo, scale, opted_in (+ truth)
    X: np.ndarray                    # [M, T, F] daily features, NaN where unavailable
    dates: pd.DatetimeIndex
    calendar: pd.DataFrame           # index: date; columns: festival, salary, weekend, match
    payers: list = field(default_factory=list)  # per merchant: array of payer ids (for the fleet graph)

    @property
    def M(self) -> int:
        return self.X.shape[0]

    @property
    def T(self) -> int:
        return self.X.shape[1]

    def has_truth(self) -> bool:
        return "is_front" in self.merchants.columns

    def frame(self, merchant: int) -> pd.DataFrame:
        return pd.DataFrame(self.X[merchant], index=self.dates, columns=FEATURES)

    def to_long(self) -> pd.DataFrame:
        idx = pd.MultiIndex.from_product([self.merchants.index, self.dates], names=["merchant_id", "date"])
        return pd.DataFrame(self.X.reshape(-1, self.X.shape[2]), index=idx, columns=FEATURES)


def build_calendar(dates: pd.DatetimeIndex, match_days: set | None = None, festivals: dict | None = None,
                   seed: int = 0) -> pd.DataFrame:
    festivals = FESTIVALS_2026 if festivals is None else festivals
    fest = pd.to_datetime(list(festivals.keys()))
    cal = pd.DataFrame(index=dates)
    cal["festival"] = dates.isin(fest).astype(float)
    cal["salary"] = (dates.day <= 5).astype(float)
    cal["weekend"] = (dates.dayofweek >= 5).astype(float)
    if match_days is None:
        rng = np.random.default_rng(seed)
        lo, hi = pd.to_datetime(IPL_WINDOW_2026[0]), pd.to_datetime(IPL_WINDOW_2026[1])
        in_window = (dates >= lo) & (dates <= hi)
        cal["match"] = (in_window & (rng.random(len(dates)) < 0.9)).astype(float)
    else:
        md = pd.to_datetime(sorted(match_days))
        cal["match"] = dates.isin(md).astype(float)
    cal["festival_name"] = [festivals.get(d.strftime("%Y-%m-%d"), "") for d in dates]
    return cal


def save_panel(panel: Panel, path) -> None:
    from pathlib import Path
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p / "features.npz", X=panel.X, dates=panel.dates.strftime("%Y-%m-%d").to_numpy(),
                        payers=np.array(panel.payers, dtype=object), allow_pickle=True)
    panel.merchants.to_csv(p / "merchants.csv")
    panel.calendar.to_csv(p / "calendar.csv")


def load_panel(path) -> Panel:
    from pathlib import Path
    p = Path(path)
    z = np.load(p / "features.npz", allow_pickle=True)
    m = pd.read_csv(p / "merchants.csv", index_col=0)
    for c in ("front_type", "benign_change"):
        if c in m:
            m[c] = m[c].fillna("")
    cal = pd.read_csv(p / "calendar.csv", index_col=0, parse_dates=True)
    return Panel(merchants=m, X=z["X"], dates=pd.DatetimeIndex(pd.to_datetime(z["dates"])), calendar=cal,
                 payers=list(z["payers"]))

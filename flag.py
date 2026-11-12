"""F-test-esque first flag.

A cheap screen run on everyone, every day. It asks only "is anything
different?": the merchant's squared deviations from peers over a rolling
window, scaled by peer noise, behave like a chi-square (an F ratio against
peers). A flag costs a legitimate merchant nothing; it only starts the cascade.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .peers import Baseline


def first_flag(b: Baseline, window: int = 14, p_cut: float = 0.01):
    z = b.z
    M, T, F = z.shape
    sq = np.nan_to_num(z**2)
    cnt = (~np.isnan(z)).astype(float)
    cs = np.cumsum(sq.sum(axis=2), axis=1)
    cc = np.cumsum(cnt.sum(axis=2), axis=1)
    lag_s = np.zeros_like(cs)
    lag_c = np.zeros_like(cc)
    lag_s[:, window:] = cs[:, :-window]
    lag_c[:, window:] = cc[:, :-window]
    S, df = cs - lag_s, cc - lag_c
    with np.errstate(invalid="ignore", divide="ignore"):
        p_change = np.where(df > 0, stats.chi2.sf(S, np.maximum(df, 1)), 1.0)
        f_ratio = np.where(df > 0, S / np.maximum(df, 1), np.nan)
    p_level = stats.chi2.sf(np.nansum(b.u**2, axis=1), F)
    flagged = (p_change < p_cut) | (p_level[:, None] < p_cut)
    flagged[:, : b.burn_in] = False
    return {"p_change": p_change, "p_level": p_level, "f_ratio": f_ratio, "flag": flagged}

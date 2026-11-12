"""Hierarchical peer baselines.

Every merchant is compared with its groups: geographic, industrial and scale
(node x geo x scale band). Small groups borrow strength from the node level.
Comparing against peers on the *same day* removes shocks every peer shares:
festivals, salary days, weekends and (for sports bars) match nights.

Day-to-day noise depends on how many payments a merchant had that day
(a share computed from 20 payments is noisier than one from 400), so the
noise model is  var_f(n) = tau_f^2 + kappa_f / n,  fitted per peer group, then
scaled per merchant (shrunk toward the group). ``inflation`` is a safety margin
that shadow mode calibrates so realised false escalations stay inside budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..panel import Panel

MAD = 1.4826


@dataclass
class Baseline:
    group: np.ndarray        # [M] peer group used for each merchant
    med: np.ndarray          # [M, T, F] peer median on each day, aligned to merchants
    offset: np.ndarray       # [M, F] merchant's own level vs peers during burn-in
    tau2: np.ndarray         # [M, F] noise floor
    kappa: np.ndarray        # [M, F] count-driven noise
    sd_day: np.ndarray       # [M, T, F] noise sd on each day
    sigma_w: np.ndarray      # [M, F] typical day noise (at the merchant's median count)
    sigma_lv: np.ndarray     # [M, F] sd of the onboarding level among honest peers
    z_raw: np.ndarray        # [M, T, F] (x - peer median) / day noise   (level + change)
    z: np.ndarray            # [M, T, F] one-step-ahead residual after burn-in, ~N(0, 1) when honest
    u: np.ndarray            # [M, F] standardised onboarding level, ~N(0, 1) when honest
    burn_in: int


def _robust_sd(a, axis=None):
    med = np.nanmedian(a, axis=axis, keepdims=True)
    return MAD * np.nanmedian(np.abs(a - med), axis=axis)


def _group_keys(m: pd.DataFrame, min_group: int) -> np.ndarray:
    k1 = (m["node"] + "|" + m["geo"] + "|" + m["scale"]).to_numpy()
    k2 = (m["node"] + "|*|" + m["scale"]).to_numpy()
    k3 = (m["node"] + "|*|*").to_numpy()
    out = k1.copy()
    c1 = pd.Series(k1).map(pd.Series(k1).value_counts()).to_numpy()
    c2 = pd.Series(k2).map(pd.Series(k2).value_counts()).to_numpy()
    small1, small2 = c1 < min_group, c2 < min_group
    out[small1] = k2[small1]
    out[small1 & small2] = k3[small1 & small2]
    return out


def _fit_noise(r: np.ndarray, inv_n: np.ndarray):
    """Fit r^2 = tau^2 + kappa / n per feature with a trimmed least squares. r, inv_n: [N, F]."""
    F = r.shape[1]
    tau2, kap = np.zeros(F), np.zeros(F)
    for f in range(F):
        y, x = r[:, f] ** 2, inv_n[:, f]
        ok = np.isfinite(y) & np.isfinite(x)
        y, x = y[ok], x[ok]
        if len(y) < 10:
            tau2[f], kap[f] = np.nanmean(y) if len(y) else 1e-4, 0.0
            continue
        keep = y <= np.quantile(y, 0.99)
        A = np.stack([np.ones(keep.sum()), x[keep]], 1)
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        t, k = coef
        if k < 0:
            t, k = y[keep].mean(), 0.0
        if t < 0:
            t, k = 0.0, max((y[keep] * x[keep]).sum() / (x[keep] ** 2).sum(), 0.0)
        tau2[f], kap[f] = max(t, 1e-6), k
    return tau2, kap


def peer_baseline(panel: Panel, burn_in: int = 28, min_group: int = 8, shrink_k: float = 10.0,
                  inflation: float = 1.1) -> Baseline:
    X = panel.X
    M, T, F = X.shape
    burn_in = min(burn_in, T - 2)
    group = _group_keys(panel.merchants, min_group)
    node = panel.merchants["node"].to_numpy()

    med = np.empty_like(X)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        med[idx] = np.nanmedian(X[idx], axis=0, keepdims=True)

    n_day = np.maximum(np.expm1(X[:, :, 0]), 1.0)
    inv_n = np.repeat((1.0 / n_day)[:, :, None], F, axis=2)
    d = X - med
    offset = np.nanmean(d[:, :burn_in], axis=1)
    resid = d[:, :burn_in] - offset[:, None, :]

    tau2 = np.empty((M, F))
    kap = np.empty((M, F))
    lv = np.empty((M, F))
    for nd in np.unique(node):
        n_idx = np.flatnonzero(node == nd)
        t_node, k_node = _fit_noise(resid[n_idx].reshape(-1, F), inv_n[n_idx, :burn_in].reshape(-1, F))
        for g in np.unique(group[n_idx]):
            idx = np.flatnonzero(group == g)
            w = len(idx) / (len(idx) + shrink_k)
            t_g, k_g = _fit_noise(resid[idx].reshape(-1, F), inv_n[idx, :burn_in].reshape(-1, F))
            tau2[idx] = w * t_g + (1 - w) * t_node
            kap[idx] = w * k_g + (1 - w) * k_node
        # honest spread of onboarding levels, net of the noise in estimating them
        var_day = tau2[n_idx][:, None, :] + kap[n_idx][:, None, :] * inv_n[n_idx, :burn_in]
        noise_in_offset = var_day.mean(axis=1) / burn_in
        lv_node = _robust_sd(offset[n_idx], axis=0) ** 2
        for g in np.unique(group[n_idx]):
            idx = np.flatnonzero(group == g)
            w = len(idx) / (len(idx) + shrink_k)
            lv_g = _robust_sd(offset[idx], axis=0) ** 2
            between = w * lv_g + (1 - w) * lv_node - np.median(noise_in_offset, axis=0)
            pos = np.flatnonzero(np.isin(n_idx, idx))
            lv[idx] = np.maximum(between, 0.05 * np.median(noise_in_offset * burn_in, axis=0)) + noise_in_offset[pos]

    var_day = tau2[:, None, :] + kap[:, None, :] * inv_n
    # merchant-specific noise scale (some shops are simply noisier), shrunk toward 1
    c_m = np.nanmean(resid**2 / var_day[:, :burn_in], axis=1)
    c_m = (burn_in * c_m + shrink_k * 2 * 1.0) / (burn_in + shrink_k * 2)
    var_day = var_day * np.maximum(c_m, 0.5)[:, None, :] * inflation**2
    sd_day = np.sqrt(var_day)
    sigma_w = np.sqrt(np.median(var_day, axis=1))
    sigma_lv = np.sqrt(lv) * inflation
    z_raw = d / sd_day

    # Recursive (one-step-ahead) residuals against the merchant's own running level.
    # For an honest merchant with a stable level these are independent N(0, 1), exactly,
    # so evidence built from them cannot drift upward by chance.
    w = 1.0 / var_day
    cw = np.cumsum(np.nan_to_num(w), axis=1)
    cwd = np.cumsum(np.nan_to_num(w * d), axis=1)
    z = np.full_like(X, np.nan)
    prev_w = cw[:, burn_in - 1:T - 1]
    prev_mean = cwd[:, burn_in - 1:T - 1] / prev_w
    z[:, burn_in:] = (d[:, burn_in:] - prev_mean) / np.sqrt(var_day[:, burn_in:] + 1.0 / prev_w)
    u = offset / sigma_lv
    return Baseline(group=group, med=med, offset=offset, tau2=tau2, kappa=kap, sd_day=sd_day,
                    sigma_w=sigma_w, sigma_lv=sigma_lv, z_raw=z_raw,
                    z=np.clip(z, -3.5, 3.5), u=np.clip(u, -4, 4), burn_in=burn_in)

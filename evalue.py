"""Running e-values (anytime-valid evidence).

For an honest merchant, E_t is a non-negative martingale with mean 1, so by
Ville's inequality  P(E_t ever reaches 1/alpha) <= alpha.  That is what turns
the error budgets into thresholds:  alpha = budget / 1e6.

Benign changes (growth, decline, an auxiliary or adjacent line) are removed
before testing: each day's deviation is projected onto the directions that no
benign explanation can produce. Whatever path a benign change takes (a ramp,
a step, anything), the projected deviation of an honest merchant stays
N(0, P), so the evidence stays exactly valid. Only movement that benign
explanations cannot produce (round amounts, late nights, new payers, a few
dominant payers together) builds evidence.

Numerator: a mixture over illicit signatures (betting, mule) and sizes.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..nodes import BENIGN, ILLICIT, load_all
from ..panel import Panel
from .peers import Baseline

SCALES = np.array([0.25, 0.5, 1.0, 1.5, 2.0])


def complement_projector(D: np.ndarray) -> np.ndarray:
    """I - Q Q' for the column span of D, batched. D [..., F, k] -> [..., F, F]."""
    Q, _ = np.linalg.qr(D)
    F = D.shape[-2]
    return np.eye(F) - Q @ np.swapaxes(Q, -1, -2)


def _mixture_logE(z: np.ndarray, mus: np.ndarray) -> np.ndarray:
    """log of mean_k exp(mu_k.z - |mu_k|^2/2). z [m,F], mus [m,C,F] -> [m]."""
    lin = np.einsum("mcf,mf->mc", mus, z) - 0.5 * (mus**2).sum(axis=2)
    return logsumexp(lin, axis=1) - np.log(mus.shape[1])


def _benign_raw(nd):
    # decline is the mirror of growth, so the span uses growth and adjacent only
    return np.stack([nd.signature_raw(k) for k in BENIGN if k != "decline"], axis=1)


def e_values(panel: Panel, b: Baseline, return_parts: bool = False):
    """Returns log E_level [M], log E_change [M, T], log E total [M, T] (0 before burn-in)."""
    nodes = load_all()
    M, T, F = b.z.shape
    node_of = panel.merchants["node"].to_numpy()
    logE_level = np.zeros(M)
    logE_change = np.zeros((M, T))
    for name in np.unique(node_of):
        idx = np.flatnonzero(node_of == name)
        nd = nodes[name]
        m = len(idx)
        dB = _benign_raw(nd)                                                # F x kb
        dI = np.stack([nd.signature_raw(k) for k in ILLICIT], axis=0)       # K x F

        # ---- onboarding level (one observation) ------------------------------
        slv = b.sigma_lv[idx]
        P = complement_projector(dB[None] / slv[:, :, None])               # m x F x F
        u = np.einsum("mfg,mg->mf", P, np.nan_to_num(b.u[idx]))
        dir_ = np.einsum("mfg,mkg->mkf", P, dI[None] / slv[:, None, :])
        dir_ /= np.linalg.norm(dI[None] / slv[:, None, :], axis=2, keepdims=True)
        mus = (SCALES[None, :, None, None] * dir_[:, None]).reshape(m, -1, F)
        logE_level[idx] = _mixture_logE(u, mus)

        # ---- change after burn-in (sequential) -------------------------------
        unit = np.linalg.norm(dI[None] / b.sigma_w[idx][:, None, :], axis=2)   # m x K
        acc = np.zeros((m, len(SCALES) * len(ILLICIT)))
        for t in range(b.burn_in, T):
            sd = b.sd_day[idx, t] * np.sqrt(1 + 1 / b.burn_in)
            P = complement_projector(dB[None] / sd[:, :, None])
            zt = np.einsum("mfg,mg->mf", P, np.nan_to_num(b.z[idx, t]))
            dir_ = np.einsum("mfg,mkg->mkf", P, dI[None] / sd[:, None, :]) / unit[:, :, None]
            mu_t = (SCALES[None, :, None, None] * dir_[:, None]).reshape(m, -1, F)
            acc += np.einsum("mcf,mf->mc", mu_t, zt) - 0.5 * (mu_t**2).sum(axis=2)
            logE_change[idx, t] = logsumexp(acc, axis=1) - np.log(acc.shape[1])
    logE = logE_change + np.where(np.arange(T)[None] >= b.burn_in, logE_level[:, None], 0.0)
    return logE_level, logE_change, logE


def lifetime_days(alpha: float, D: float) -> float:
    """Expected days for a front to reach a rung: T ~ ln(1/alpha) / D, D = KL divergence per day."""
    return np.log(1 / alpha) / D

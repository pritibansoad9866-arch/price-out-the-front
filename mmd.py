"""What moved? Kernel MMD per channel (amount, timing, payers)."""
from __future__ import annotations

import numpy as np

from ..nodes import CHANNELS


def _rbf(a, b, gamma):
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


def mmd2(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    z = np.vstack([x, y])
    if gamma is None:
        d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        med = np.median(d[d > 0]) if (d > 0).any() else 1.0
        gamma = 1.0 / med
    kxx, kyy, kxy = _rbf(x, x, gamma), _rbf(y, y, gamma), _rbf(x, y, gamma)
    n, m = len(x), len(y)
    return (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()


def mmd_test(x, y, n_perm: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = mmd2(x, y)
    z = np.vstack([x, y])
    n = len(x)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(z))
        cnt += mmd2(z[p[:n]], z[p[n:]]) >= obs
    return obs, (cnt + 1) / (n_perm + 1)


def what_moved(z_raw: np.ndarray, merchant: int, peers: np.ndarray, t: int, window: int = 14, seed: int = 0):
    """Compare a merchant's recent window with peers' same window, channel by channel."""
    lo = max(0, t - window + 1)
    out = {}
    rng = np.random.default_rng(seed)
    peers = peers[peers != merchant]
    pick = rng.choice(peers, size=min(4, len(peers)), replace=False) if len(peers) else peers
    for ch, cols in CHANNELS.items():
        x = np.nan_to_num(z_raw[merchant, lo:t + 1][:, cols])
        y = np.nan_to_num(z_raw[pick, lo:t + 1][:, :, cols].reshape(-1, len(cols)))
        if len(x) < 3 or len(y) < 3:
            out[ch] = (np.nan, 1.0)
            continue
        out[ch] = mmd_test(x, y, n_perm=150, seed=seed)
    return out

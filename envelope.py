"""Volume envelope: a bakery can only bake so much.

Every business type and scale band gets a monthly volume an unverified
merchant can settle at normal speed. Inside it, a front only earns what
that business would. Above it, extra volume settles slower and stands out.
"""
from __future__ import annotations

import numpy as np

from ..config import Config


def monthly_envelope(scale: str, verified: bool, cfg: Config, tenure_months: float = 0.0) -> float:
    base = cfg.envelope.unverified[scale]
    if not verified:
        return base
    # Verified headroom grows only as fast as what a member would forfeit (vesting),
    # so an exploit never pays more than it costs: G(tau) < F(tau).
    pr = cfg.programme
    headroom = (cfg.envelope.verified_multiplier - 1) * base
    vested_share = min(tenure_months / pr.vest_months, 1.0)
    return base + headroom * vested_share


def over_envelope_share(volume: np.ndarray, envelope: float) -> np.ndarray:
    volume = np.asarray(volume, dtype=float)
    return np.where(volume > envelope, (volume - envelope) / np.maximum(volume, 1), 0.0)

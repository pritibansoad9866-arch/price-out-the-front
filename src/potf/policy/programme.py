"""Opt-in verified programme: the carrot.

Benefits are told upfront: faster settlement, a wider envelope and fee rebates.
Rebates vest slowly and headroom grows with tenure, because incentives that
eventually reduce scrutiny can encourage behaviour that is ideal upfront and
then exploitative later on. Scrutiny never fully drops for members.

No-exploit condition (Shapiro-Stiglitz style): at every tenure tau, the gain
from exploiting G(tau) must be below what is forfeited F(tau).
"""
from __future__ import annotations

import numpy as np

from ..config import Config

BENEFITS_UPFRONT = [
    "Settlement the same day instead of T+1",
    "A wider monthly volume envelope that grows with tenure",
    "Fee rebates every month, vesting over 12 months",
    "Fewer routine prompts (scrutiny is lighter, never zero)",
]

LOW_FRICTION_STEPS = [
    "Upload one GST invoice a month that matches settled value",
    "Geo-tagged shop photo once a quarter",
    "Keep a share of repeat local payers similar to peers",
]


def no_exploit_curve(cfg: Config, months: int = 24, spread: float = 0.02):
    """Returns tenure, G(tau) (exploit gain) and F(tau) (forfeit) for one month of exploitation.

    G: extra headroom available to an exploiter x what moving it through PayU saves vs the
    alternative channel (spread). F: stake + vested rebates + one month of future rebates.
    """
    from .envelope import monthly_envelope

    tau = np.arange(months + 1)
    base = cfg.envelope.unverified["small"]
    G = np.array([(monthly_envelope("small", True, cfg, t) - base) * spread for t in tau])
    F = np.array([cfg.programme.forfeit(t) + cfg.programme.rebate_per_month for t in tau])
    return tau, G, F


def scrutiny_multiplier(verified: bool, vest: bool) -> float:
    """How much detection hazard members keep. With vesting design, scrutiny never fully drops."""
    if not verified:
        return 1.0
    return 0.8 if vest else 0.2

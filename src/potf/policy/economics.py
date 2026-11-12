"""Front economics: cost per rupee moved through PayU.

    c = (A + k) / (V * T)  +  h * rho * (hold days / lifetime)  +  o  +  envelope friction

A + k : cost to acquire and dress one front account
V     : monthly volume per front,  T : expected lifetime (months)
h, rho: share frozen when a hold lands, and the chance one lands
o     : operating cost (mule cuts, cash-out)

The syndicate stays only while c < c_alt (its next-best channel, ~2-3%).
Our win line is c > 1.5 x c_alt (~4.5%), with friction for normal businesses ~0.
Detection hazard D (evidence per day) rises with how far a front sits from its
peers, and with fleet size when fleets are linked: lambda(N) = lambda0 (1 + N / N0).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..config import LADDER, RUNG_INDEX, Config
from .envelope import monthly_envelope, over_envelope_share
from .programme import scrutiny_multiplier

HORIZON_DAYS = 365.0
V_PEER = 3e5            # monthly volume of a typical small kirana (Rs)
D_AUDIT = 0.003         # status quo: random audits and complaints, per day
D_SIGNAL = 0.02         # evidence per day for a front at 1x peer volume of illicit flow
HOLD_DAYS = 10.0        # days a hold stays on before review resolves


@dataclass
class Levers:
    envelope: bool = False   # volume envelope per business type
    ladder: bool = False     # e-value evidence + graded ladder (holds before exit)
    link: bool = False       # fleet graph (payer overlap)
    rotate: bool = False     # randomised, rotating thresholds (no probing)
    vest: bool = False       # slow-vesting rebates; scrutiny never fully drops

    def as_dict(self):
        return asdict(self)


def detection_rate(V: float, n_accounts: float, lv: Levers, probing: bool = False, verified: bool = False) -> float:
    """Evidence per day D for one front moving V a month, in a fleet of n_accounts."""
    if not lv.ladder:
        return D_AUDIT
    D = D_AUDIT + D_SIGNAL * (V / V_PEER) ** 2
    if lv.link:
        D *= 1 + n_accounts / 5.0
    if probing and not lv.rotate:
        D *= 0.15          # a front that has learnt fixed thresholds sits just under them
    D *= scrutiny_multiplier(verified, lv.vest)
    return D


def front_cost(V: float, M: float, lv: Levers, cfg: Config, probing: bool = False,
               verified: bool = False, extra_fixed: float = 0.0) -> dict:
    """Cost per Rs for a syndicate moving M a month using fronts of V a month each."""
    e = cfg.economics
    N = max(M / V, 1.0)
    D = detection_rate(V, N, lv, probing, verified)
    alpha_exit = LADDER[RUNG_INDEX["exit"]].alpha
    t_exit = np.log(1 / alpha_exit) / D if lv.ladder else 1 / D
    rho = 1 - np.exp(-HORIZON_DAYS / t_exit)
    life_days = t_exit * rho                       # E[min(detection, horizon)]
    fixed = (e.account_cost + e.setup_cost + extra_fixed) / (V * life_days / 30.0)
    hold = e.hold_share * rho * min(HOLD_DAYS / life_days, 1.0) if lv.ladder else 0.15 * rho * min(3 / life_days, 1)
    env_cost = 0.0
    if lv.envelope:
        env = monthly_envelope("small", verified, cfg, tenure_months=3 if verified else 0)
        env_cost = float(over_envelope_share(V, env)) * 0.03
    total = fixed + hold + e.operating + env_cost
    return {"V": V, "N": N, "D": D, "life_days": life_days, "fixed": fixed, "hold": hold,
            "operating": e.operating, "envelope": env_cost, "total": total}


def best_volume(M: float, lv: Levers, cfg: Config, lo: float = 2e4, hi: float = 3e7, **kw) -> dict:
    grid = np.geomspace(lo, hi, 200)
    costs = [front_cost(v, M, lv, cfg, **kw) for v in grid]
    return min(costs, key=lambda c: c["total"])


def cost_curve(M: float, lv: Levers, cfg: Config, volumes=None, **kw):
    volumes = np.geomspace(2e4, 3e7, 120) if volumes is None else volumes
    return [front_cost(v, M, lv, cfg, **kw) for v in volumes]


def legit_friction(rung_share: dict) -> float:
    """Expected friction per Rs for a normal business, given the share of honest merchants on each rung."""
    return float(sum(LADDER[RUNG_INDEX[k]].friction * s for k, s in rung_share.items()))
